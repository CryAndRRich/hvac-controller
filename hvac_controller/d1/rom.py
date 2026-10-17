"""Step B of direction 1: a reduced-order model (ROM) of the thermal core, and how well it replays the original.

The organisers' step (step A, ``core.core_step``) is backward Euler on the augmented system x = (T, T_m) of the
solved CVs and their interior mass nodes,

    (E + K) x' = E x + F w,      E = diag(C / dt, C_m / dt),   K symmetric positive definite,

with w = (heat per zone, air power per VAV, outside temperature). Their solver eliminates the mass nodes into the
diagonal of its matrix A and solves the air power implicitly, p = G (T_supply - Q^T T'), by a V x V Woodbury
system; ``FullModel`` repeats exactly that arithmetic from the arrays ``core.save_core`` wrote.

The ROM is a Galerkin projection x ~ V x_r with V orthonormal in the E inner product, so its step
(I + V^T K V) x_r' = x_r + V^T F w is again backward Euler on an SPD matrix and stable at any order. V is a block
rational Krylov basis, (sigma E + K)^-1 F for a few shifts sigma in units of 1/dt:

- sigma = 1 is the step itself: the basis holds A^-1 P, so the one-step response (and the air-side matrix
  S = Q^T A^-1 P) is exact;
- sigma = 0 is steady state: every DC gain is exact;
- sigma = 1/12 and 1/288 cover hours and a day (walls, mass).

The reduced matrices are diagonalised (modal form: one decay factor per state), then cut down either by modal
dominance or by balanced truncation (closed-form Gramians in modal coordinates). Dynamics that are cut keep their DC
gain through a feedthrough term, so a truncated ROM is still exact at steady state.

Everything runs in deviations from a reference temperature T_ref: the conduction rows of K sum to the ambient
conductance, so a uniform shift of the field equals shifting the outside and supply temperatures by the same amount.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

PERMC = "MMD_AT_PLUS_A"  # the organisers' column ordering
BLOCK = 64  # right-hand sides per triangular solve
# (shift in 1/dt, number of moments); sigma = 1 first so that A^-1 F is never deflated.
DEFAULT_SHIFTS = ((1.0, 2), (0.0, 1), (1.0 / 12.0, 1), (1.0 / 288.0, 1))
DEFAULT_ORDERS = (400, 200, 100, 50, 25)
CHECKPOINTS = (12, 288)


# --------------------------------------------------------------------------------------------- the full linear core

@dataclass
class Core:
    """One (building, h) thermal core as ``core.save_core`` wrote it; everything indexed by solved CV."""

    a: scipy.sparse.csc_matrix  # A(h): the organisers' step matrix, mass eliminated
    cap: np.ndarray  # C / dt (W/K)
    amb: np.ndarray  # conductance to ambient: pinned exterior neighbours + exterior convection (W/K)
    mass_coupling: np.ndarray
    mass_retained: np.ndarray
    mass_gained: np.ndarray
    applied: scipy.sparse.csc_matrix  # P (n x V): share of each VAV's air power reaching each CV
    sensed: scipy.sparse.csr_matrix  # Q^T (V x n): weights of the mean each VAV reads
    zones: scipy.sparse.csc_matrix  # U (n x Z): 1/n_z on each CV of zone z
    temp0: np.ndarray
    mass0: np.ndarray
    zone_names: list[str]
    h: float

    @property
    def n(self) -> int:
        return self.a.shape[0]


def load_core(core_dir: Path) -> Core:
    d = Path(core_dir)
    v = np.load(d / "vectors.npz")
    return Core(
        a=scipy.sparse.load_npz(d / "matrix.npz").tocsc(),
        cap=v["capacitance_rate"], amb=v["dirichlet"] + v["exposure"],
        mass_coupling=v["mass_coupling"], mass_retained=v["mass_retained"], mass_gained=v["mass_gained"],
        applied=scipy.sparse.load_npz(d / "applied.npz").tocsc(),
        sensed=scipy.sparse.load_npz(d / "sensed.npz").tocsr(),
        zones=scipy.sparse.load_npz(d / "zones.npz").tocsc(),
        temp0=v["temp0"], mass0=v["mass0"], zone_names=[str(z) for z in v["zone_names"]], h=float(v["h"]),
    )


def load_sequence(core_dir: Path) -> dict[str, np.ndarray]:
    s = np.load(Path(core_dir) / "sequence.npz")
    return {k: s[k] for k in s.files}


def solve_block(lu, rhs) -> np.ndarray:
    """lu.solve on many right-hand sides, BLOCK at a time (dense result)."""
    n_cols = rhs.shape[1]
    out = np.empty((rhs.shape[0], n_cols))
    for start in range(0, n_cols, BLOCK):
        chunk = rhs[:, start:start + BLOCK]
        chunk = chunk.toarray() if scipy.sparse.issparse(chunk) else np.asarray(chunk)
        out[:, start:start + BLOCK] = lu.solve(np.asfortranarray(chunk))
    return out


class FullModel:
    """The exact core: the same arithmetic as FiniteDifferenceSolver.solve (verified in step A)."""

    def __init__(self, core: Core, lu=None):
        self.core = core
        self.lu = lu if lu is not None else scipy.sparse.linalg.splu(core.a, permc_spec=PERMC)
        self.s = np.asarray(core.sensed @ solve_block(self.lu, core.applied))  # Q^T A^-1 P (V x V)

    def initial(self):
        return self.core.temp0.copy(), self.core.mass0.copy()

    def step(self, t, m, q_zone, t_amb, g, ts):
        c = self.core
        forcing = c.cap * t + c.zones @ q_zone + t_amb * c.amb + c.mass_coupling * m
        y = self.lu.solve(forcing)
        p = np.linalg.solve(np.eye(len(g)) + g[:, None] * self.s, g * (ts - c.sensed @ y))
        t = y + self.lu.solve(c.applied @ p)
        m = m * c.mass_retained + c.mass_gained * t
        return t, m

    def outputs(self, t) -> np.ndarray:
        """(sensed per VAV, mean per zone)."""
        return np.concatenate([self.core.sensed @ t, self.core.zones.T @ t])

    def replay(self, seq: dict, n_steps: int) -> np.ndarray:
        t, m = self.initial()
        out = []
        for k in range(n_steps):
            t, m = self.step(t, m, seq["q_zone"][k], seq["t_amb"][k], seq["g"][k], seq["ts"][k])
            out.append(self.outputs(t))
        return np.asarray(out)


# ------------------------------------------------------------------------------------------ the augmented system

class Augmented:
    """x = (T, T_m) with E = diag(cap, C_m / dt) and K = [[K_air + G_m, -G_m], [-G_m, G_m]].

    The organisers keep C_m / dt and G_m only through mass_coupling = (C_m/dt) G_m / (C_m/dt + G_m) and the
    weights mass_retained / mass_gained; both are recovered from those.
    """

    def __init__(self, core: Core):
        self.core = core
        self.n = core.n
        self.mi = np.flatnonzero(core.mass_coupling > 0.0)
        mc = core.mass_coupling[self.mi]
        self.g_m = mc / core.mass_retained[self.mi]
        self.c_m = mc / core.mass_gained[self.mi]
        self.N = self.n + len(self.mi)
        self.e = np.concatenate([core.cap, self.c_m])
        self.sqrt_e = np.sqrt(self.e)
        self.k_air = (core.a - scipy.sparse.diags(core.cap + core.mass_coupling)).tocsr()

    def k_mul(self, x: np.ndarray) -> np.ndarray:
        n, mi, g = self.n, self.mi, self.g_m[:, None]
        xt, xm = x[:n], x[n:]
        out_t = np.asarray(self.k_air @ xt)
        diff = xt[mi] - xm
        out_t[mi] += g * diff
        return np.vstack([out_t, -g * diff])

    def shifted_solver(self, sigma: float):
        """b -> (sigma E + K)^-1 b, with the mass rows eliminated as the organisers do."""
        n, mi = self.n, self.mi
        dm = sigma * self.c_m + self.g_m
        if sigma == 1.0:
            a_s = self.core.a
        else:
            extra = np.zeros(n)
            extra[mi] = sigma * self.c_m * self.g_m / dm
            a_s = (self.k_air + scipy.sparse.diags(sigma * self.core.cap + extra)).tocsc()
        lu = scipy.sparse.linalg.splu(a_s, permc_spec=PERMC)

        def solve(b: np.ndarray) -> np.ndarray:
            bt = np.array(b[:n], dtype=np.float64)
            bm = b[n:]
            bt[mi] += (self.g_m / dm)[:, None] * bm
            zt = solve_block(lu, bt)
            zm = (bm + self.g_m[:, None] * zt[mi]) / dm[:, None]
            return np.vstack([zt, zm])

        return solve, lu

    def inputs(self) -> scipy.sparse.csc_matrix:
        """F (N x (Z + V + 1)): columns [zone heat | VAV air power | outside temperature]."""
        c = self.core
        top = scipy.sparse.hstack([c.zones, c.applied, scipy.sparse.csc_matrix(c.amb[:, None])])
        return scipy.sparse.vstack([top, scipy.sparse.csc_matrix((len(self.mi), top.shape[1]))]).tocsc()

    def state0(self, t_ref: float) -> np.ndarray:
        return np.concatenate([self.core.temp0, self.core.mass0[self.mi]]) - t_ref


# --------------------------------------------------------------------------------------------------- the basis

class Basis:
    """Columns orthonormal in the E inner product, stored scaled: B = E^{1/2} V has Euclidean-orthonormal columns."""

    def __init__(self, sqrt_e: np.ndarray, capacity: int):
        self.sqrt_e = sqrt_e
        self.b = np.empty((len(sqrt_e), capacity))
        self.r = 0
        self.dropped = 0

    def _project_out(self, w):
        if self.r:
            cur = self.b[:, :self.r]
            w -= cur @ (cur.T @ w)
        return w

    def extend(self, z: np.ndarray, tol: float) -> int:
        """Adds the part of span(z) not yet in the basis; drops directions below tol (relative)."""
        w = z * self.sqrt_e[:, None]
        norms = np.linalg.norm(w, axis=0)
        w = w[:, norms > 0.0] / norms[norms > 0.0]
        for _ in range(2):
            w = self._project_out(w)
        mu, vec = np.linalg.eigh(w.T @ w)
        keep = mu > tol ** 2
        self.dropped += int(np.sum(~keep)) + int(np.sum(norms == 0.0))
        w = w @ (vec[:, keep] / np.sqrt(mu[keep]))
        w = self._project_out(w)  # second pass: orthogonal to working precision
        mu, vec = np.linalg.eigh(w.T @ w)
        keep = mu > 1e-2  # unit vectors already; anything that lost most of its length here is rounding noise
        self.dropped += int(np.sum(~keep))
        w = w @ (vec[:, keep] / np.sqrt(mu[keep]))
        k = w.shape[1]
        self.b[:, self.r:self.r + k] = w
        self.r += k
        return k

    @property
    def cols(self) -> np.ndarray:
        return self.b[:, :self.r]

    def v_chunks(self, size: int = 256):
        """(slice, V[:, slice]) without building all of V at once."""
        for start in range(0, self.r, size):
            sl = slice(start, min(start + size, self.r))
            yield sl, self.b[:, sl] / self.sqrt_e[:, None]


def build_basis(aug: Augmented, f: scipy.sparse.csc_matrix, x0: np.ndarray, shifts=DEFAULT_SHIFTS,
                tol: float = 1e-6, log=print) -> tuple[Basis, dict]:
    rhs = f.toarray()
    if np.linalg.norm(x0 * aug.sqrt_e) > 0.0:
        rhs = np.hstack([rhs, (aug.e * x0)[:, None]])  # the initial state acts like an input at step 0
    basis = Basis(aug.sqrt_e, rhs.shape[1] * sum(d for _, d in shifts))
    info = {"blocks": []}
    for sigma, depth in shifts:
        t0 = time.time()
        solve, lu = aug.shifted_solver(sigma)
        t_lu = time.time() - t0
        z = rhs
        for j in range(depth):
            z = solve(z if j == 0 else aug.e[:, None] * z)
            added = basis.extend(z, tol)
            info["blocks"].append({"sigma": sigma, "moment": j + 1, "added": added, "r": basis.r,
                                   "lu_s": round(t_lu, 2), "s": round(time.time() - t0, 2)})
            log(f"  basis sigma={sigma:.5f} moment={j + 1}: +{added} -> r={basis.r} ({time.time() - t0:.1f}s)")
        del lu, solve
    info["dropped"] = basis.dropped
    return basis, info


# ------------------------------------------------------------------------------------------------------- the ROM

@dataclass
class LinearRom:
    """x' = A x + B w,  y' = C x' + D w,  with w = [zone heat (Z) | air power (V) | T_out - T_ref (1)].

    y = [sensed per VAV (V) | mean per zone (Z)], in deviations from t_ref. A is a vector when diagonal (modal).
    """

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    x0: np.ndarray
    t_ref: float
    n_zones: int
    n_vavs: int
    scales: np.ndarray
    name: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        z, v = self.n_zones, self.n_vavs
        self._iq, self._ip, self._ia = slice(0, z), slice(z, z + v), z + v
        self._bp = self.b[:, self._ip]
        self._np = self.c @ self._bp + self.d[:, self._ip]  # output response to air power, (V + Z) x V
        self._m = self._np[:v]  # the reduced S = Q^T A^-1 P

    @property
    def order(self) -> int:
        return len(self.x0)

    def _amul(self, x):
        return self.a * x if self.a.ndim == 1 else self.a @ x

    def step(self, x, q_zone, t_amb, g, ts):
        """One step with the air power solved implicitly, as the organisers do; returns (x', y' + t_ref)."""
        da = t_amb - self.t_ref
        x_pre = self._amul(x) + self.b[:, self._iq] @ q_zone + self.b[:, self._ia] * da
        y_pre = self.c @ x_pre + self.d[:, self._iq] @ q_zone + self.d[:, self._ia] * da
        v = self.n_vavs
        p = np.linalg.solve(np.eye(v) + g[:, None] * self._m, g * ((ts - self.t_ref) - y_pre[:v]))
        return x_pre + self._bp @ p, y_pre + self._np @ p + self.t_ref

    def replay(self, seq: dict, n_steps: int) -> np.ndarray:
        x = self.x0.copy()
        out = []
        for k in range(n_steps):
            x, y = self.step(x, seq["q_zone"][k], seq["t_amb"][k], seq["g"][k], seq["ts"][k])
            out.append(y)
        return np.asarray(out)

    def step_gain(self) -> np.ndarray:
        """Response of y' to w in the same step (first Markov parameter)."""
        return self.c @ self.b + self.d

    def dc_gain(self) -> np.ndarray:
        if self.a.ndim == 1:
            return self.c @ (self.b / (1.0 - self.a)[:, None]) + self.d
        return self.c @ np.linalg.solve(np.eye(self.order) - self.a, self.b) + self.d

    def save(self, path: Path) -> None:
        np.savez_compressed(path, a=self.a, b=self.b, c=self.c, d=self.d, x0=self.x0, t_ref=self.t_ref,
                            n_zones=self.n_zones, n_vavs=self.n_vavs, scales=self.scales, name=self.name,
                            **{f"extra_{k}": np.asarray(v) for k, v in self.extra.items()})


def reduce(aug: Augmented, basis: Basis, f: scipy.sparse.csc_matrix, x0: np.ndarray) -> dict[str, np.ndarray]:
    """Galerkin matrices: E_r (should be I), K_r, F_r, outputs C_r, initial state."""
    r, n = basis.r, aug.n
    b = basis.cols
    k_r = np.empty((r, r))
    for sl, v in basis.v_chunks():
        k_r[:, sl] = b.T @ (aug.k_mul(v) / aug.sqrt_e[:, None])
    f_r = np.asarray((scipy.sparse.diags(1.0 / aug.sqrt_e) @ f).T @ b).T
    x0_r = b.T @ (aug.sqrt_e * x0)
    c = aug.core
    out_t = scipy.sparse.vstack([c.sensed, c.zones.T]).tocsr() @ scipy.sparse.diags(1.0 / aug.sqrt_e[:n])
    c_r = np.asarray(out_t @ b[:n])
    return {"k": 0.5 * (k_r + k_r.T), "k_asym": float(np.abs(k_r - k_r.T).max() / np.abs(k_r).max()),
            "e_err": float(np.abs(b.T @ b - np.eye(r)).max()), "f": f_r, "c": c_r, "x0": x0_r}


def modal_rom(red: dict, t_ref: float, n_zones: int, n_vavs: int, scales: np.ndarray) -> LinearRom:
    lam, w = np.linalg.eigh(red["k"])
    lam_raw_min = float(lam.min())
    lam = np.maximum(lam, 1e-12)  # K_r is SPD; a clipped value means rounding, and lambda_raw_min reports it
    a = 1.0 / (1.0 + lam)
    b = a[:, None] * (w.T @ red["f"])
    return LinearRom(a=a, b=b, c=red["c"] @ w, d=np.zeros((red["c"].shape[0], b.shape[1])), x0=w.T @ red["x0"],
                     t_ref=t_ref, n_zones=n_zones, n_vavs=n_vavs, scales=scales, name=f"modal_{len(a)}",
                     extra={"lambda": lam, "lambda_raw_min": lam_raw_min})


def modal_truncate(full: LinearRom, k: int) -> LinearRom:
    """Keep the k most dominant modes (DC contribution, scaled inputs); the rest become a DC feedthrough."""
    one_minus = 1.0 - full.a
    score = np.linalg.norm(full.c, axis=0) * np.linalg.norm(full.b * full.scales, axis=1) / one_minus
    keep = np.sort(np.argsort(score)[::-1][:k])
    drop = np.setdiff1d(np.arange(full.order), keep)
    d = full.d + full.c[:, drop] @ (full.b[drop] / one_minus[drop, None])
    return LinearRom(a=full.a[keep], b=full.b[keep], c=full.c[:, keep], d=d, x0=full.x0[keep], t_ref=full.t_ref,
                     n_zones=full.n_zones, n_vavs=full.n_vavs, scales=full.scales, name=f"modal_{k}")


def _psd_sqrt(m: np.ndarray) -> np.ndarray:
    mu, vec = np.linalg.eigh(0.5 * (m + m.T))
    return vec * np.sqrt(np.maximum(mu, 0.0))


def hankel(full: LinearRom):
    """Square-root factors and Hankel singular values of the modal ROM (scaled inputs)."""
    aa = 1.0 - np.outer(full.a, full.a)
    bs = full.b * full.scales
    lp = _psd_sqrt((bs @ bs.T) / aa)
    lq = _psd_sqrt((full.c.T @ full.c) / aa)
    u, s, vt = scipy.linalg.svd(lq.T @ lp, full_matrices=False)
    return lp, lq, u, s, vt


def balanced_truncate(full: LinearRom, k: int, factors, keep: str = "dc") -> LinearRom:
    """Square-root balanced truncation; the feedthrough then keeps either the steady state ("dc") or the one-step
    response ("step": the air-side S and every input's first-step effect) exact."""
    lp, lq, u, s, vt = factors
    k = min(k, int(np.sum(s > s[0] * 1e-14)))
    root = 1.0 / np.sqrt(s[:k])
    t = lp @ vt[:k].T * root
    ti = (root[:, None] * u[:, :k].T) @ lq.T
    rom = LinearRom(a=ti @ (full.a[:, None] * t), b=ti @ full.b, c=full.c @ t, d=full.d.copy(), x0=ti @ full.x0,
                    t_ref=full.t_ref, n_zones=full.n_zones, n_vavs=full.n_vavs, scales=full.scales,
                    name=f"balanced_{keep}_{k}", extra={"hsv_bound": 2.0 * float(np.sum(s[k:]))})
    if keep == "dc":
        rom.d = rom.d + (full.dc_gain() - rom.dc_gain())
    else:
        rom.d = rom.d + (full.step_gain() - rom.step_gain())
    rom.__post_init__()
    return rom


# -------------------------------------------------------------------------------------------- inputs to replay

def input_scales(seq: dict, t_ref: float) -> np.ndarray:
    """Typical size of each input in the logged rollout (W, W, K), for ranking and balancing."""
    q = np.max(np.abs(seq["q_zone"]), axis=0)
    p = np.max(np.abs(seq["g"] * (seq["ts"] - seq["sensed_free"])), axis=0)
    a = np.max(np.abs(seq["t_amb"] - t_ref))
    return np.maximum(np.concatenate([q, p, [a]]), 1.0)


def random_sequence(seq: dict, n_steps: int, seed: int) -> dict[str, np.ndarray]:
    """Random excitation around the logged ranges: piecewise-constant heat, flows and supply temperatures, with
    building-wide air-off blocks (nights), and the logged outside temperature."""
    rng = np.random.default_rng(seed)
    n_z, n_v = seq["q_zone"].shape[1], seq["g"].shape[1]
    q_max = np.maximum(np.max(seq["q_zone"], axis=0), 100.0) * 1.5
    g_max = np.max(seq["g"], axis=0)
    g_max = np.maximum(g_max, 0.2 * g_max.max() + 1.0)  # VAVs the baseline never opened still get some air

    def held(shape, low, high, p_switch=1 / 6):
        x = rng.uniform(low, high, size=shape)
        out = np.empty((n_steps,) + shape)
        for k in range(n_steps):
            if k:
                switch = rng.random(shape) < p_switch
                x = np.where(switch, rng.uniform(low, high, size=shape), x)
            out[k] = x
        return out

    on = np.zeros(n_steps, dtype=bool)
    k, state = 0, True
    while k < n_steps:
        length = int(rng.integers(36, 145))
        on[k:k + length] = state
        k, state = k + length, not state
    t_amb = np.resize(seq["t_amb"], n_steps)
    return {"q_zone": held((n_z,), 0.0, 1.0) * q_max, "g": held((n_v,), 0.0, 1.0) * g_max * on[:, None],
            "ts": held((n_v,), 285.0, 300.0), "t_amb": t_amb, "share_air_off": float(1.0 - on.mean())}


def error_stats(pred: np.ndarray, ref: np.ndarray) -> dict:
    e = np.abs(pred - ref)
    out = {}
    for k in CHECKPOINTS + (len(e),):
        k = min(k, len(e))
        out[f"{k}_steps"] = {"mae": float(e[:k].mean()), "max": float(e[:k].max())}
    return out


def time_steps(rom: LinearRom, seq: dict, n: int = 500) -> float:
    x = rom.x0.copy()
    t0 = time.perf_counter()
    for k in range(n):
        x, _ = rom.step(x, seq["q_zone"][k], seq["t_amb"][k], seq["g"][k], seq["ts"][k])
    return (time.perf_counter() - t0) / n * 1e6


# ------------------------------------------------------------------------------------------------------ the check

def run_rom_check(core_dir: Path, out_dir: Path, shifts=DEFAULT_SHIFTS, orders=DEFAULT_ORDERS, n_check: int = 288,
                  n_random: int = 1152, seed: int = 0, tol: float = 1e-6, log=print) -> dict:
    """Build the ROM of one saved core, replay it against the original, save ROMs and a report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    timings = {}
    core, seq = load_core(core_dir), load_sequence(core_dir)
    n_steps = len(seq["t_amb"])
    n_zones, n_vavs = len(core.zone_names), core.applied.shape[1]
    t_ref = float(np.mean(core.temp0))

    t0 = time.time()
    full = FullModel(core)
    timings["full_setup_s"] = time.time() - t0
    log(f"[rom] {core_dir.name}: n={core.n} zones={n_zones} vavs={n_vavs} steps={n_steps}")

    # 0. Our copy of the core, from the saved arrays, against the swap-free replay logged in step A.
    t0 = time.time()
    n_check = min(n_check, n_steps)
    full_logged = full.replay(seq, n_check)
    timings["full_step_ms"] = (time.time() - t0) / n_check * 1e3
    loader_err = float(np.abs(full_logged[:, :n_vavs] - seq["sensed_free"][:n_check]).max())
    log(f"  saved core vs step A replay over {n_check} steps: {loader_err:.2e} K")

    # 1. Basis and reduced matrices.
    t0 = time.time()
    aug = Augmented(core)
    f = aug.inputs()
    x0 = aug.state0(t_ref)
    basis, basis_info = build_basis(aug, f, x0, shifts, tol, log)
    timings["basis_s"] = time.time() - t0
    t0 = time.time()
    red = reduce(aug, basis, f, x0)
    del basis
    timings["reduce_s"] = time.time() - t0
    scales = input_scales(seq, t_ref)
    full_rom = modal_rom(red, t_ref, n_zones, n_vavs, scales)
    lam = full_rom.extra["lambda"]
    s_exact = full.s
    s_rom = full_rom._m
    log(f"  r={full_rom.order}  E_r err {red['e_err']:.1e}  K_r asym {red['k_asym']:.1e}  "
        f"S err {np.abs(s_rom - s_exact).max() / np.abs(s_exact).max():.1e} (rel)")

    # 2. Truncations.
    t0 = time.time()
    factors = hankel(full_rom)
    roms = [full_rom]
    for k in orders:
        if k < full_rom.order:
            roms.append(modal_truncate(full_rom, k))
            roms.append(balanced_truncate(full_rom, k, factors, "dc"))
            roms.append(balanced_truncate(full_rom, k, factors, "step"))
    timings["truncate_s"] = time.time() - t0

    # 3. Replays: the logged baseline rollout (vs the swap-free core and vs the real sensors) and a random excitation
    #    (vs the full core, which also gives the zone means).
    t0 = time.time()
    rand = random_sequence(seq, n_random, seed)
    full_rand = full.replay(rand, n_random)
    timings["full_random_s"] = time.time() - t0
    variants = []
    for rom in roms:
        t0 = time.time()
        y_log = rom.replay(seq, n_steps)
        y_rand = rom.replay(rand, n_random)
        variants.append({
            "name": rom.name, "order": rom.order,
            "logged_vs_core": error_stats(y_log[:, :n_vavs], seq["sensed_free"]),
            "logged_vs_sensors": error_stats(y_log[:, :n_vavs], seq["sensed_env"]),
            "random_vs_core_sensed": error_stats(y_rand[:, :n_vavs], full_rand[:, :n_vavs]),
            "random_vs_core_zones": error_stats(y_rand[:, n_vavs:], full_rand[:, n_vavs:]),
            "s_rel_err": float(np.abs(rom._m - s_exact).max() / np.abs(s_exact).max()),
            "dc_gain_rel_err": float(np.abs(rom.dc_gain() - full_rom.dc_gain()).max()
                                     / np.abs(full_rom.dc_gain()).max()),
            "step_us": time_steps(rom, seq, min(500, n_steps)),
            "hsv_bound_scaled": rom.extra.get("hsv_bound"),
            "replay_s": time.time() - t0,
        })
        rom.save(out_dir / f"rom_{rom.name}.npz")
        v, day, lg, rd = variants[-1], f"{min(288, n_steps)}_steps", f"{n_steps}_steps", f"{n_random}_steps"
        log(f"  {rom.name:>14}: logged MAE@24h {v['logged_vs_core'][day]['mae']:.4f} K max {v['logged_vs_core'][lg]['max']:.4f} K"
            f" | random MAE@24h {v['random_vs_core_zones'][f'{min(288, n_random)}_steps']['mae']:.4f} K"
            f" max {v['random_vs_core_zones'][rd]['max']:.4f} K | S {v['s_rel_err']:.1e} | {v['step_us']:.0f} us/step")

    report = {
        "core": str(core_dir), "n_solved": core.n, "n_mass": len(aug.mi), "n_aug": aug.N,
        "n_zones": n_zones, "n_vavs": n_vavs, "h": core.h, "t_ref": t_ref, "steps_logged": n_steps,
        "shifts": [list(s) for s in shifts], "tol": tol, "basis": basis_info,
        "loader_err_k": loader_err, "e_r_err": red["e_err"], "k_r_asym": red["k_asym"],
        "s_rel_err": float(np.abs(s_rom - s_exact).max() / np.abs(s_exact).max()),
        "lambda_raw_min": full_rom.extra["lambda_raw_min"], "lambda_min": float(lam.min()), "lambda_max": float(lam.max()),
        "slowest_time_constant_h": float(1.0 / max(lam.min(), 1e-300) * 300.0 / 3600.0),
        "hsv": factors[3][:60].tolist(), "random_share_air_off": rand["share_air_off"],
        "x0_dev_max_k": float(np.abs(x0).max()),
        "timings": timings, "variants": variants, "seconds": time.time() - t_start,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("core_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--n-random", type=int, default=1152)
    ap.add_argument("--n-check", type=int, default=288)
    a = ap.parse_args()
    run_rom_check(a.core_dir, a.out_dir, n_check=a.n_check, n_random=a.n_random,
                  log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
