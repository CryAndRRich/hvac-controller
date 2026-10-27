"""The differentiable copy of one building: ROM (step B) + plant (plant.py), batched, on NumPy or PyTorch.

``BuildingEnv`` follows the batched interface SHAC-style learners use (DiffRL, rewarped): ``num_envs``, ``num_obs``,
``num_actions``, ``reset`` and ``step(action) -> obs, reward, done, extras``. The action is the competition's hybrid
action: continuous values in [-1, 1] and discrete run commands in {0, 1} (any value in [0, 1] when ``exact=False``;
with 0 / 1 that relaxed plant computes exactly what the exact one does). Continuous values must stay in [-1, 1], as
the organisers require: the air solve's sweep count relies on it.

One environment holds one building and S scenarios of it (same plant; each scenario its own ROM, for its materials
and convection coefficient, and its own exogenous series). Its batch of B = S * b rows is grouped: row i runs
scenario i // b. Every row has its own step index ``k``, so rows can sit at different times of their episode; a row
whose episode ends is put back to the scenario's initial state by ``reset_done``. Nothing in a step reads a tensor
value on the host. With the NumPy backend and S = 1 it is the validation replica that env_check compares with the
organisers' simulator.

``replica_from_env`` builds the three inputs (spec, ROM, exogenous series) from a freshly loaded environment;
``replica_parts`` builds all but the ROM, so that the environment can be freed before the ROM build (cache.py).
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np

from hvac_controller.d1.core import core_from_env
from hvac_controller.d1.ops import Ops
from hvac_controller.d1.plant import Plant
from hvac_controller.d1.rom import Core, LinearRom, build_rom
from hvac_controller.d1.spec import BuildingSpec, Exogenous

EXO_STEP = ("t_amb", "t_amb_next", "comfort", "win_now", "win_next", "occ_step", "occ_rew", "price_e", "carbon_e",
            "price_g", "carbon_g", "max_cost", "max_carbon", "reward_dt")


def replica_parts(env, n_steps: int) -> tuple[BuildingSpec, Core, Exogenous, np.ndarray, dict]:
    """Spec, thermal core, exogenous series and reset observation of a freshly loaded environment, which this resets;
    and their build timings. The core holds copies only, so the environment can be dropped before ``build_rom``.

    The occupancy object is copied before ``env.reset()`` asks it anything, so the exogenous occupancy replays the
    environment's own draws.
    """
    t0 = time.time()
    spec = BuildingSpec.from_env(env)
    occupancy = copy.deepcopy(env.building._occupancy)
    obs0 = env.reset()[0]
    core = core_from_env(env)
    t1 = time.time()
    exo = Exogenous.from_env(env, spec, n_steps, occupancy)
    return spec, core, exo, obs0, {"spec_s": t1 - t0, "exo_s": time.time() - t1}


def replica_from_env(env, n_steps: int, log=None) -> tuple[BuildingSpec, LinearRom, Exogenous, np.ndarray, dict]:
    """replica_parts with the ROM built: spec, ROM, exogenous series, reset observation, build timings."""
    spec, core, exo, obs0, timings = replica_parts(env, n_steps)
    t0 = time.time()
    rom, info = build_rom(core, log=log)
    return spec, rom, exo, obs0, {**timings, "rom_s": time.time() - t0, "rom": info}


class ThermalRom:
    """LinearRom.step for S stacked ROMs of one building and a grouped batch: x' = a * x + B w, y' = C x', with the
    VAV air power solved implicitly. Orders are padded to the largest with modes a = 0, B = 0, C = 0, which stay
    zero and add nothing, so every scenario steps exactly as alone.

    The air power solves (I + G M) p = G r, G = diag(g). M (V x V) is nearly diagonal: within one step a VAV's air
    barely reaches the sensors of other zones. So it is solved by Jacobi sweeps p <- D^-1 G (r - M_off p),
    D = I + diag(g * m_ii), whose error contracts by ||D^-1 G M_off||_inf <= q = max_i g_i / (1 + g_i m_ii) *
    sum_j |m_ij|. That grows with every g_i, so q is taken at ``g_max``, the largest conductance any valid action
    gives, and over all S scenarios. The number of sweeps is fixed from q and the dtype, so the result is within 1e-8
    (float32) or 1e-15 (float64) of the exact solve for every valid action, with no device sync. If q >= 0.5 it falls
    back to the batched LU solve.
    """

    def __init__(self, roms: list[LinearRom], ops: Ops, g_max: np.ndarray):
        o = ops
        Z, V = roms[0].n_zones, roms[0].n_vavs
        S, r = len(roms), max(rom.order for rom in roms)
        self.o, self.S, self.V = o, S, V

        def stack(get, shape):
            out = np.zeros((S,) + shape)
            for s, rom in enumerate(roms):
                v = get(rom)
                out[(s,) + tuple(slice(0, n) for n in v.shape)] = v
            return out

        self.a = o.asarray(stack(lambda m: m.a, (r,))[:, None, :])  # (S, 1, r)
        self.b_q_t = o.asarray(stack(lambda m: m.b[:, :Z].T, (Z, r)))  # (S, Z, r)
        self.b_p_t = o.asarray(stack(lambda m: m.b[:, Z:Z + V].T, (V, r)))
        self.b_a = o.asarray(stack(lambda m: m.b[:, Z + V], (r,))[:, None, :])
        self.c_t = o.asarray(stack(lambda m: m.c.T, (r, V + Z)))  # (S, r, V + Z)
        self.n_p_t = o.asarray(np.stack([rom._np.T for rom in roms]))  # (S, V, V + Z): output response to air power
        m = np.stack([np.asarray(rom._m, dtype=np.float64) for rom in roms])  # (S, V, V): reduced Q^T A^-1 P
        d = np.diagonal(m, axis1=1, axis2=2).copy()
        if d.min() <= 0.0:
            raise ValueError("the air-side matrix has a non-positive diagonal")
        off = m - d[:, :, None] * np.eye(V)
        weight = g_max / (1.0 + g_max * d)
        self.q = float(np.max(weight * np.abs(off).sum(axis=2)))
        tol = 1e-8 if "float32" in str(o.dtype) else 1e-15
        if self.q >= 0.5:
            self.sweeps = None
            self.m, self.eye = o.asarray(m[:, None]), o.eye(V)
        else:  # the diagonal solve already leaves error <= q, each sweep multiplies it by q
            self.sweeps = max(0, math.ceil(math.log(tol) / math.log(self.q)) - 1) if self.q > 0.0 else 0
        self.m_diag, self.m_off_t = o.asarray(d[:, None, :]), o.asarray(np.transpose(off, (0, 2, 1)))
        self.t_ref = o.asarray(np.array([rom.t_ref for rom in roms])[:, None, None])  # (S, 1, 1)
        self.x0 = stack(lambda m: m.x0, (r,))  # (S, r), NumPy: rows take theirs in BuildingEnv
        self.y0 = np.stack([rom.y0 for rom in roms])  # (S, V + Z)

    def step(self, x, q_zone, t_amb, g, ts):
        """x (B, r), q_zone (B, Z), t_amb (B,), g and ts (B, V) -> x' (B, r) and y' (B, V + Z) in kelvin."""
        S = self.S

        def grouped(v):
            return v.reshape(S, v.shape[0] // S, *v.shape[1:])

        x_pre = (self.a * grouped(x) + grouped(q_zone) @ self.b_q_t
                 + (grouped(t_amb)[:, :, None] - self.t_ref) * self.b_a)
        y_pre = x_pre @ self.c_t
        p = self.air_power(grouped(g), (grouped(ts) - self.t_ref) - y_pre[:, :, :self.V])
        x_new, y_new = x_pre + p @ self.b_p_t, y_pre + p @ self.n_p_t + self.t_ref
        return x_new.reshape(x.shape), y_new.reshape(x.shape[0], -1)

    def air_power(self, g, r):
        """p with (I + G M) p = G r, r = supply minus the sensed temperature before the air acts; g, r (S, b, V)."""
        if self.sweeps is None:
            return self.o.solve(self.eye + g[..., None] * self.m, g * r)
        inv = g / (1.0 + g * self.m_diag)
        p = inv * r
        for _ in range(self.sweeps):
            p = inv * (r - p @ self.m_off_t)
        return p


class BuildingEnv:
    def __init__(self, spec: BuildingSpec, rom: LinearRom | list[LinearRom], exo: Exogenous | list[Exogenous],
                 num_envs: int = 1, ops: Ops | None = None, exact: bool = True, emulate_f32: bool = True):
        o = ops or Ops("numpy")
        roms, exos = (rom if isinstance(rom, list) else [rom]), (exo if isinstance(exo, list) else [exo])
        S = len(roms)
        if len(exos) != S or num_envs % S:
            raise ValueError(f"{S} ROMs, {len(exos)} exogenous series, {num_envs} rows: need one series per ROM and "
                             "the same number of rows for each")
        for m, e in zip(roms, exos):
            if np.max(np.abs(e.h - m.h)) > 0.0:
                raise ValueError("a ROM was built for another convection coefficient than its weather gives")
        self.n_steps = exos[0].n_steps
        if any(e.n_steps != self.n_steps for e in exos):
            raise ValueError("exogenous series of different lengths")
        self.o, self.spec, self.S, self.num_envs = o, spec, S, num_envs
        self.plant = Plant(spec, o, exact=exact, emulate_f32=emulate_f32)
        self.rom = ThermalRom(roms, o, self.plant.max_air_conductance())
        # Exogenous series as one (S * n, F) table: a step gathers one row per batch row.
        cols, self.exo_cols, start = [], {}, 0
        for name in EXO_STEP:
            v = np.stack([np.asarray(getattr(e, name), dtype=np.float64).reshape(self.n_steps, -1) for e in exos])
            width = v.shape[2]
            self.exo_cols[name] = start if getattr(exos[0], name).ndim == 1 else slice(start, start + width)
            cols.append(v)
            start += width
        self.exo = o.asarray(np.concatenate(cols, 2).reshape(S * self.n_steps, start))
        self.obs_exo = o.asarray(np.concatenate([e.obs_exo for e in exos]))  # (S * (n + 1), n_exo)
        self.row_s = o.index(np.arange(num_envs) // (num_envs // S))  # scenario of each row
        rows = np.arange(num_envs) // (num_envs // S)
        self.init = {"x": o.asarray(self.rom.x0[rows]), "y": o.asarray(self.rom.y0[rows]),
                     "tank": o.full((num_envs,), spec.tank0), "ret": o.full((num_envs,), spec.ret0)}
        self.num_obs = self.plant.n_obs
        self.num_actions = (len(spec.cont), len(spec.disc))
        self.obs0 = self.plant.reset_obs(num_envs, self.obs_exo[self.row_s * (self.n_steps + 1)])
        self.k = None
        self.state = None

    # ---- episode ----------------------------------------------------------------------------------------------
    def reset(self):
        """Every row at the start of its scenario's episode."""
        o = self.o
        self.state = {name: v + 0.0 for name, v in self.init.items()}
        self.k = o.index(np.zeros(self.num_envs, dtype=np.int64))
        self.obs = self.obs0 + 0.0
        return self.obs

    def reset_done(self, done):
        """Rows whose episode ended (done, bool (B,)) back to their initial state and observation; others untouched."""
        o = self.o
        self.state = {name: o.where(done[:, None] if v.ndim == 2 else done, self.init[name], v)
                      for name, v in self.state.items()}
        self.k = o.where(done, 0 * self.k, self.k)
        self.obs = o.where(done[:, None], self.obs0, self.obs)
        return self.obs

    def step(self, a_cont, a_disc):
        o, k = self.o, self.k
        e = self.exo[self.row_s * self.n_steps + k]
        obs_exo_next = self.obs_exo[self.row_s * (self.n_steps + 1) + k + 1]
        self.state, out = self._advance(self.state, o.asarray(a_cont), o.asarray(a_disc), e, obs_exo_next)
        self.k = k + 1
        self.obs = out["obs"]
        return out["obs"], out["reward"], self.k >= self.n_steps, {"post": out, "y": self.state["y"]}

    def _advance(self, s: dict, a_cont, a_disc, e, obs_exo_next) -> tuple[dict, dict]:
        """One step as a pure function of arrays: state, actions and each row's exogenous row -> state', outputs."""
        c = self.exo_cols
        dev = self.plant.decode(a_cont, a_disc)
        aux = self.plant.pre(s["y"], s["tank"], dev, e[:, c["t_amb"]], e[:, c["comfort"]], e[:, c["occ_step"]])
        x_new, y_new = self.rom.step(s["x"], aux["q_zone"], e[:, c["t_amb"]], aux["g"], aux["ts"])
        out = self.plant.post(
            y_new, s["ret"], dev, aux, e[:, c["t_amb_next"]], e[:, c["win_now"]], e[:, c["win_next"]],
            e[:, c["occ_rew"]], e[:, c["price_e"]], e[:, c["carbon_e"]], e[:, c["price_g"]], e[:, c["carbon_g"]],
            e[:, c["max_cost"]], e[:, c["max_carbon"]], e[:, c["reward_dt"]], obs_exo_next)
        return {"x": x_new, "y": y_new, "tank": out["tank"], "ret": out["ret"]}, out

