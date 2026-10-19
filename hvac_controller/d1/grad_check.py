"""Step 5 of direction 1: are the replica's gradients correct, and usable over long windows?

Open loop (no policy yet), torch, no protobuf float32 emulation (the training configuration). The objective of a
window of h steps is J = the summed reward of each batch row. Per scenario the replica is first run for a day of
structured random actions (trace.ExcitePolicy, no gradients); windows then start from that state at 00:00 and from
the state 6 h later (06:00, when the schedule switches to comfort). Window actions are another ExcitePolicy sequence,
continuous ones mapped to 0.7 a + U(-0.05, 0.05): inside [-0.75, 0.75] and off every device threshold (see _base).
Two modes:
"relaxed" (exact=False, run commands 0.1 / 0.9 and differentiable) and "exact" (run commands fixed at 0 / 1).

E1 correct (float64): autograd directional derivatives dJ/da . v against central differences (eps 1e-4 and 1e-6,
   the better one kept), for 8 random unit directions over every action of the window and one per action group;
   error |FD - AD| / ||grad J|| (|v| = 1), so a direction that happens to be nearly orthogonal to the gradient does
   not inflate it. The one-sided differences flag a direction whose perturbation crosses a kink (min / max / clamp):
   there the difference quotient is not a derivative, so the error is also reported without those directions. And
   the same window's autograd gradient in float32 (the training dtype) against float64.
E2 through time (float64, h = 288): ||dJ/da_k|| by lag from the window's end, in total and per action group.
E3 usable (float32, B rows): each row has its own constant action offset theta_i (zero) and exploration noise
   sigma xi_i. First order (FO): the per-row gradients dJ_i/dtheta_i, one backward pass. Zeroth order (ZO): the score
   function (J_i - mean J) sum_t xi_i,t / sigma. For each: signal-to-noise ratio ||mean|| / sqrt(tr Cov / B), and
   the cosine of its two half-batch means; and the cosine of the FO mean with the ZO mean.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hvac_controller.d1.bench import _new_env, _sync, excite_actions
from hvac_controller.d1.torch_env import replica_from_env
from hvac_controller.sim import load_env

STARTS = {"00:00": 288, "06:00": 360}  # warm-up steps before the window
H_MAX = 288
LAGS = (0, 1, 3, 12, 36, 72, 144, 287)
EPS = (1e-4, 1e-6)
KINK = 1e-3  # |forward - backward difference| / ||grad J|| above this: the perturbation crossed a kink
N_RANDOM = 8
GROUPS = ("ahu_sp", "ahu_p", "ahu_r", "w_sp", "w_dp", "damper", "valve")


def _env_at(spec, rom, exo, start: dict, B: int, device: str, dtype, exact: bool):
    """A fresh environment of B rows, all in the warmed state."""
    env = _new_env(spec, rom, exo, B, device, dtype=dtype, exact=exact)
    env.reset()
    env.state = {k: v.to(dtype).expand(B, *v.shape[1:]).clone() for k, v in start["state"].items()}
    env.k = start["k"]
    return env


def _window(env, a_c, a_d):
    """Summed reward per row over the window; a_c (h, B, n_cont), a_d (h, B, n_disc)."""
    total = 0.0
    for k in range(a_c.shape[0]):
        _, r, _, _ = env.step(a_c[k], a_d[k])
        total = total + r
    return total


def _warm(spec, rom, exo, device: str, a_c, a_d) -> dict:
    """Run the given warm-up actions (one row, float64, no gradients); returns the state reached and its step."""
    env = _new_env(spec, rom, exo, 1, device, dtype=torch.float64)
    env.reset()
    with torch.no_grad():
        _window(env, torch.as_tensor(a_c, device=device), torch.as_tensor(a_d, device=device))
    return {"state": {k: v.detach().clone() for k, v in env.state.items()}, "k": env.k}


def _base(a_c: np.ndarray, a_d: np.ndarray, relaxed: bool, rng):
    """Window actions. Continuous: 0.7 a + U(-0.05, 0.05), so no value sits on a device threshold. Normalised -0.8 is
    exactly the damper floor (0.1) and the static-pressure floor (200 Pa of 0-2000): the first run of notebook 14
    (2026-10-19) clipped ExcitePolicy's bound actions to -0.8, and its finite differences straddled those kinks.
    Run commands: 0.1 / 0.9 when relaxed, else 0 / 1."""
    return 0.7 * a_c + rng.uniform(-0.05, 0.05, a_c.shape), (0.1 + 0.8 * a_d if relaxed else a_d)


def _directions(spec, h: int, relaxed: bool, rng) -> list[tuple[str, np.ndarray, np.ndarray]]:
    n_c, n_d = len(spec.cont), len(spec.disc)
    out = []
    for j in range(N_RANDOM):
        vc, vd = rng.normal(size=(h, 1, n_c)), rng.normal(size=(h, 1, n_d)) * relaxed
        out.append((f"random{j}", vc, vd))
    for g in GROUPS:
        mask = np.array([row[0] == g for row in spec.cont], dtype=float)
        if mask.any():
            out.append((g, rng.normal(size=(h, 1, n_c)) * mask, np.zeros((h, 1, n_d))))
    if relaxed:
        out.append(("run", np.zeros((h, 1, n_c)), rng.normal(size=(h, 1, n_d))))
    norm = [np.sqrt((vc ** 2).sum() + (vd ** 2).sum()) for _, vc, vd in out]
    return [(name, vc / s, vd / s) for (name, vc, vd), s in zip(out, norm)]


def fd_report(dirs, j0: float, j: np.ndarray, g_c: np.ndarray, g_d: np.ndarray) -> dict:
    """E1 from the autograd gradient (g_c, g_d), the unperturbed objective j0 and the perturbed ones j
    (n_dirs * len(EPS) * 2, in the order direction, eps, +/-)."""
    g_norm = float(np.sqrt((g_c ** 2).sum() + (g_d ** 2).sum()))
    j = np.asarray(j, dtype=np.float64).reshape(len(dirs), len(EPS), 2)
    eps = np.asarray(EPS)
    fwd, bwd = (j[..., 0] - j0) / eps, (j0 - j[..., 1]) / eps
    fd = 0.5 * (fwd + bwd)
    errors = {}
    for i, (name, vc, vd) in enumerate(dirs):
        ad = float((g_c * vc).sum() + (g_d * vd).sum())
        err = np.abs(fd[i] - ad) / g_norm
        best = int(np.argmin(err))
        errors[name] = {"ad": ad, "fd": float(fd[i, best]), "err": float(err[best]), "eps": EPS[best],
                        "err_by_eps": err.tolist(), "kink": bool(abs(fwd[i, best] - bwd[i, best]) / g_norm > KINK)}
    all_err = np.array([e["err"] for e in errors.values()])
    smooth = np.array([e["err"] for e in errors.values() if not e["kink"]])
    return {"grad_norm": g_norm, "grad_finite": bool(np.isfinite(g_c).all() and np.isfinite(g_d).all()),
            "err_median": float(np.median(all_err)), "err_max": float(all_err.max()),
            "share_above_1e-3": float(np.mean(all_err > 1e-3)), "n_kink": int(len(all_err) - len(smooth)),
            "err_median_smooth": float(np.median(smooth)) if smooth.size else None,
            "err_max_smooth": float(smooth.max()) if smooth.size else None, "directions": errors}


def lag_report(spec, g_c: np.ndarray, g_d: np.ndarray, relaxed: bool) -> dict:
    """E2: gradient norm per action step, by lag from the window's end; g_c (h, 1, n_cont), g_d (h, 1, n_disc)."""
    h = len(g_c)
    by_lag = np.sqrt((g_c ** 2).sum(axis=(1, 2)) + (g_d ** 2).sum(axis=(1, 2)))[::-1]
    peak = float(by_lag.max()) or 1.0
    groups = {g: np.array([row[0] == g for row in spec.cont]) for g in GROUPS}
    out = {"h": h, "lag_rel": {str(lag): float(by_lag[lag] / peak) for lag in LAGS if lag < h}, "peak": peak,
           "peak_lag": int(np.argmax(by_lag)),
           "groups": {g: {str(lag): float(np.sqrt((g_c[h - 1 - lag][:, m] ** 2).sum())) for lag in (0, 144) if lag < h}
                      for g, m in groups.items() if m.any()},
           "profile": by_lag.tolist()}
    if relaxed:
        out["groups"]["run"] = {str(lag): float(np.sqrt((g_d[h - 1 - lag] ** 2).sum())) for lag in (0, 144) if lag < h}
    return out


def _autograd(spec, rom, exo, start, base_c, base_d, relaxed: bool, device: str, dtype) -> tuple[np.ndarray, np.ndarray]:
    """dJ/da of one row over the window, as float64 arrays (h, 1, n_cont) and (h, 1, n_disc)."""
    t = (lambda x: torch.as_tensor(x, dtype=dtype, device=device))
    env = _env_at(spec, rom, exo, start, 1, device, dtype, exact=not relaxed)
    a_c = t(base_c[:, None]).requires_grad_(True)
    a_d = t(base_d[:, None]).requires_grad_(relaxed)
    _window(env, a_c, a_d).sum().backward()
    g_d = a_d.grad if relaxed else torch.zeros_like(a_d)
    return a_c.grad.double().cpu().numpy(), g_d.double().cpu().numpy()


def check_fd(spec, rom, exo, start, base_c, base_d, relaxed: bool, device: str, seed: int) -> tuple[dict, dict]:
    """E1 on one window, and E2 from the same (float64) autograd pass."""
    h = len(base_c)
    g_c, g_d = _autograd(spec, rom, exo, start, base_c, base_d, relaxed, device, torch.float64)
    f_c, f_d = _autograd(spec, rom, exo, start, base_c, base_d, relaxed, device, torch.float32)
    g64, g32 = np.concatenate([g_c.ravel(), g_d.ravel()]), np.concatenate([f_c.ravel(), f_d.ravel()])
    dirs = _directions(spec, h, relaxed, np.random.default_rng(seed))
    rows_c = [base_c[:, None]] + [base_c[:, None] + sign * eps * vc for _, vc, _ in dirs for eps in EPS
                                  for sign in (1.0, -1.0)]
    rows_d = [base_d[:, None]] + [base_d[:, None] + sign * eps * vd for _, _, vd in dirs for eps in EPS
                                  for sign in (1.0, -1.0)]
    t = (lambda x: torch.as_tensor(x, dtype=torch.float64, device=device))
    env = _env_at(spec, rom, exo, start, len(rows_c), device, torch.float64, exact=not relaxed)
    with torch.no_grad():
        j = _window(env, t(np.concatenate(rows_c, 1)), t(np.concatenate(rows_d, 1))).cpu().numpy()
    e1 = {"h": h, **fd_report(dirs, float(j[0]), j[1:], g_c, g_d),
          "float32": {"cos": _cos(g32, g64), "rel": float(np.linalg.norm(g32 - g64) / np.linalg.norm(g64)),
                      "finite": bool(np.isfinite(g32).all())}}
    return e1, lag_report(spec, g_c, g_d, relaxed)


def _stats(per_row: np.ndarray) -> tuple[dict, np.ndarray]:
    """SNR of the batch mean and the cosine of the two half-batch means; per_row (B, n)."""
    B = per_row.shape[0]
    mean = per_row.mean(0)
    halves = per_row[: B // 2].mean(0), per_row[B // 2:].mean(0)
    return {"snr": float(np.linalg.norm(mean) / np.sqrt(per_row.var(0, ddof=1).sum() / B)),
            "mean_norm": float(np.linalg.norm(mean)), "split_cos": _cos(*halves)}, mean


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def variance_report(fo: np.ndarray, score: np.ndarray, j: np.ndarray) -> dict:
    """E3 from per-row first-order gradients fo (B, n), score-function weights score (B, n) and objectives j (B,)."""
    fo, score, j = (np.asarray(v, dtype=np.float64) for v in (fo, score, j))
    zo = (j - j.mean())[:, None] * score
    s_fo, m_fo = _stats(fo)
    s_zo, m_zo = _stats(zo)
    return {"J_mean": float(j.mean()), "J_std": float(j.std()), "fo_finite": bool(np.isfinite(fo).all()), "fo": s_fo,
            "zo": s_zo, "cos_fo_zo": _cos(m_fo, m_zo), "snr_ratio": s_fo["snr"] / max(s_zo["snr"], 1e-30)}


def check_variance(spec, rom, exo, start, base_c, base_d, relaxed: bool, device: str, B: int, sigma: float,
                   seed: int) -> dict:
    """E3 on one window (float32)."""
    h = len(base_c)
    n_c, n_d = len(spec.cont), len(spec.disc)
    env = _env_at(spec, rom, exo, start, B, device, torch.float32, exact=not relaxed)
    g = torch.Generator(device=device).manual_seed(seed)
    xi_c = torch.randn((h, B, n_c), generator=g, device=device)
    xi_d = torch.randn((h, B, n_d), generator=g, device=device)
    th_c = torch.zeros((B, n_c), device=device, requires_grad=True)
    th_d = torch.zeros((B, n_d), device=device, requires_grad=True)
    bc = torch.as_tensor(base_c[:, None], dtype=torch.float32, device=device)
    bd = torch.as_tensor(base_d[:, None], dtype=torch.float32, device=device)
    a_c = (bc + th_c + sigma * xi_c).clamp(-1.0, 1.0)
    a_d = (bd + th_d + sigma * xi_d).clamp(0.0, 1.0) if relaxed else bd.expand(h, B, n_d)
    j = _window(env, a_c, a_d)
    j.sum().backward()
    if relaxed:
        fo = torch.cat([th_c.grad, th_d.grad], 1)
        score = torch.cat([xi_c.sum(0), xi_d.sum(0)], 1) / sigma
    else:
        fo, score = th_c.grad, xi_c.sum(0) / sigma
    return {"h": h, "B": B, "sigma": sigma,
            **variance_report(fo.cpu().numpy(), score.cpu().numpy(), j.detach().cpu().numpy())}


def run_grad_check(config_gin: Path, out_dir: Path, device: str, fd_windows, var_windows, B: int, sigma: float,
                   log=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    t_start = time.time()
    report = {"config": str(config_gin), "torch": torch.__version__, "device": device,
              "device_name": torch.cuda.get_device_name() if device.startswith("cuda") else "cpu",
              "B": B, "sigma": sigma, "starts": {}}

    def save():
        (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))

    env, _ = load_env(config_gin)
    n_warm_max = max(STARTS.values())
    spec, rom, exo, _, build = replica_from_env(env, n_warm_max + H_MAX)
    report.update(label=spec.label, n_zones=spec.n_zones, n_vavs=spec.n_vavs, n_cont=len(spec.cont),
                  n_disc=len(spec.disc), rom_order=rom.order, setup={k: v for k, v in build.items() if k != "rom"})
    log(f"[grad] {spec.label}: Z={spec.n_zones} V={spec.n_vavs} actions {len(spec.cont)}+{len(spec.disc)} "
        f"on {report['device_name']}")
    a_c, a_d = excite_actions(env.action_spec(), n_warm_max + H_MAX, 2)  # row 0 warms up, row 1 drives the windows
    for s_i, (label, n_warm) in enumerate(STARTS.items()):
        t0 = time.time()
        start = _warm(spec, rom, exo, device, a_c[:n_warm, :1], a_d[:n_warm, :1])
        res = report["starts"][label] = {"warm_steps": n_warm, "E1": {}, "E2": {}, "E3": {}}
        for relaxed in (True, False):
            mode = "relaxed" if relaxed else "exact"
            for h in fd_windows:
                bc, bd = _base(a_c[n_warm:n_warm + h, 1], a_d[n_warm:n_warm + h, 1], relaxed,
                               np.random.default_rng(10 * s_i + h))
                e1, e2 = check_fd(spec, rom, exo, start, bc, bd, relaxed, device, seed=100 * s_i + h)
                res["E1"][f"{mode}_h{h}"] = e1
                if h == max(fd_windows):
                    res["E2"][mode] = e2
                log(f"  {label} {mode} h={h}: E1 err median {e1['err_median']:.1e} max {e1['err_max']:.1e} "
                    f"(|grad| {e1['grad_norm']:.2e}, finite {e1['grad_finite']})")
            for h in var_windows:
                bc, bd = _base(a_c[n_warm:n_warm + h, 1], a_d[n_warm:n_warm + h, 1], relaxed,
                               np.random.default_rng(10 * s_i + h))
                e3 = check_variance(spec, rom, exo, start, bc, bd, relaxed, device, B, sigma, seed=1000 * s_i + h)
                res["E3"][f"{mode}_h{h}"] = e3
                log(f"  {label} {mode} h={h}: E3 FO snr {e3['fo']['snr']:.1f} ZO snr {e3['zo']['snr']:.2f} "
                    f"cos {e3['cos_fo_zo']:.2f}")
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            save()
        _sync(device)
        res["seconds"] = time.time() - t0
    report["seconds"] = time.time() - t_start
    save()
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fd-windows", default="36,288", help="E1 window lengths (the longest also gives E2)")
    ap.add_argument("--var-windows", default="12,36,72,144,288", help="E3 window lengths")
    ap.add_argument("--batch", type=int, default=1024, help="E3 rows")
    ap.add_argument("--sigma", type=float, default=0.05, help="E3 exploration noise (normalised action units)")
    a = ap.parse_args()
    ints = (lambda s: tuple(int(v) for v in s.split(",") if v))
    fd, var = ints(a.fd_windows), ints(a.var_windows)
    if max(fd + var) > H_MAX:
        ap.error(f"windows longer than {H_MAX} steps")
    run_grad_check(a.config, a.out_dir, a.device, fd, var, a.batch, a.sigma, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
