"""Step D: speed of the differentiable replica on a device, and what float32 costs in accuracy.

For one scenario, after building the replica once (torch_env.replica_from_env):

1. setup: seconds to load the environment and build the spec, the ROM and the exogenous series;
2. accuracy: structured random actions (trace.ExcitePolicy, one seed per batch row) for the whole episode on the
   NumPy float64 replica (validated against the original in step C) against torch float64 and float32 on the
   device; and batch rows against a single run with the same actions (batch independence);
3. forward: steps per second without gradients at each batch size, eager and, if asked, torch.compile'd;
4. forward + backward: gradient windows of h steps on the relaxed plant (exact=False), loss = summed reward,
   actions as leaf tensors; env-steps per second, peak device memory, finite gradients;
5. breakdown: device time of the ROM step (and of its V x V solve alone), plant.pre and plant.post at one batch size.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hvac_controller.d1.ops import Ops
from hvac_controller.d1.torch_env import BuildingEnv, replica_from_env
from hvac_controller.d1.trace import ExcitePolicy
from hvac_controller.sim import load_env

ACC_ROWS = 8  # action sequences compared with the NumPy reference
ACC_BATCH = 64  # batch the device runs them in (rows ACC_ROWS.. are other seeds)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def excite_actions(action_spec, n_steps: int, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """(n_steps, n_rows, n_cont) and (n_steps, n_rows, n_disc): one ExcitePolicy seed per row."""
    pols = [ExcitePolicy(action_spec, seed=j) for j in range(n_rows)]
    acts = [[p.get_action(None) for p in pols] for _ in range(n_steps)]
    a_c = np.array([[a["continuous_action"] for a in row] for row in acts], dtype=np.float64)
    a_d = np.array([[a["discrete_action"] for a in row] for row in acts], dtype=np.float64)
    return a_c, a_d


def _rollout(env: BuildingEnv, a_c: np.ndarray, a_d: np.ndarray, rows: int) -> dict[str, np.ndarray]:
    """Zone temperatures of the first `rows` batch rows and every row's reward, per step."""
    o, V = env.o, env.spec.n_vavs
    env.reset()
    zone, reward = [], []
    for k in range(len(a_c)):
        _, r, _, info = env.step(a_c[k], a_d[k])
        zone.append(o.to_numpy(info["y"][:rows, V:]))
        reward.append(o.to_numpy(r))
    return {"zone": np.asarray(zone), "reward": np.asarray(reward)}


def _compare(run: dict, ref: dict, rows: int) -> dict:
    dz = np.abs(run["zone"][:, :rows] - ref["zone"][:, :rows])
    s_run, s_ref = run["reward"][:, :rows].sum(0), ref["reward"][:, :rows].sum(0)
    return {"zone_max_k": float(dz.max()), "zone_max_k_by_day": [float(dz[i:i + 288].max()) for i in range(0, len(dz), 288)],
            "reward_step_max_abs": float(np.abs(run["reward"][:, :rows] - ref["reward"][:, :rows]).max()),
            "reward_sum_rel_max": float(np.max(np.abs(s_run - s_ref) / np.abs(s_ref)))}


def check_accuracy(spec, rom, exo, action_spec, device: str, n_steps: int, log=print) -> dict:
    a_c, a_d = excite_actions(action_spec, n_steps, ACC_BATCH)
    t0 = time.time()
    ref = _rollout(BuildingEnv(spec, rom, exo, ACC_ROWS, Ops("numpy")), a_c[:, :ACC_ROWS], a_d[:, :ACC_ROWS], ACC_ROWS)
    out = {"steps": n_steps, "rows": ACC_ROWS, "batch": ACC_BATCH, "numpy_s": time.time() - t0}
    runs = {}
    for name, dtype, f32 in (("float64", torch.float64, True), ("float32", torch.float32, False)):
        t0 = time.time()
        env = BuildingEnv(spec, rom, exo, ACC_BATCH, Ops("torch", dtype=dtype, device=device), emulate_f32=f32)
        with torch.no_grad():
            runs[name] = _rollout(env, a_c, a_d, ACC_ROWS)
        out[name] = {**_compare(runs[name], ref, ACC_ROWS), "seconds": time.time() - t0}
        log(f"  accuracy {name}: zone max {out[name]['zone_max_k']:.2e} K, reward sum {out[name]['reward_sum_rel_max']:.2e}")
    env = BuildingEnv(spec, rom, exo, 1, Ops("torch", dtype=torch.float32, device=device), emulate_f32=False)
    with torch.no_grad():
        single = _rollout(env, a_c[:, :1], a_d[:, :1], 1)
    out["batch_independence_float32"] = _compare(single, runs["float32"], 1)
    log(f"  batch independence: zone max {out['batch_independence_float32']['zone_max_k']:.2e} K")
    return out


def _random_actions(B: int, n_c: int, n_d: int, device: str, n: int = 16):
    """A short cycle of random actions on the device (the timing does not depend on their values)."""
    g = torch.Generator(device=device).manual_seed(0)
    a_c = torch.rand((n, B, n_c), generator=g, device=device) * 2.0 - 1.0
    a_d = (torch.rand((n, B, n_d), generator=g, device=device) > 0.2).float()
    return a_c, a_d


def forward_speed(spec, rom, exo, device: str, batches, n_steps: int = 288, compile_step: bool = False) -> dict:
    out = {}
    for B in batches:
        try:
            env = BuildingEnv(spec, rom, exo, B, Ops("torch", dtype=torch.float32, device=device), emulate_f32=False,
                              compile_step=compile_step)
            a_c, a_d = _random_actions(B, len(spec.cont), len(spec.disc), device)
            with torch.no_grad():
                env.reset()
                t0 = time.time()
                env.step(a_c[0], a_d[0])  # includes compilation when compiled
                _sync(device)
                first = time.time() - t0
                for k in range(1, 10):
                    env.step(a_c[k % 16], a_d[k % 16])
                _sync(device)
                t0 = time.time()
                for k in range(n_steps):
                    env.step(a_c[k % 16], a_d[k % 16])
                _sync(device)
                dt = time.time() - t0
            out[str(B)] = {"ms_per_step": dt / n_steps * 1e3, "env_steps_per_s": B * n_steps / dt, "first_step_s": first}
        except Exception as e:  # out of memory or a compile failure: report it and go on
            out[str(B)] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return out


def backward_speed(spec, rom, exo, device: str, batches, windows, n_windows: int = 3,
                   compile_step: bool = False) -> dict:
    out = {}
    cuda = device.startswith("cuda")
    for h in windows:
        for B in batches:
            key = f"h{h}_B{B}"
            env = None
            try:
                env = BuildingEnv(spec, rom, exo, B, Ops("torch", dtype=torch.float32, device=device), exact=False,
                                  emulate_f32=False, compile_step=compile_step)
                env.reset()
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                times, finite = [], True
                for w in range(n_windows + 1):  # the first window warms up (and compiles)
                    a_c, a_d = _random_actions(B, len(spec.cont), len(spec.disc), device, n=h)
                    a_c = (a_c * 0.9).requires_grad_(True)
                    a_d = a_d.clamp(0.05, 0.95).requires_grad_(True)  # relaxed run commands
                    _sync(device)
                    t0 = time.time()
                    total = 0.0
                    for k in range(h):
                        _, r, _, _ = env.step(a_c[k], a_d[k])
                        total = total + r.sum()
                    total.backward()
                    _sync(device)
                    times.append(time.time() - t0)
                    finite &= bool(torch.isfinite(a_c.grad).all()) and bool(torch.isfinite(a_d.grad).all())
                    env.clear_grad()
                dt = float(np.mean(times[1:]))
                out[key] = {"h": h, "B": B, "s_per_window": dt, "env_steps_per_s": B * h / dt, "grad_finite": finite,
                            "first_window_s": times[0],
                            "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None}
            except Exception as e:
                out[key] = {"h": h, "B": B, "error": f"{type(e).__name__}: {str(e)[:300]}"}
            del env
            if cuda:
                torch.cuda.empty_cache()
    return out


def _timeit(fn, device: str, n: int = 50) -> float:
    for _ in range(3):
        fn()
    _sync(device)
    t0 = time.time()
    for _ in range(n):
        fn()
    _sync(device)
    return (time.time() - t0) / n * 1e3


def breakdown(spec, rom, exo, device: str, B: int) -> dict:
    """Milliseconds per call of each part of one step, on the state after a few steps."""
    env = BuildingEnv(spec, rom, exo, B, Ops("torch", dtype=torch.float32, device=device), emulate_f32=False)
    a_c, a_d = _random_actions(B, len(spec.cont), len(spec.disc), device)
    pl, tr, o = env.plant, env.rom, env.o
    with torch.no_grad():
        env.reset()
        for k in range(5):
            env.step(a_c[k], a_d[k])
        s, e = env.state, {name: v[env.k] for name, v in env.exo.items()}
        ones = env.ones
        t_amb = ones * e["t_amb"]
        dev = pl.decode(a_c[0], a_d[0])
        aux = pl.pre(s["y"], s["tank"], dev, t_amb, ones * e["comfort"], ones[:, None] * e["occ_step"])
        _, y_new = tr.step(s["x"], aux["q_zone"], t_amb, aux["g"], aux["ts"])
        lhs = tr.eye + aux["g"][:, :, None] * tr.m
        rhs = aux["g"] * (aux["ts"] - tr.t_ref)

        def post():
            return pl.post(y_new, s["ret"], dev, aux, ones * e["t_amb_next"], ones[:, None] * e["win_now"],
                           ones[:, None] * e["win_next"], ones[:, None] * e["occ_rew"], ones * e["price_e"],
                           ones * e["carbon_e"], ones * e["price_g"], ones * e["carbon_g"], ones * e["max_cost"],
                           ones * e["max_carbon"], ones * e["reward_dt"], ones[:, None] * env.obs_exo[env.k + 1])

        return {"B": B,
                "step_ms": _timeit(lambda: env._advance(s, a_c[0], a_d[0], e, env.obs_exo[env.k + 1]), device),
                "decode_ms": _timeit(lambda: pl.decode(a_c[0], a_d[0]), device),
                "pre_ms": _timeit(lambda: pl.pre(s["y"], s["tank"], dev, t_amb, ones * e["comfort"],
                                                 ones[:, None] * e["occ_step"]), device),
                "rom_ms": _timeit(lambda: tr.step(s["x"], aux["q_zone"], t_amb, aux["g"], aux["ts"]), device),
                "solve_ms": _timeit(lambda: o.solve(lhs, rhs), device),
                "post_ms": _timeit(post, device)}


def run_bench(config_gin: Path, out_dir: Path, device: str, batches, windows, back_batches, n_steps: int,
              compile_step: bool, log=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False  # keep float32 matmuls float32 on GPUs that have TF32
    report = {"config": str(config_gin), "torch": torch.__version__, "device": device,
              "device_name": torch.cuda.get_device_name() if device.startswith("cuda") else "cpu"}

    def save():
        (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))

    t0 = time.time()
    env, _ = load_env(config_gin)
    load_s = time.time() - t0
    spec, rom, exo, _, build = replica_from_env(env, n_steps)
    report.update(label=spec.label, n_zones=spec.n_zones, n_vavs=spec.n_vavs, n_ahus=spec.n_ahus, n_obs=len(spec.obs),
                  n_cont=len(spec.cont), n_disc=len(spec.disc), rom_order=rom.order,
                  setup={"load_env_s": load_s, **{k: v for k, v in build.items() if k != "rom"}})
    log(f"[bench] {spec.label}: Z={spec.n_zones} V={spec.n_vavs} r={rom.order} on {report['device_name']} | setup "
        f"load {load_s:.0f}s spec {build['spec_s']:.0f}s ROM {build['rom_s']:.0f}s exo {build['exo_s']:.0f}s")
    save()

    report["accuracy"] = check_accuracy(spec, rom, exo, env.action_spec(), device, n_steps, log=log)
    save()
    report["forward"] = forward_speed(spec, rom, exo, device, batches)
    log("  forward " + " ".join(f"B{b}:{v.get('env_steps_per_s', 0):.0f}/s" for b, v in report["forward"].items()))
    save()
    report["backward"] = backward_speed(spec, rom, exo, device, back_batches, windows)
    log("  backward " + " ".join(f"{k}:{v.get('env_steps_per_s', 0):.0f}/s" for k, v in report["backward"].items()))
    save()
    report["breakdown"] = breakdown(spec, rom, exo, device, 1024)
    log(f"  breakdown B1024: {report['breakdown']}")
    save()
    if compile_step:
        report["forward_compiled"] = forward_speed(spec, rom, exo, device, batches, compile_step=True)
        log("  forward compiled " + " ".join(f"B{b}:{v.get('env_steps_per_s', 0):.0f}/s"
                                            for b, v in report["forward_compiled"].items()))
        save()
        report["backward_compiled"] = backward_speed(spec, rom, exo, device, back_batches, windows[:1],
                                                     compile_step=True)
        log("  backward compiled " + " ".join(f"{k}:{v.get('env_steps_per_s', 0):.0f}/s"
                                             for k, v in report["backward_compiled"].items()))
        save()
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batches", default="1,64,256,1024,4096", help="forward batch sizes")
    ap.add_argument("--back-batches", default="64,256,1024", help="forward + backward batch sizes")
    ap.add_argument("--windows", default="32,128", help="gradient window lengths h")
    ap.add_argument("--steps", type=int, default=8640, help="episode length for the accuracy check")
    ap.add_argument("--compile", action="store_true", help="also time the torch.compile'd step")
    a = ap.parse_args()
    ints = (lambda s: tuple(int(v) for v in s.split(",") if v))
    run_bench(a.config, a.out_dir, a.device, ints(a.batches), ints(a.windows), ints(a.back_batches), a.steps,
              a.compile, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
