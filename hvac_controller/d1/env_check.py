"""Step C check: the plant + ROM replica (torch_env.BuildingEnv) against the organisers' environment, tier by tier.

Per scenario: build the ROM and the exogenous series once, then for each policy run the real environment with a
Tracer and compare:

- T0 exogenous: weather, occupancy (step and reward intervals), the observation's time features -> exact;
- T1 teacher-forced: plant.pre/post fed the traced state before and after each step and the traced device settings
  -> every intermediate (q, g, Ts, flows, hot water, tank, billed energy, reward terms, observation) to rounding;
  plus decode(action) against the settings the devices hold;
- T3 open loop: the replica alone, fed only the traced actions -> zone temperatures, rewards, energy;
- T4 closed loop: the same policy acting on the replica's own observations -> scenario score (the gate);
- T5 torch: T3 again in torch float64 / float32, gradcheck on a relaxed window, a long backward pass, step timing.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from hvac_controller.d1.core import core_from_env
from hvac_controller.d1.ops import Ops
from hvac_controller.d1.plant import Plant
from hvac_controller.d1.rom import build_rom
from hvac_controller.d1.spec import BuildingSpec, Exogenous
from hvac_controller.d1.torch_env import BuildingEnv
from hvac_controller.d1.trace import Tracer, make_policy
from hvac_controller.sim import load_env
from smart_buildings import evaluate


def _f32(x):
    return np.asarray(x, dtype=np.float32).astype(np.float64)


def _err(pred, ref) -> dict:
    pred, ref = np.asarray(pred, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    e = np.abs(pred - ref)
    scale = float(np.max(np.abs(ref))) if ref.size else 0.0
    return {"max_abs": float(e.max()) if e.size else 0.0, "rel": float(e.max() / scale) if scale > 0 else 0.0}


def check_exogenous(spec: BuildingSpec, exo: Exogenous, tr: dict) -> dict:
    n = len(tr["reward"])
    plain = [(j, i) for j, (kind, _, i, _, _) in enumerate(spec.obs) if kind == "exo"]
    js, cols = [j for j, _ in plain], [i for _, i in plain]
    return {
        "t_amb": _err(exo.t_amb[:n], tr["core_t_amb"]),
        "h": _err(exo.h[:n], tr["core_h"]),
        "occ_step": _err(exo.occ_step[:n], tr["occ_step"]),
        "occ_rew": _err(exo.occ_rew[:n], tr["occ_rew"]),
        "win_next_f32": _err(_f32(exo.win_next[:n]), np.stack([tr["rw_heat_sp"][:, 0], tr["rw_cool_sp"][:, 0]], 1)),
        "obs_time_features": _err(_f32(exo.obs_exo[1:n + 1][:, cols]), tr["obs"][:, js]),
        "obs0_time_features": _err(_f32(exo.obs_exo[0][cols]), tr["obs0"][js]),
    }


def _traced_settings(plant: Plant, tr: dict) -> dict:
    dev = plant.settings(ahu_on=tr["nat_ahu_on"], w_on=tr["nat_w_on"], ahu_sp=tr["nat_ahu_sp"], ahu_p=tr["nat_ahu_p"],
                         ahu_r=tr["nat_ahu_r"], w_sp=tr["nat_w_sp"], w_dp=tr["nat_w_head"] * plant.head_per_bar,
                         damper=tr["nat_damper_cmd"], valve=tr["nat_valve_cmd"])
    dev["w_head"] = tr["nat_w_head"]
    return dev


def check_teacher_forced(spec: BuildingSpec, exo: Exogenous, tr: dict) -> dict:
    """Plant equations on the traced state, all steps as one batch (NumPy float64, exact, float32 emulation)."""
    plant = Plant(spec, Ops("numpy"), exact=True, emulate_f32=True)
    n = len(tr["reward"])
    share = spec.vav_zone_share
    dev = _traced_settings(plant, tr)
    dec = plant.decode(tr["a_cont"], tr["a_disc"])
    decode = {k: _err(dec[k], dev[k]) for k in ("ahu_on", "ahu_sp", "ahu_p_cmd", "ahu_r", "w_on", "w_sp", "w_head",
                                                "damper", "valve")}
    y = np.concatenate([tr["sens_pre"], tr["zone_pre"]], 1)
    pre = plant.pre(y, tr["tank_pre"], dev, exo.t_amb[:n], exo.comfort[:n], exo.occ_step[:n])
    sens_post = tr["zone_post"] @ share
    y_new = np.concatenate([sens_post, tr["zone_post"]], 1)
    post = plant.post(y_new, tr["ret_pre"], dev, pre, exo.t_amb_next[:n], exo.win_now[:n], exo.win_next[:n],
                      exo.occ_rew[:n], exo.price_e[:n], exo.carbon_e[:n], exo.price_g[:n], exo.carbon_g[:n],
                      exo.max_cost[:n], exo.max_carbon[:n], exo.reward_dt[:n], exo.obs_exo[1:n + 1])
    obs0 = plant.reset_obs(1, exo.obs_exo[0][None])[0]
    return {
        "decode": decode,
        "damper": _err(dev["damper"], tr["damper"]), "valve": _err(dev["valve"], tr["valve"]),
        "sens_pre_vs_zone_pre": _err(tr["zone_pre"] @ share, tr["sens_pre"]),
        "q_zone": _err(pre["q_zone"], tr["core_q_zone"]), "g": _err(pre["g"], tr["core_g"]),
        "ts": _err(pre["ts"], tr["core_ts"]), "flow": _err(pre["flow"], tr["flow"]),
        "ratio": _err(pre["ratio"][:, spec.vav_ahu], tr["ratio"]), "ahu_raw": _err(pre["ahu_raw"], tr["ahu_raw"]),
        "ahu_flow": _err(pre["ahu_flow"], tr["ahu_flow"]), "p_eff": _err(pre["p_eff"], tr["ahu_p_eff"]),
        "ff_sum": _err(pre["ff_sum"], tr["ff_sum"]), "q_water": _err(pre["q_water"], tr["q_water"]),
        "ret": _err(post["ret"], tr["ret"]), "tank": _err(post["tank"], tr["tank"]),
        "d_tank": _err(post["d_tank"], tr["d_tank"]),
        "blower": _err(_f32(post["blower"]), tr["rw_blower"]), "cool": _err(_f32(post["cool"]), tr["rw_cool"]),
        "gas": _err(_f32(post["gas"]), tr["rw_gas"]), "pump": _err(_f32(post["pump"]), tr["rw_pump"]),
        "zone_temp_f32": _err(_f32(tr["zone_post"]), tr["rw_zone_temp"]),
        "regret": _err(_f32(post["regret"]), tr["rr_normalized_productivity_regret"]),
        "norm_cost": _err(_f32(post["norm_cost"]), tr["rr_normalized_energy_cost"]),
        "norm_carbon": _err(_f32(post["norm_carbon"]), tr["rr_normalized_carbon_emission"]),
        "reward": _err(post["reward"], tr["reward"]),
        "obs": _err(post["obs"], tr["obs"]), "obs0": _err(obs0, tr["obs0"]),
        "obs_worst_field": spec.field_names[int(np.argmax(np.abs(post["obs"] - tr["obs"]).max(axis=0)))],
    }


def _rollout(env: BuildingEnv, actions=None, policy=None, action_spec=None, n_steps: int = 0) -> dict:
    """Open loop (actions given) or closed loop (policy acting on the replica's observations), batch 1."""
    o = env.o
    obs = env.reset()
    out = {"obs0": o.to_numpy(obs)[0], "reward": [], "zone": [], "elec": [], "gas": [], "regret": [], "obs": [],
           "tank": [], "a_cont": [], "a_disc": []}
    for k in range(n_steps):
        if actions is not None:
            a_c, a_d = actions[0][k], actions[1][k]
        else:
            act = evaluate.validate_action(
                evaluate.get_policy_action(policy, o.to_numpy(obs)[0].astype(np.float32)), action_spec)
            a_c, a_d = act["continuous_action"], act["discrete_action"].astype(np.float64)
        obs, rew, done, info = env.step(np.asarray(a_c, dtype=np.float64)[None], np.asarray(a_d, dtype=np.float64)[None])
        post = info["post"]
        out["reward"].append(float(o.to_numpy(rew)[0]))
        out["zone"].append(o.to_numpy(info["y"])[0, env.spec.n_vavs:])
        out["elec"].append(float(o.to_numpy(post["elec"])[0]))
        out["gas"].append(float(o.to_numpy(post["gas"])[0]))
        out["regret"].append(float(o.to_numpy(post["regret"])[0]))
        out["tank"].append(float(o.to_numpy(post["tank"])[0]))
        out["obs"].append(o.to_numpy(obs)[0])
        out["a_cont"].append(np.asarray(a_c, dtype=np.float64))
        out["a_disc"].append(np.asarray(a_d, dtype=np.float64))
        if done:
            break
    return {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in out.items()}


def _sum_rel(pred, ref) -> float:
    s = float(np.sum(ref))
    return float(abs(np.sum(pred) - s) / abs(s)) if s != 0 else float(abs(np.sum(pred)))


def check_open_loop(spec, rom, exo, tr) -> tuple[dict, dict]:
    n = len(tr["reward"])
    env = BuildingEnv(spec, rom, exo, 1, Ops("numpy"))
    r = _rollout(env, actions=(tr["a_cont"], tr["a_disc"]), n_steps=n)
    elec_true = tr["rw_blower"] + np.abs(tr["rw_cool"]) + tr["rw_pump"]
    zone_err = np.abs(r["zone"] - tr["zone_post"])
    return {
        "zone_max_k": float(zone_err.max()), "zone_mae_k": float(zone_err.mean()),
        "zone_max_k_by_day": [float(zone_err[i:i + 288].max()) for i in range(0, n, 288)],
        "reward_sum_rel": _sum_rel(r["reward"], tr["reward"]), "reward_step_max_abs": _err(r["reward"], tr["reward"])["max_abs"],
        "elec_sum_rel": _sum_rel(r["elec"], elec_true), "gas_sum_rel": _sum_rel(r["gas"], tr["rw_gas"]),
        "regret_sum_rel": _sum_rel(r["regret"], tr["rr_normalized_productivity_regret"]),
        "tank": _err(r["tank"], tr["tank"]), "obs": _err(r["obs"], tr["obs"]), "obs0": _err(r["obs0"], tr["obs0"]),
        "score_true": float(abs(np.sum(tr["reward"])) / n), "score_replica": float(abs(np.sum(r["reward"])) / n),
    }, r


def check_closed_loop(spec, rom, exo, tr, policy_name, env_real, meta, seed) -> dict:
    n = len(tr["reward"])
    env = BuildingEnv(spec, rom, exo, 1, Ops("numpy"))
    policy = make_policy(policy_name, env_real, meta, seed)
    r = _rollout(env, policy=policy, action_spec=env_real.action_spec(), n_steps=n)
    true, rep = float(abs(np.sum(tr["reward"])) / n), float(abs(np.sum(r["reward"])) / len(r["reward"]))
    return {"score_true": true, "score_replica": rep, "score_rel": abs(rep - true) / true if true else rep,
            "steps": int(len(r["reward"])),
            "zone_max_k_vs_true": float(np.abs(r["zone"] - tr["zone_post"][:len(r["zone"])]).max())}


def check_torch(spec, rom, exo, tr, ref: dict, n_steps: int) -> dict:
    try:
        import torch
    except ImportError:
        return {"skipped": "torch not installed"}
    out = {"torch": torch.__version__}
    n = min(n_steps, len(tr["reward"]))
    acts = (tr["a_cont"][:n], tr["a_disc"][:n])
    for name, dtype, f32 in (("float64", torch.float64, True), ("float32", torch.float32, False)):
        env = BuildingEnv(spec, rom, exo, 1, Ops("torch", dtype=dtype), emulate_f32=f32)
        with torch.no_grad():
            r = _rollout(env, actions=acts, n_steps=n)
        out[name] = {"reward_vs_numpy": _err(r["reward"], ref["reward"][:n]),
                     "zone_vs_numpy_k": float(np.abs(r["zone"] - ref["zone"][:n]).max()),
                     "reward_sum_rel_vs_true": _sum_rel(r["reward"], tr["reward"][:n])}

    # Gradients through a relaxed window (no float32 rounding, run commands continuous).
    env = BuildingEnv(spec, rom, exo, 1, Ops("torch", dtype=torch.float64), exact=False, emulate_f32=False)
    a_disc = torch.ones((1, len(spec.disc)), dtype=torch.float64)
    base = torch.as_tensor(tr["a_cont"][:3], dtype=torch.float64)

    def window(a):
        env.reset()
        total = 0.0
        for k in range(a.shape[0]):
            _, rew, _, _ = env.step(a[k:k + 1], a_disc)
            total = total + rew.sum()
        return total

    a = base.clone().clamp(-0.95, 0.95).requires_grad_(True)
    try:
        out["gradcheck"] = bool(torch.autograd.gradcheck(window, (a,), eps=1e-6, atol=1e-6, rtol=1e-3))
    except RuntimeError as e:  # report, do not abort the check
        out["gradcheck"] = f"failed: {str(e)[:300]}"
    long = min(288, len(tr["reward"]))
    a = torch.as_tensor(tr["a_cont"][:long], dtype=torch.float64).clamp(-0.95, 0.95).requires_grad_(True)
    t0 = time.time()
    total = window(a)
    total.backward()
    g = a.grad
    out["long_backward"] = {"steps": long, "seconds": time.time() - t0, "grad_finite": bool(torch.isfinite(g).all()),
                            "grad_norm": float(g.norm()), "grad_norm_first_step": float(g[0].norm()),
                            "grad_norm_last_step": float(g[-1].norm())}
    # Speed, float32, no grad.
    speed = {}
    for B in (1, 64, 1024):
        env = BuildingEnv(spec, rom, exo, B, Ops("torch", dtype=torch.float32), emulate_f32=False)
        env.reset()
        a_c = torch.zeros((B, len(spec.cont)), dtype=torch.float32)
        a_d = torch.ones((B, len(spec.disc)), dtype=torch.float32)
        with torch.no_grad():
            for _ in range(3):
                env.step(a_c, a_d)
            t0 = time.time()
            steps = 30
            for _ in range(steps):
                env.step(a_c, a_d)
        dt = time.time() - t0
        speed[str(B)] = {"ms_per_step": dt / steps * 1e3, "env_steps_per_s": B * steps / dt}
    out["speed_cpu_float32"] = speed
    return out


def run_env_check(config_gin: Path, out_dir: Path, policies=("pid_water",), steps=(576,), torch_steps: int = 576,
                  seed: int = 0, log=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    steps = list(steps) if len(steps) == len(policies) else [steps[0]] * len(policies)
    report = {"config": str(config_gin), "policies": {}}
    spec = rom = exo = None
    for name, n_steps in zip(policies, steps):
        t0 = time.time()
        env, meta = load_env(config_gin)
        if spec is None:
            spec = BuildingSpec.from_env(env)
            occupancy = copy.deepcopy(env.building._occupancy)  # pristine: reset() already asks it
        obs0 = env.reset()[0]
        if rom is None:
            t1 = time.time()
            rom = build_rom(core_from_env(env))
            report["rom_build_s"] = time.time() - t1
            report["rom_order"] = rom.order
            exo = Exogenous.from_env(env, spec, max(steps), occupancy)
            report.update({"label": spec.label, "n_zones": spec.n_zones, "n_vavs": spec.n_vavs,
                           "n_ahus": spec.n_ahus, "n_obs": len(spec.obs), "n_cont": len(spec.cont),
                           "n_disc": len(spec.disc), "h": float(exo.h[0])})
            log(f"[env] {spec.label}: Z={spec.n_zones} V={spec.n_vavs} A={spec.n_ahus} r={rom.order} "
                f"(ROM {report['rom_build_s']:.0f}s)")
        tracer = Tracer(env)
        policy = make_policy(name, env, meta, seed)
        t1 = time.time()
        tr = tracer.run(policy, obs0, n_steps)
        tracer.detach()
        res = {"steps": int(len(tr["reward"])), "trace_s": time.time() - t1}
        log(f"  {name}: traced {res['steps']} steps in {res['trace_s']:.0f}s")
        res["T0"] = check_exogenous(spec, exo, tr)
        res["T1"] = check_teacher_forced(spec, exo, tr)
        res["T3"], replica = check_open_loop(spec, rom, exo, tr)
        res["T4"] = check_closed_loop(spec, rom, exo, tr, name, env, meta, seed)
        res["T5"] = check_torch(spec, rom, exo, tr, replica, torch_steps)
        res["seconds"] = time.time() - t0
        report["policies"][name] = res
        t3, t4 = res["T3"], res["T4"]
        log(f"  {name}: T1 reward {res['T1']['reward']['max_abs']:.1e} obs {res['T1']['obs']['max_abs']:.1e} | "
            f"T3 zone max {t3['zone_max_k']:.4f} K reward sum {t3['reward_sum_rel']:.2e} | "
            f"T4 score {t4['score_true']:.5f} vs {t4['score_replica']:.5f} ({t4['score_rel']:.2%})")
        del env, tracer, tr
        (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    report["seconds"] = time.time() - t_start
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--policies", default="pid_water", help="comma-separated: pid_water, bang_bang, excite, ...")
    ap.add_argument("--steps", default="576", help="steps per policy (one value, or one per policy)")
    ap.add_argument("--torch-steps", type=int, default=576)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run_env_check(a.config, a.out_dir, policies=tuple(a.policies.split(",")),
                  steps=tuple(int(s) for s in a.steps.split(",")), torch_steps=a.torch_steps, seed=a.seed,
                  log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
