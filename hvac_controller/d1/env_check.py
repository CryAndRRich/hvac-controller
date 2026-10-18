"""Step C check: the plant + ROM replica (torch_env.BuildingEnv) against the organisers' environment, tier by tier.

Per scenario: build the ROM and the exogenous series once, then for each policy run the real environment with a
Tracer and compare:

- T0 exogenous: weather, occupancy (step and reward intervals), the observation's time features -> exact;
- T1 teacher-forced: plant.pre/post fed the traced state before and after each step and the traced device settings
  -> every intermediate (q, g, Ts, flows, hot water, tank, billed energy, reward terms, observation) to rounding;
  plus decode(action) against the settings the devices hold;
- T3 open loop: the replica alone, fed only the traced actions -> zone temperatures, rewards, energy;
- T4 closed loop: the same policy acting on the replica's own observations -> scenario score (the gate), with where
  and in which component the two trajectories part;
- T5 torch: T3 again in torch float64 / float32, gradcheck on a relaxed window, a long backward pass, step timing;
- T6 noise floor (optional): the real environment against itself, the policy seeing its observations plus Gaussian
  noise of the size of the replica's observation error -> how far the score moves when nothing but the observations
  changes. A closed-loop gap inside that spread is the policy's sensitivity, not a replica error. The replica runs
  under the same noise draws, so the two ensembles are compared pairwise.
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
    out = {"obs0": o.to_numpy(obs)[0], "reward": [], "zone": [], "elec": [], "gas": [], "regret": [], "cost": [],
           "carbon": [], "obs": [], "tank": [], "a_cont": [], "a_disc": []}
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
        out["cost"].append(float(o.to_numpy(post["norm_cost"])[0]))
        out["carbon"].append(float(o.to_numpy(post["norm_carbon"])[0]))
        out["tank"].append(float(o.to_numpy(post["tank"])[0]))
        out["obs"].append(o.to_numpy(obs)[0])
        out["a_cont"].append(np.asarray(a_c, dtype=np.float64))
        out["a_disc"].append(np.asarray(a_d, dtype=np.float64))
        if done:
            break
    return {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in out.items()}


def _traj(tr: dict) -> dict:
    """A trace in the layout _rollout returns, for the comparisons below."""
    return {"reward": tr["reward"], "zone": tr["zone_post"], "a_cont": tr["a_cont"], "a_disc": tr["a_disc"],
            "obs": tr["obs"], "obs0": tr["obs0"], "regret": tr["rr_normalized_productivity_regret"],
            "cost": tr["rr_normalized_energy_cost"], "carbon": tr["rr_normalized_carbon_emission"]}


def _score(t: dict) -> dict:
    """Scenario score and the per-step means of the three reward terms (before their weights)."""
    n = len(t["reward"])
    return {"score": float(abs(np.sum(t["reward"])) / n), "regret": float(np.mean(t["regret"])),
            "cost": float(np.mean(t["cost"])), "carbon": float(np.mean(t["carbon"])), "steps": int(n)}


def _first(mask) -> int:
    """First step (axis 0) where mask holds anywhere, -1 if never."""
    rows = np.flatnonzero(np.asarray(mask).reshape(len(mask), -1).any(1))
    return int(rows[0]) if rows.size else -1


def _action_name(spec, rows, j: int) -> str:
    target, i = rows[j][0], rows[j][1]
    return f"{target}[{i}]"


def _divergence(spec, exo, ref: dict, run: dict, act_tol: float = 1e-3) -> dict:
    """Where and how far a closed-loop trajectory leaves a reference one (both in the _rollout layout)."""
    n = min(len(ref["reward"]), len(run["reward"]))
    da = np.abs(np.asarray(run["a_cont"][:n], dtype=np.float64) - np.asarray(ref["a_cont"][:n], dtype=np.float64))
    dd = np.asarray(run["a_disc"][:n]) != np.asarray(ref["a_disc"][:n])
    dz = np.abs(np.asarray(run["zone"][:n]) - np.asarray(ref["zone"][:n]))
    occ = np.asarray(exo.occ_rew[:n]) > 0.0
    k_c, k_d = _first(da > act_tol), _first(dd)
    out = {"first_cont_step": k_c, "first_disc_step": k_d,
           "first_cont_action": _action_name(spec, spec.cont, int(np.argmax(da[k_c]))) if k_c >= 0 else "",
           "first_disc_action": _action_name(spec, spec.disc, int(np.argmax(dd[k_d]))) if k_d >= 0 else "",
           "disc_differ_frac": float(dd.any(1).mean()) if dd.size else 0.0,
           "first_zone_0p1k_step": _first(dz > 0.1), "first_zone_1k_step": _first(dz > 1.0),
           "zone_max_k": float(dz.max()), "zone_mae_k": float(dz.mean()),
           "zone_max_k_occupied": float(dz[occ].max()) if occ.any() else 0.0,
           "zone_mae_k_occupied": float(dz[occ].mean()) if occ.any() else 0.0}
    if k_c >= 0:  # the observations that produced the first diverging action: obs0 at step 0, else obs[k - 1]
        o_ref = ref["obs0"] if k_c == 0 else ref["obs"][k_c - 1]
        o_run = run["obs0"] if k_c == 0 else run["obs"][k_c - 1]
        out["obs_err_before_first_cont"] = float(np.max(np.abs(np.asarray(o_run, dtype=np.float64)
                                                               - np.asarray(o_ref, dtype=np.float64))))
    z = int(np.argmax(dz.max(0)))
    big = dz[:, z] > 1.0
    out.update({"worst_zone": spec.zone_names[z], "worst_zone_k": float(dz[:, z].max()),
                "worst_zone_occupied_share_of_gap": float(occ[big, z].mean()) if big.any() else 0.0,
                "worst_zone_occupancy_mean": float(np.mean(exo.occ_rew[:n, z]))})
    for c in ("reward", "regret", "cost", "carbon"):
        out[f"{c}_sum_rel"] = _sum_rel(run[c][:n], ref[c][:n])
    return out


class NoisyPolicy:
    """A policy that sees its observations plus Gaussian noise (normalised units) on the masked fields.

    The draws depend only on the seed and the step, not on the observation, so a real and a replica run with the
    same seed see the same noise. sigma = 0 adds exact zeros and leaves the policy's input bit-identical.
    """

    def __init__(self, policy, mask, sigma: float, seed: int):
        self.policy, self.sigma = policy, float(sigma)
        self.mask = np.asarray(mask, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def get_action(self, obs):
        obs = np.asarray(obs)
        noise = self.sigma * self.rng.standard_normal(obs.shape) * self.mask
        return evaluate.get_policy_action(self.policy, obs + noise.astype(obs.dtype))


def obs_noise_mask(spec) -> np.ndarray:
    """The observation fields the replica can get wrong: everything computed from its state ("src").

    Time features, setpoints-by-schedule and the outdoor temperature come from the exogenous series and are exact
    (T0), so they stay clean; perturbing the hour would flip the schedule, which no replica error ever does.
    """
    return np.array([row[0] == "src" for row in spec.obs], dtype=bool)


def _sum_rel(pred, ref) -> float:
    s = float(np.sum(ref))
    return float(abs(np.sum(pred) - s) / abs(s)) if s != 0 else float(abs(np.sum(pred)))


def check_open_loop(spec, rom, exo, tr) -> tuple[dict, dict]:
    n = len(tr["reward"])
    env = BuildingEnv(spec, rom, exo, 1, Ops("numpy"))
    r = _rollout(env, actions=(tr["a_cont"], tr["a_disc"]), n_steps=n)
    elec_true = tr["rw_blower"] + np.abs(tr["rw_cool"]) + tr["rw_pump"]
    zone_err = np.abs(r["zone"] - tr["zone_post"])
    obs_err = np.abs(r["obs"] - tr["obs"])[:, obs_noise_mask(spec)]  # the fields T6 perturbs
    return {
        "zone_max_k": float(zone_err.max()), "zone_mae_k": float(zone_err.mean()),
        "zone_max_k_by_day": [float(zone_err[i:i + 288].max()) for i in range(0, n, 288)],
        "reward_sum_rel": _sum_rel(r["reward"], tr["reward"]), "reward_step_max_abs": _err(r["reward"], tr["reward"])["max_abs"],
        "elec_sum_rel": _sum_rel(r["elec"], elec_true), "gas_sum_rel": _sum_rel(r["gas"], tr["rw_gas"]),
        "regret_sum_rel": _sum_rel(r["regret"], tr["rr_normalized_productivity_regret"]),
        "tank": _err(r["tank"], tr["tank"]), "obs": _err(r["obs"], tr["obs"]), "obs0": _err(r["obs0"], tr["obs0"]),
        "obs_src_mae": float(obs_err.mean()), "obs_src_p99": float(np.quantile(obs_err, 0.99)),
        "score_true": float(abs(np.sum(tr["reward"])) / n), "score_replica": float(abs(np.sum(r["reward"])) / n),
    }, r


def check_closed_loop(spec, rom, exo, tr, policy_name, env_real, meta, seed) -> tuple[dict, dict]:
    n = len(tr["reward"])
    env = BuildingEnv(spec, rom, exo, 1, Ops("numpy"))
    policy = make_policy(policy_name, env_real, meta, seed)
    r = _rollout(env, policy=policy, action_spec=env_real.action_spec(), n_steps=n)
    true, rep = float(abs(np.sum(tr["reward"])) / n), float(abs(np.sum(r["reward"])) / len(r["reward"]))
    real = _traj(tr)
    return {"score_true": true, "score_replica": rep, "score_rel": abs(rep - true) / true if true else rep,
            "steps": int(len(r["reward"])),
            "zone_max_k_vs_true": float(np.abs(r["zone"] - tr["zone_post"][:len(r["zone"])]).max()),
            "parts_true": _score(real), "parts_replica": _score(r),
            "divergence": _divergence(spec, exo, real, r)}, r


def _spread(rels) -> dict:
    v = np.asarray(rels, dtype=np.float64)
    return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "max_abs": float(np.abs(v).max())}


def check_noise_floor(config_gin, spec, rom, exo, base_real: dict, base_rep: dict, policy_name: str, seed: int,
                      sigmas, n_seeds: int, n_steps: int, log=print) -> dict:
    """T6: the real environment's own closed-loop spread under observation noise, and the replica's under the same draws.

    Runs, all closed loop from a fresh environment, n_steps each:
      - real, sigma = 0, once: must reproduce the traced baseline exactly (determinism of env + wrapper);
      - for each sigma > 0 and noise seed: real and replica, the same noise draws for both.
    Scores are relative to the traced (unperturbed) real score.
    """
    mask = obs_noise_mask(spec)
    s0 = _score(base_real)["score"]
    rel = (lambda s: (s - s0) / s0 if s0 else s)
    runs = []
    plan = [(0.0, 0)] + [(float(sg), k) for sg in sigmas if float(sg) > 0.0 for k in range(n_seeds)]
    for sigma, k in plan:
        noise_seed = 7919 * (seed + 1) + k
        t0 = time.time()
        env, meta = load_env(config_gin)
        obs0 = env.reset()[0]
        tracer = Tracer(env)
        policy = NoisyPolicy(make_policy(policy_name, env, meta, seed), mask, sigma, noise_seed)
        real = _traj(tracer.run(policy, obs0, n_steps))
        tracer.detach()
        row = {"sigma": sigma, "seed": k, "real": _score(real), "real_vs_base": _divergence(spec, exo, base_real, real),
               "real_s": time.time() - t0}
        row["real_rel"] = rel(row["real"]["score"])
        if sigma > 0.0:
            rep_env = BuildingEnv(spec, rom, exo, 1, Ops("numpy"))
            policy = NoisyPolicy(make_policy(policy_name, env, meta, seed), mask, sigma, noise_seed)
            rep = _rollout(rep_env, policy=policy, action_spec=env.action_spec(), n_steps=n_steps)
            row["replica"] = _score(rep)
            row["replica_rel"] = rel(row["replica"]["score"])
            row["replica_vs_real"] = _divergence(spec, exo, real, rep)
            del rep
        runs.append(row)
        log(f"  T6 sigma {sigma:.0e} seed {k}: real {row['real']['score']:.5f} ({row['real_rel']:+.2%})"
            + (f" replica {row['replica']['score']:.5f} ({row['replica_rel']:+.2%})" if sigma > 0.0 else "")
            + f" | {row['real_s']:.0f}s")
        del env, tracer, real
    by_sigma = {}
    for sigma in sorted({r["sigma"] for r in runs if r["sigma"] > 0.0}):
        rows = [r for r in runs if r["sigma"] == sigma]
        real_rel = [r["real_rel"] for r in rows]
        rep_rel = [r["replica_rel"] for r in rows]
        pair = [r["replica_rel"] - r["real_rel"] for r in rows]
        by_sigma[f"{sigma:.0e}"] = {
            "real_rel": _spread(real_rel), "replica_rel": _spread(rep_rel), "paired_gap": _spread(pair),
            "ensemble_gap_rel": float(np.mean(rep_rel) - np.mean(real_rel)),
            "ensemble_gap_se": float(np.sqrt(np.var(real_rel, ddof=1) / len(rows) + np.var(rep_rel, ddof=1) / len(rows)))
            if len(rows) > 1 else 0.0}
    det = runs[0]
    return {"mask_fields": int(mask.sum()), "n_obs": int(mask.size), "baseline_score": s0,
            "replica_unperturbed_rel": rel(_score(base_rep)["score"]),
            "determinism": {"score_rel": det["real_rel"], "zone_max_k": det["real_vs_base"]["zone_max_k"],
                            "first_cont_step": det["real_vs_base"]["first_cont_step"]},
            "by_sigma": by_sigma, "runs": runs}


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
                  seed: int = 0, noise_sigmas=(), noise_seeds: int = 3, log=print) -> dict:
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
        res["T4"], closed = check_closed_loop(spec, rom, exo, tr, name, env, meta, seed)
        res["T5"] = check_torch(spec, rom, exo, tr, replica, torch_steps) if torch_steps > 0 else {"skipped": "off"}
        report["policies"][name] = res
        (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
        if noise_sigmas and name != "excite":  # excite never reads its observations
            res["T6"] = check_noise_floor(config_gin, spec, rom, exo, _traj(tr), closed, name, seed, noise_sigmas,
                                          noise_seeds, res["steps"], log=log)
        res["seconds"] = time.time() - t0
        t3, t4 = res["T3"], res["T4"]
        log(f"  {name}: T1 reward {res['T1']['reward']['max_abs']:.1e} obs {res['T1']['obs']['max_abs']:.1e} | "
            f"T3 zone max {t3['zone_max_k']:.4f} K reward sum {t3['reward_sum_rel']:.2e} | "
            f"T4 score {t4['score_true']:.5f} vs {t4['score_replica']:.5f} ({t4['score_rel']:.2%})")
        del env, tracer, tr, closed
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
    ap.add_argument("--noise-sigmas", default="", help="T6: comma-separated observation-noise levels, e.g. 1e-6,1e-4")
    ap.add_argument("--noise-seeds", type=int, default=3, help="T6: noise draws per level")
    a = ap.parse_args()
    run_env_check(a.config, a.out_dir, policies=tuple(a.policies.split(",")),
                  steps=tuple(int(s) for s in a.steps.split(",")), torch_steps=a.torch_steps, seed=a.seed,
                  noise_sigmas=tuple(float(s) for s in a.noise_sigmas.split(",") if s),
                  noise_seeds=a.noise_seeds, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
