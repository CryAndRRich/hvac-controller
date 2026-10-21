"""Scoring a policy: in the organisers' simulator (the truth) and in the replica, over noise ensembles.

The closed loop is chaotic around on/off decisions (D1 doc §2.7): Gaussian noise of 1e-6 (normalised units) on the
state observations moves a 30-day score by up to 6.6 % in hot and humid climates. So a policy is scored by an
ensemble: seed 0 sees its observations as they are, seed k >= 1 sees them plus noise of SIGMA drawn from seed k
(``NoisyPolicy``). The baseline's ensemble is the null group; a candidate is judged per scenario by
z = (its mean - null mean) / sd (``compare``).

Every run is one job, written to its own JSON file as soon as it finishes, so a long sharded evaluation can be
restarted and skips what is done. Jobs run in fresh processes (the simulator keeps gin and occupancy state).

Policies by name: ``k10`` (the baseline, hvac_controller.baseline_k10) or ``pkg:<dir>`` (a submission package: the
KagglePolicy of the d1_policy.py in dir with its d1_policy.pt, so every version runs with its own code; named after
dir).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing
import time
import traceback
from pathlib import Path

import numpy as np

SIGMA = 1e-6


def obs_noise_mask(spec) -> np.ndarray:
    """The observation fields the replica can get wrong: everything computed from its state ("src").

    Time features, setpoints-by-schedule and the outdoor temperature come from the exogenous series and are exact,
    so they stay clean; perturbing the hour would flip the schedule, which no replica error ever does.
    """
    return np.array([row[0] == "src" for row in spec.obs], dtype=bool)


class NoisyPolicy:
    """A policy that sees its observations plus Gaussian noise (normalised units) on the masked fields.

    The draws depend only on the seed and the step, not on the observation, so a real and a replica run with the
    same seed see the same noise.
    """

    def __init__(self, policy, mask, sigma: float, seed: int):
        self.policy, self.sigma = policy, float(sigma)
        self.mask = np.asarray(mask, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def get_action(self, obs):
        from smart_buildings import evaluate

        obs = np.asarray(obs)
        noise = self.sigma * self.rng.standard_normal(obs.shape) * self.mask
        return evaluate.get_policy_action(self.policy, obs + noise.astype(obs.dtype))


def make_policy(name: str, meta: dict, action_spec, spec, seed: int):
    if name == "k10":
        from hvac_controller.baseline_k10 import SubmissionPolicy

        policy = SubmissionPolicy(meta, action_spec)
    elif name.startswith("pkg:"):
        import importlib.util

        d = Path(name.split(":", 1)[1])
        spec_file = importlib.util.spec_from_file_location(f"pkg_{d.name}", d / "d1_policy.py")
        module = importlib.util.module_from_spec(spec_file)
        spec_file.loader.exec_module(module)
        policy = module.KagglePolicy(meta, action_spec, weights=d / "d1_policy.pt")
    else:
        raise ValueError(f"unknown policy {name!r}: k10 or pkg:<dir>")
    return policy if seed == 0 else NoisyPolicy(policy, obs_noise_mask(spec), SIGMA, seed)


def _summary(rewards, terms, seconds: float, policy_s: float) -> dict:
    """Scenario score and the per-step means of the three reward terms (before their weights)."""
    n = len(rewards)
    t = np.asarray(terms, dtype=np.float64).reshape(n, 3)
    return {"score": float(abs(np.sum(rewards)) / n), "regret": float(t[:, 0].mean()), "cost": float(t[:, 1].mean()),
            "carbon": float(t[:, 2].mean()), "steps": n, "seconds": seconds, "policy_ms": policy_s / n * 1e3}


def run_original(config: str, policy: str, seed: int, n_steps: int) -> dict:
    """One episode in the organisers' simulator, from a fresh environment."""
    from hvac_controller.d1.spec import BuildingSpec
    from hvac_controller.sim import load_env
    from smart_buildings import evaluate

    t0 = time.time()
    env, meta = load_env(Path(config))
    spec = BuildingSpec.from_env(env)  # reads attributes only; before reset, as replica_from_env does
    obs, _ = env.reset()
    pol = make_policy(policy, meta, env.action_spec(), spec, seed)
    rf, terms = env.reward_function, []
    compute = rf.compute_reward

    def recorded(info):
        r = compute(info)
        terms.append((r.normalized_productivity_regret, r.normalized_energy_cost, r.normalized_carbon_emission))
        return r

    rf.compute_reward = recorded
    rewards, policy_s = [], 0.0
    for _ in range(n_steps):
        t1 = time.perf_counter()
        action = evaluate.validate_action(evaluate.get_policy_action(pol, np.asarray(obs)), env.action_spec())
        policy_s += time.perf_counter() - t1
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        if terminated or truncated:
            break
    return _summary(rewards, terms, time.time() - t0, policy_s)


def run_replica(cache: str, policy: str, seed: int, n_steps: int) -> dict:
    """One episode in the NumPy replica (float64, protobuf float32 emulation: step C's configuration)."""
    from hvac_controller.d1.cache import load_replica
    from hvac_controller.d1.ops import Ops
    from hvac_controller.d1.torch_env import BuildingEnv
    from smart_buildings import evaluate

    t0 = time.time()
    rep = load_replica(Path(cache))
    spec, action_spec = rep["spec"], rep["action_spec"]
    env = BuildingEnv(spec, rep["rom"], rep["exo"], 1, Ops("numpy"))
    pol = make_policy(policy, rep["meta"], action_spec, spec, seed)
    obs = env.reset()
    rewards, terms, policy_s = [], [], 0.0
    for _ in range(min(n_steps, env.n_steps)):
        t1 = time.perf_counter()
        act = evaluate.validate_action(evaluate.get_policy_action(pol, obs[0].astype(np.float32)), action_spec)
        policy_s += time.perf_counter() - t1
        obs, reward, _, info = env.step(act["continuous_action"][None].astype(np.float64),
                                        act["discrete_action"][None].astype(np.float64))
        post = info["post"]
        rewards.append(float(reward[0]))
        terms.append((post["regret"][0], post["norm_cost"][0], post["norm_carbon"][0]))
    return _summary(rewards, terms, time.time() - t0, policy_s)


def policy_tag(name: str) -> str:
    return Path(name.split(":", 1)[1]).name if name.startswith("pkg:") else name


def job_name(job: dict) -> str:
    return f"{job['where']}__{job['scenario'].replace('/', '__')}__{policy_tag(job['policy'])}__s{job['seed']}"


def _run_job(job: dict) -> dict:
    try:
        if job["where"] == "original":
            res = run_original(job["config"], job["policy"], job["seed"], job["steps"])
        else:
            res = run_replica(job["cache"], job["policy"], job["seed"], job["steps"])
    except Exception as e:  # reported in the job's file; the other jobs go on
        res = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]}
    return {**job, **res}


def run_jobs(jobs: list[dict], out_dir: Path, n_procs: int, log=print) -> list[dict]:
    """Runs the jobs whose result file is missing, n_procs at a time, each in a fresh process."""
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (out_dir / f"{job_name(j)}.json").exists()]
    log(f"[eval] {len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run on {n_procs} processes")
    t0 = time.time()
    ctx = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(n_procs, mp_context=ctx, max_tasks_per_child=1) as pool:
        futures = {pool.submit(_run_job, j): j for j in todo}
        for f in cf.as_completed(futures):
            res = f.result()
            (out_dir / f"{job_name(res)}.json").write_text(json.dumps(res, indent=1))
            msg = res.get("error") or f"score {res['score']:.5f} ({res['seconds']:.0f}s, policy {res['policy_ms']:.2f} ms/step)"
            log(f"[{time.time() - t0:6.0f}s] {job_name(res)}: {msg}")
    return [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]


def ensembles(results: list[dict]) -> dict:
    """(where, scenario, policy) -> mean, sd and the seeds' scores and terms."""
    groups: dict = {}
    for r in results:
        if "error" not in r:
            groups.setdefault((r["where"], r["scenario"], policy_tag(r["policy"])), []).append(r)
    out = {}
    for key, rows in groups.items():
        s = np.array([r["score"] for r in rows])
        out[key] = {"n": len(rows), "mean": float(s.mean()), "sd": float(s.std(ddof=1)) if len(rows) > 1 else 0.0,
                    "scores": s.tolist(), **{t: float(np.mean([r[t] for r in rows])) for t in ("regret", "cost", "carbon")}}
    return out


def compare(ens: dict, candidate: str, null: str, where: str = "original", sd_floor: float = 0.005) -> dict:
    """Per scenario both are scored on: relative gap of the means and z = gap / sd, sd from both ensembles' spreads
    with a floor of sd_floor x the null mean. K10 hardly moves under observation noise (2026-10-19: sd 0 on 27 of 30
    scenarios), and the replica scores it within 0.7 %; the floor keeps z from dividing by zero."""
    out = {}
    for (w, scenario, policy), c in ens.items():
        b = ens.get((w, scenario, null))
        if w != where or policy != candidate or b is None:
            continue
        sd = max(float(np.hypot(c["sd"], b["sd"])), sd_floor * b["mean"])
        out[scenario] = {"candidate": c["mean"], "null": b["mean"], "rel": (c["mean"] - b["mean"]) / b["mean"],
                         "z": (c["mean"] - b["mean"]) / sd, "sd_rel": sd / b["mean"], "n": (c["n"], b["n"])}
    if out:
        out["mean"] = {"candidate": float(np.mean([v["candidate"] for v in out.values()])),
                       "null": float(np.mean([v["null"] for v in out.values()]))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--scenarios", required=True, help="comma-separated building/weather")
    ap.add_argument("--scenario-root", type=Path, required=True, help="training_scenarios directory")
    ap.add_argument("--cache-dir", type=Path, help="replica caches (cache.py), for --where replica")
    ap.add_argument("--policies", default="k10")
    ap.add_argument("--seeds", type=int, default=4, help="seeds 0 (no noise) .. n-1 per scenario and policy")
    ap.add_argument("--where", default="original", help="comma-separated: original, replica")
    ap.add_argument("--steps", type=int, default=8640)
    ap.add_argument("--n-procs", type=int, default=4)
    a = ap.parse_args()
    jobs = []
    for scenario in a.scenarios.split(","):
        for where in a.where.split(","):
            for policy in a.policies.split(","):
                for seed in range(a.seeds):
                    jobs.append({"where": where, "scenario": scenario, "policy": policy, "seed": seed,
                                 "steps": a.steps, "config": str(a.scenario_root / scenario / "config.gin"),
                                 "cache": str(a.cache_dir / f"{scenario.replace('/', '__')}.pkl") if a.cache_dir else None})
    results = run_jobs(jobs, a.out_dir, a.n_procs, log=lambda s: print(s, flush=True))
    ens = ensembles(results)
    for (where, scenario, policy), e in sorted(ens.items()):
        print(f"{where:>8} {scenario:>28} {policy:>10}: {e['mean']:.5f} +- {e['sd']:.5f} (n={e['n']}) "
              f"regret {e['regret']:.4f} cost {e['cost']:.4f} carbon {e['carbon']:.4f}")


if __name__ == "__main__":
    main()
