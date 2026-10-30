"""Direction 2: label states with the teacher's decisions, on the replica, for the student to imitate.

The teacher is an oracle MPC (mpc.py, preset ``oracle`` or ``heavy``): it plans on the scenario's own exact model
with the true weather and occupancy, which no submission may do, but nothing limits it offline. Every ``rp`` steps
its tracked state is set to the true state of the replica (env), it plans, and its decision for the coming hour is
recorded with the student's features (features.py) of that moment.

``mode``:
- ``teacher``: the teacher's decisions are executed, so the states are the ones the teacher visits;
- ``dagger``: a student (student.py) acts, the teacher only labels the states the student visits (DAgger). With
  probability ``beta`` per hour the teacher's decision is executed instead, which keeps the states near both.

One ``.npz`` per scenario: features X (N, F), decisions Y (N, 6), step k (N,), the hour's reward R (N,), and the
episode's score.

    python -m hvac_controller.d2.teacher OUT_DIR --caches DIR --manifest manifest_v2.json --tags gen_train \\
        --policy heavy.pa0.hs1 --shard 0 --shards 4 --n-procs 4
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


def label_scenario(cache: str, out: str, policy: str, mode: str, student: str | None, beta: float,
                   steps: int, seed: int) -> dict:
    from hvac_controller.d1.cache import load_replica
    from hvac_controller.d1.ops import Ops
    from hvac_controller.d1.torch_env import BuildingEnv
    from hvac_controller.d2.features import FeatureBuilder
    from hvac_controller.d2.mpc import make_mpc

    t0 = time.time()
    rep = load_replica(Path(cache))
    meta, action_spec = rep["meta"], rep["action_spec"]
    env = BuildingEnv(rep["spec"], rep["rom"], rep["exo"], 1, Ops("numpy"))
    teacher = make_mpc(f"{policy}.lag0", meta, action_spec, cache)
    if teacher.cfg["source"] != "oracle":
        raise ValueError("the teacher plans on the scenario's own model: an oracle preset")
    ll, rp = teacher.ll, teacher.cfg["rp"]
    fb = FeatureBuilder(meta)
    pupil = None
    if mode == "dagger":
        from hvac_controller.d2.student import StudentPolicy

        pupil = StudentPolicy(meta, action_spec, student)
    rng = np.random.default_rng(seed)
    lo, hi = action_spec["continuous_action"].low, action_spec["continuous_action"].high
    obs = env.reset()
    integral = np.zeros((1, ll.n_int), dtype=np.float32)
    X, Y, K, R = [], [], [], []
    total, hour_r, plan, own = 0.0, 0.0, None, True
    n = min(steps, env.n_steps)
    for k in range(n):
        o = obs[0].astype(np.float32)
        f = fb.update(o)
        if k % rp == 0:
            # The teacher plans from the true state, with the executor's PI integrals.
            teacher.track.state = {name: np.array(v, copy=True) for name, v in env.state.items()}
            teacher.track.k, teacher.track.obs = env.k.copy(), env.obs.copy()
            teacher.k, teacher.integral = k, integral.copy()
            teacher._replan(o[None])
            plan = teacher.plan
            if k > 0:
                R.append(hour_r)
            hour_r = 0.0
            X.append(f)
            Y.append(np.asarray(plan[0], dtype=np.float32))
            K.append(k)
            own = pupil is None or rng.random() < beta
            if not own:
                student_dec = pupil.decide(f)
        dec = plan[min(k - teacher.plan_k, len(plan) - 1)] if own else student_dec
        a_c, a_d, integral = ll.act(o[None], integral, np.asarray(dec, dtype=np.float32)[None])
        a_c = np.clip(a_c, lo, hi)
        obs, reward, _, _ = env.step(a_c.astype(np.float64), a_d.astype(np.float64))
        total += float(reward[0])
        hour_r += float(reward[0])
    R.append(hour_r)
    score = abs(total) / n
    np.savez_compressed(out, X=np.array(X), Y=np.array(Y), K=np.array(K), R=np.array(R), score=score)
    return {"cache": cache, "out": out, "score": score, "labels": len(X), "seconds": time.time() - t0}


def _job(kw: dict) -> dict:
    try:
        return label_scenario(**kw)
    except Exception as e:
        return {"cache": kw["cache"], "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--caches", type=Path, required=True, help="folder of <building>__<weather>.pkl caches")
    ap.add_argument("--manifest", type=Path, help="manifest_v2.json: take the scenarios with --tags")
    ap.add_argument("--tags", default="gen_train")
    ap.add_argument("--scenarios", default="", help="or: comma-separated <building>__<weather> labels")
    ap.add_argument("--policy", default="heavy.pa0.hs1")
    ap.add_argument("--mode", default="teacher", choices=["teacher", "dagger"])
    ap.add_argument("--student", default=None)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=8640)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--n-procs", type=int, default=4)
    a = ap.parse_args()
    if a.scenarios:
        labels = a.scenarios.split(",")
    else:
        tags = set(a.tags.split(","))
        labels = [e["label"] for e in json.loads(a.manifest.read_text()) if e.get("tag") in tags]
    labels = sorted(labels)[a.shard::a.shards]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, lab in enumerate(labels):
        out = a.out_dir / f"{lab}__{a.mode}.npz"
        cache = a.caches / f"{lab}.pkl"
        if out.exists() or not cache.exists():
            continue
        jobs.append({"cache": str(cache), "out": str(out), "policy": a.policy, "mode": a.mode, "student": a.student,
                     "beta": a.beta, "steps": a.steps, "seed": i})
    print(f"[teacher] {len(labels)} scenarios in this shard, {len(jobs)} to label on {a.n_procs} processes", flush=True)
    t0 = time.time()
    ctx = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(a.n_procs, mp_context=ctx, max_tasks_per_child=1) as pool:
        for f in cf.as_completed([pool.submit(_job, j) for j in jobs]):
            r = f.result()
            msg = r.get("error") or f"score {r['score']:.5f}, {r['labels']} labels, {r['seconds']:.0f}s"
            print(f"[{time.time() - t0:6.0f}s] {Path(r['cache']).stem}: {msg}", flush=True)
            if "traceback" in r:
                print(r["traceback"], flush=True)


if __name__ == "__main__":
    main()
