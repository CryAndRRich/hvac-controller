"""Direction 2: how long the planner's rollouts take on this machine, before committing a long run to them.

Loads one replica cache, steps a batched float32 torch model of ``--n`` rows driven by the low-level controller, and
projects the cost of one replan and of a whole episode for an MPC configuration (mpc.parse_config).

    python -m hvac_controller.d2.bench <cache.pkl> --mpc oracle --threads 1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache", type=Path)
    ap.add_argument("--mpc", default="oracle")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--steps", type=int, default=48, help="batched steps to time")
    ap.add_argument("--episode", type=int, default=8640)
    a = ap.parse_args()

    from hvac_controller.d1.cache import load_replica
    from hvac_controller.d2.mpc import MPCPolicy, parse_config

    cfg = dict(parse_config(a.mpc), threads=a.threads)
    rep = load_replica(a.cache)
    info = rep["info"]
    t0 = time.perf_counter()
    pol = MPCPolicy(rep["meta"], rep["action_spec"], cfg, rep["spec"], rep["rom"], exo=rep["exo"])
    build_s = time.perf_counter() - t0
    snap = pol._snapshot(torch.as_tensor(np.asarray(pol.model.obs0[:1], dtype=np.float32)))
    cand = torch.rand(cfg["n"], 1, 6)
    pol._rollout(cand, [2], snap)  # warm-up
    t0 = time.perf_counter()
    pol._rollout(cand, [a.steps], snap)
    step_ms = (time.perf_counter() - t0) / a.steps * 1e3
    t0 = time.perf_counter()
    for _ in range(a.steps):
        pol.track.step(np.zeros((1, pol.ll.lay.n_cont)), np.ones((1, pol.ll.lay.n_disc)))
    track_ms = (time.perf_counter() - t0) / a.steps * 1e3
    replan_s = step_ms * cfg["h"] * cfg["it"] / 1e3
    per_episode_min = replan_s * a.episode / cfg["rp"] / 60
    print(f"[bench] {info['label']}: Z={info['n_zones']} V={info['n_vavs']} A={info['n_ahus']} ROM order {info['order']}")
    print(f"[bench] {a.mpc} n={cfg['n']} it={cfg['it']} h={cfg['h']} rp={cfg['rp']} threads={a.threads}: "
          f"build {build_s:.1f}s, batched step {step_ms:.2f} ms, tracker step {track_ms:.2f} ms, "
          f"replan {replan_s:.1f}s, planning {per_episode_min:.0f} min per {a.episode}-step episode", flush=True)


if __name__ == "__main__":
    main()
