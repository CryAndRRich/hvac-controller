"""Direction 2: the gateway's policy. One MPC per scenario; the building model is rebuilt from the metadata and cached
per floor plan (build.py), so the five weather strips of a building build it once.

If anything in the model build or the planner fails, the policy falls back to K10 for the rest of the scenario, so
an error costs score, never the submission.

BLAS is held to one thread: the planner's matrices are small, and a multi-threaded BLAS only competes for the
session's few CPUs with the simulator. Every 2880 steps the policy prints its own time against the wall clock, so a
run shows how much of a scenario is the simulator and how much is the policy.
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np

MPC_CONFIG = "meta.rp24.lag6.it2"
_BLAS_LIMIT = None


def _one_blas_thread() -> None:
    global _BLAS_LIMIT
    if _BLAS_LIMIT is None:
        try:
            from threadpoolctl import threadpool_limits

            _BLAS_LIMIT = threadpool_limits(1)
        except Exception:  # no threadpoolctl: leave BLAS as it is
            _BLAS_LIMIT = False


class D2Policy:
    def __init__(self, building_meta_data, action_spec, config: str = MPC_CONFIG):
        from hvac_controller.d2.mpc import make_mpc

        _one_blas_thread()
        self.action_spec, self.fallback = action_spec, None
        t0 = self.t_start = time.time()
        self.steps, self.policy_s = 0, 0.0
        try:
            if config.startswith("d2s:"):  # the distilled student: no model, no planning
                from hvac_controller.d2.student import StudentPolicy

                self.mpc = StudentPolicy(building_meta_data, action_spec, config[4:])
                self.mpc.meta, self.mpc.stats = building_meta_data, {"plan_s": 0.0}
            else:
                self.mpc = make_mpc(config, building_meta_data, action_spec, None)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            self._fall_back(building_meta_data)
        print(f"[d2] {len(building_meta_data['zone_temp_indices'])} zones, model ready in {time.time() - t0:.0f}s",
              flush=True)

    def _fall_back(self, meta) -> None:
        from hvac_controller.baseline_k10 import SubmissionPolicy

        print("[d2] falling back to K10", flush=True)
        self.fallback = SubmissionPolicy(meta, self.action_spec)

    def get_action(self, obs):
        t0 = time.perf_counter()
        action = self._act(obs)
        self.policy_s += time.perf_counter() - t0
        self.steps += 1
        if self.steps % 2880 == 0:
            wall = time.time() - self.t_start
            plan = self.mpc.stats["plan_s"] if self.fallback is None else float("nan")
            print(f"[d2] step {self.steps}: wall {wall:.0f}s, policy {self.policy_s:.0f}s "
                  f"({self.policy_s / self.steps * 1e3:.1f} ms/step), planning {plan:.0f}s, "
                  f"rest (simulator, gateway) {wall - self.policy_s:.0f}s", flush=True)
        return action

    def _act(self, obs):
        if self.fallback is None:
            try:
                action = self.mpc.get_action(obs)
            except Exception:
                traceback.print_exc(file=sys.stdout)
                self._fall_back(self.mpc.meta)
        if self.fallback is not None:
            action = self.fallback.get_action(np.asarray(obs))
        for key, dtype in (("continuous_action", np.float32), ("discrete_action", np.int32)):
            space = self.action_spec[key]
            action[key] = np.clip(action[key], space.low, space.high).astype(dtype)
        return action
