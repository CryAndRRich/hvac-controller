"""Direction 2: the gateway's policy. One MPC per scenario; the building model is rebuilt from the metadata and cached
per floor plan (build.py), so the five weather strips of a building build it once.

If anything in the model build or the planner fails, the policy falls back to the default plan for the rest of the
scenario (the low-level controller on K10's schedule), so an error costs score, never the submission.
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np

MPC_CONFIG = "meta.rp24.lag6"


class D2Policy:
    def __init__(self, building_meta_data, action_spec, config: str = MPC_CONFIG):
        from hvac_controller.d2.mpc import make_mpc

        self.action_spec, self.fallback = action_spec, None
        t0 = time.time()
        try:
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
