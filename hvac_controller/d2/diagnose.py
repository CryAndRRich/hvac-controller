"""Direction 2: where a policy's score goes, per term and per device, in the organisers' simulator.

The score of a scenario is the mean per-step loss 0.6 |regret| + 0.2 cost / max_cost + 0.2 carbon / max_carbon.
This splits the energy part by device (supply and exhaust fans, cooling coil, hot-water pump, boiler gas), each
device's share of the electricity or gas cost and carbon of that step, and splits every part by whether anybody is
in the building, so energy spent on an empty building shows up as such.

    python -m hvac_controller.d2.diagnose <out.json> --config <scenario>/config.gin --policy d2:meta.rp24.lag6.it2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

PARTS = ("comfort", "fan", "cooling", "pump", "gas")


def diagnose(config: str, policy: str, n_steps: int) -> dict:
    from hvac_controller.d1.evaluate import make_policy
    from hvac_controller.d1.spec import BuildingSpec
    from hvac_controller.sim import load_env
    from smart_buildings import evaluate
    from smart_buildings.smart_control.utils import conversion_utils as cu

    t0 = time.time()
    env, meta = load_env(Path(config))
    spec = BuildingSpec.from_env(env)
    obs, _ = env.reset()
    pol = make_policy(policy, meta, env.action_spec(), spec, 0)
    rf = env.reward_function
    ec, gc = rf._electricity_energy_cost, rf._natural_gas_energy_cost
    w = (rf._productivity_weight, rf._energy_cost_weight, rf._carbon_emission_weight)
    compute = rf.compute_reward
    rows = []

    def recorded(info):
        r = compute(info)
        start = cu.proto_to_pandas_timestamp(info.start_timestamp)
        end = cu.proto_to_pandas_timestamp(info.end_timestamp)
        rates = {"fan": sum(a.blower_electrical_energy_rate for a in info.air_handler_reward_infos.values()),
                 "cooling": sum(abs(a.air_conditioning_electrical_energy_rate)
                                for a in info.air_handler_reward_infos.values()),
                 "pump": sum(b.pump_electrical_energy_rate for b in info.boiler_reward_infos.values()),
                 "gas": sum(b.natural_gas_heating_energy_rate for b in info.boiler_reward_infos.values())}
        max_cost = ec.cost(start, end, rf._max_electricity_rate) + gc.cost(start, end, rf._max_natural_gas_rate)
        max_carbon = ec.carbon(start, end, rf._max_electricity_rate) + gc.carbon(start, end, rf._max_natural_gas_rate)
        row = {"comfort": w[0] * abs(r.normalized_productivity_regret), "occupied": float(r.total_occupancy > 0)}
        for part, rate in rates.items():
            c = gc if part == "gas" else ec
            row[part] = (w[1] * c.cost(start, end, rate) / max_cost + w[2] * c.carbon(start, end, rate) / max_carbon)
            row[f"{part}_kw"] = rate / 1000.0
        rows.append(row)
        return r

    rf.compute_reward = recorded
    total = 0.0
    for _ in range(n_steps):
        action = evaluate.validate_action(evaluate.get_policy_action(pol, np.asarray(obs)), env.action_spec())
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    n = len(rows)
    occ = np.array([r["occupied"] for r in rows]) > 0.5
    out = {"config": config, "policy": policy, "steps": n, "score": abs(total) / n, "n_ahus": spec.n_ahus,
           "n_zones": spec.n_zones, "seconds": time.time() - t0, "occupied_share": float(occ.mean())}
    for part in PARTS:
        v = np.array([r[part] for r in rows])
        out[part] = float(v.sum() / n)  # contribution to the score
        out[f"{part}_empty"] = float(v[~occ].sum() / n)  # ... of which while nobody is in
        if part != "comfort":
            kw = np.array([r[f"{part}_kw"] for r in rows])
            out[f"{part}_kw_mean"] = float(kw.mean())
    out["parts_sum"] = float(sum(out[p] for p in PARTS))  # should equal the score up to rounding
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--config", required=True)
    ap.add_argument("--policy", default="k10")
    ap.add_argument("--steps", type=int, default=8640)
    a = ap.parse_args()
    res = diagnose(a.config, a.policy, a.steps)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    parts = " ".join(f"{p} {res[p]:.5f} (empty {res[p + '_empty']:.5f})" for p in PARTS)
    print(f"[diagnose] {Path(a.config).parent.parent.name}/{Path(a.config).parent.name} {a.policy}: "
          f"score {res['score']:.5f} = {parts} | sum {res['parts_sum']:.5f}", flush=True)


if __name__ == "__main__":
    main()
