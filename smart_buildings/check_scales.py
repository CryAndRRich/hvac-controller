"""Assert every scenario's live reward denominators equal c * K.

    python -m smart_buildings.check_scales [pattern ...]
"""
import sys, os, pathlib
from smart_buildings.evaluate import get_action_indices, load_hybrid_environment
from smart_buildings.smart_control.reward.energy_scale import (
    C_ELECTRICITY_PER_AHU as C_ELEC,
    C_NATURAL_GAS_PER_AHU as C_GAS,
)
# SCEN_DIR lets the same check run over a generated test set without editing it.
SCEN = pathlib.Path(os.environ.get("SCEN_DIR", "smart_buildings/training_scenarios"))
pats = sys.argv[1:] or [""]
files = [f for f in sorted(SCEN.glob("building_*/*/config.gin"))
         if any(p in str(f) for p in pats)]
bad = 0
for f in files:
    env = load_hybrid_environment(f)
    # K is the air handler count. Count the per-AHU supply-air temperature
    # setpoints in the constructed environment, rather than back-deriving from
    # the length of the continuous action. The old `(n_cont - 2) // 2` assumed
    # the action was exactly `2K+2`. That stopped being true once the zone
    # dampers, the reheat valves and AHU recirculation became actions: it is now
    # `3K+2+2V`, so the old form reported K=38 for a 7-AHU building and failed
    # every scenario while the configs were in fact correct. Still derived
    # from the constructed environment, so it still cannot pass vacuously.
    k = len(get_action_indices(env)["temp_indices"])
    rf = env.reward_function
    e, g = float(rf._max_electricity_rate), float(rf._max_natural_gas_rate)
    ok = (e == C_ELEC*k) and (g == C_GAS*k)
    bad += not ok
    print(f"{'OK  ' if ok else 'FAIL'} {f.parent.parent.name}/{f.parent.name:<16} "
          f"K={k}  elec={e:>10,.0f} (want {C_ELEC*k:>10,.0f})  "
          f"gas={g:>10,.0f} (want {C_GAS*k:>10,.0f})")
    del env
print(f"\n{len(files)-bad}/{len(files)} correct" + ("" if not bad else f"  -- {bad} FAILED"))
sys.exit(1 if bad else 0)
