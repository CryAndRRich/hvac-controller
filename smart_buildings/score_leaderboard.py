"""Compute the single leaderboard score for a policy from an eval results tree.

The metric is the mean absolute error of the per-scenario mean per-step reward
against a target of zero:

    mean_reward_i = total_reward_i / 8640                     per scenario i
    score         = mean_i |mean_reward_i - 0|                ONE number

LOWER IS BETTER. The per-step reward is never positive -- comfort shortfall,
energy cost and carbon emission are all charges and nothing is credited -- so
the absolute value is just the size of the loss, and zero would mean perfect
comfort at no energy and no carbon cost. No baselines enter the metric.

The mean is flat and equal-weight over every building x weather pair. Grouping
by weather or by building is deliberately avoided, so any per-weather table
printed here is a diagnostic, never the metric.

The four shipped baselines are scored on the same scenarios and printed
alongside, as a reference range to develop against. Their returns come from
baseline_returns.json next to this script -- the four baselines run on all 30
shipped training scenarios with 8640-step (30-day) episodes -- so scoring needs
only your own policy in the results tree:

    python -m smart_buildings.evaluate \\
        --policies transfer --results-dir results/my_run
    python -m smart_buildings.score_leaderboard results/my_run --policy transfer

8640 steps is the --steps default, so the run above is already scoreable.
Returns from different episode lengths are not comparable, so scoring is
refused outright if your policy was run at any other step count. Pass
--baselines self to read the reference returns out of the results tree
instead, which requires the four baselines to have been run alongside your
policy at the same step count.

The count of scored scenarios is printed, so check that number against what
you expected to run.

Usage:
    python -m smart_buildings.score_leaderboard results/my_run --policy transfer
    python -m smart_buildings.score_leaderboard results/my_run --policy transfer --expect 30
    python -m smart_buildings.score_leaderboard results/my_run --policy transfer --baselines self
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# The reference baselines: four hand-written controllers, all of which run the
# plant. They do not enter the metric, which needs no reference policies. They
# are scored on the same scenarios and printed alongside your policy, so you
# can see the range a reasonable controller spans before you start.
#
# Every member is a controller someone might plausibly write. `all_off` is
# runnable via POLICY_MAP but is kept out of this set, because a policy that
# never runs the plant is not a controller anyone would write and tells you
# nothing about where a real one stands.
#
# The set spans the range deliberately. `bang_bang` and `fixed_setpoint` fail
# in opposite climates -- the first in the cold, the second in the heat -- so
# between them they show what going wrong looks like in either direction.
#
# POLICY_MAP and this tuple are deliberately different lists: a policy can ship
# as a readable example without being a reference point.
BASELINES = ("bang_bang", "fixed_setpoint", "pid", "pid_water")

# The episode length the shipped baseline_returns.json was run at. It cannot be
# recorded inside that file, because every top-level key there is read as a
# scenario, so it lives here instead.
BASELINE_STEPS = 8640


def load_tree(root: Path) -> dict[str, dict[str, float]]:
    """{"building/weather": {policy: total_reward}} from summary.json files."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for f in root.glob("building_*/*/*/summary.json"):
        key = f"{f.parts[-4]}/{f.parts[-3]}"
        out[key][f.parts[-2]] = float(json.load(open(f))["total_reward"])
    return dict(out)


def load_steps(root: Path, policy: str) -> tuple[set[int], int]:
    """Episode lengths `policy` was run at, and how many runs did not say."""
    lengths: set[int] = set()
    unknown = 0
    for f in root.glob(f"building_*/*/{policy}/summary.json"):
        n = json.load(open(f)).get("total_steps")
        if n is None:
            unknown += 1
        else:
            lengths.add(int(n))
    return lengths, unknown


def check_episode_length(root: Path, policy: str, baselines_file: Path) -> bool:
    """False if `policy` was not run at the length the shipped ruler assumes.

    A shorter episode is less negative, so scoring one against the shipped
    returns inflates the result silently rather than failing. The step count
    is recorded in every summary.json, so check it rather than trusting the
    caller to have matched it.
    """
    lengths, unknown = load_steps(root, policy)
    wrong = sorted(n for n in lengths if n != BASELINE_STEPS)
    if wrong:
        got = ", ".join(str(n) for n in wrong)
        print(f"ERROR: {policy} was evaluated at {got} step(s), but "
              f"{Path(baselines_file).name} was run at {BASELINE_STEPS}.")
        print("Returns from different episode lengths are not comparable. A "
              "shorter episode is less negative, so this would score too high.")
        print(f"Re-run at --steps {BASELINE_STEPS} (the default), or pass "
              "--baselines self to score against baselines from your own tree.")
        return False
    if unknown:
        print(f"WARNING: {unknown} summary.json file(s) for {policy} have no "
              "total_steps field, so the episode length could not be checked "
              f"against {Path(baselines_file).name}.")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--policy", default="transfer",
                    help="policy key to score (default: transfer)")
    ap.add_argument("--baselines",
                    default=Path(__file__).parent / "baseline_returns.json",
                    help="Path to a json keyed 'building/weather' -> "
                         "{policy: total_reward}. 'self' instead reads the "
                         "baseline returns out of results_dir, which requires "
                         "that the baselines were run alongside the policy. "
                         "(default: the shipped baseline_returns.json next to "
                         "this script)")
    ap.add_argument("--expect", type=int, default=None,
                    help="fail if fewer than this many scenarios scored")
    a = ap.parse_args()

    tree = load_tree(a.results_dir)
    if not tree:
        print(f"no summary.json under {a.results_dir}")
        return 1

    if a.baselines == "self":
        # Baselines come from the same tree as the policy, so both sides share
        # whatever episode length was used and the comparison stays internally
        # consistent. Nothing to check.
        base = {k: {p: v for p, v in d.items() if p in BASELINES}
                for k, d in tree.items()}
    else:
        if not check_episode_length(a.results_dir, a.policy, a.baselines):
            return 1
        base = json.load(open(a.baselines))

    rows, missing_base, missing_pol = [], [], []
    for key in sorted(base if a.baselines != "self" else tree):
        b = base.get(key)
        if not b or not set(BASELINES) <= set(b):
            missing_base.append(key)
            continue
        r = tree.get(key, {}).get(a.policy)
        if r is None:
            missing_pol.append(key)
            continue
        # The metric: |mean per-step reward - 0|. Lower is better.
        rows.append((key, r, abs(r) / BASELINE_STEPS, max(b, key=b.get)))

    if missing_pol:
        print(f"WARNING: {a.policy} missing on {len(missing_pol)} scenario(s): "
              f"{', '.join(missing_pol[:6])}{' ...' if len(missing_pol) > 6 else ''}")
    if missing_base:
        print(f"WARNING: incomplete baseline set on {len(missing_base)} scenario(s)")
    if not rows:
        print("nothing scored")
        return 1

    # The metric.
    score = sum(s for _, _, s, _ in rows) / len(rows)
    print(f"\n=== {a.policy}: LEADERBOARD SCORE = {score:.5f} "
          f"over {len(rows)} scenarios  (lower is better) ===")
    if a.expect and len(rows) < a.expect:
        print(f"INCOMPLETE: {len(rows)}/{a.expect} scenarios -- "
              f"this is NOT the final number")

    # Same metric applied to each baseline, as a reference range.
    print("\nreference baselines on the same scenarios (same metric, lower is better):")
    keys = [k for k, _, _, _ in rows]
    bscore = {p: sum(abs(base[k][p]) / BASELINE_STEPS for k in keys) / len(keys)
              for p in BASELINES}
    for p in sorted(BASELINES, key=lambda p: bscore[p]):
        print(f"  {p:<10} {bscore[p]:.5f}")

    # Diagnostics below this line. Not the metric.
    print("\n-- diagnostic only, NOT the metric (the leaderboard never groups) --")
    byw = defaultdict(list)
    for k, _, s, bb in rows:
        byw[k.split("/")[1]].append((s, bb))
    print(f"{'weather':<17}{'n':>3}{'mean':>9}{'min':>9}{'max':>9}  best baseline")
    for w, v in sorted(byw.items(), key=lambda kv: sum(x[0] for x in kv[1]) / len(kv[1])):
        s = [x[0] for x in v]
        bb = max(set(x[1] for x in v), key=[x[1] for x in v].count)
        print(f"{w:<17}{len(v):>3}{sum(s)/len(s):>9.5f}{min(s):>9.5f}{max(s):>9.5f}  {bb}")

    worst = sorted(rows, key=lambda r: -r[2])[:5]
    print("\nworst 5 scenarios (largest score):")
    for k, r, s, bb in worst:
        print(f"  {k:<32} score {s:>8.5f}  return {r:>10.1f}  best baseline {bb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
