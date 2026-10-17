"""Probes run once on Kaggle before the data pipeline is designed.

Each probe answers one question the pipeline depends on and returns a JSON-able dict. ``run_all``
runs them in order, records any exception instead of stopping, and returns one report.

- environment: CPU count, library versions, repo commit.
- step_time: seconds per simulator step on one shipped scenario.
- determinism: are two rollouts with the same policy identical (same env after reset, fresh env)?
- snapshot: can the environment be deep-copied mid-episode, and does the copy evolve identically?
- ahu_off: with every AHU commanded off, do airflow and reheat still reach the zones while fan
  energy bills as zero (the suspected defect)?
- generator: does the scenario generator rebuild a shipped scenario exactly?
- new_weather: does a scenario built from a TMYx station of the packed dataset run, and does the
  simulator read back the temperatures we wrote?

All rollouts use the ``pid_water`` baseline so the plant runs as it would under a real controller.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from hvac_controller import weather
from hvac_controller.sim import load_env, snapshot
from smart_buildings import evaluate
from smart_buildings.smart_control.utils import scenario_generator as sg

POLICY = "pid_water"
REWARD_FIELDS = (
    "agent_reward_value",
    "productivity_reward",
    "electricity_energy_cost",
    "natural_gas_energy_cost",
    "carbon_emitted",
    "normalized_productivity_regret",
    "normalized_energy_cost",
    "normalized_carbon_emission",
    "total_occupancy",
)
EPOCH_2023 = np.datetime64("2023-01-01T00:00")


# ---------------------------------------------------------------------------- helpers

def make_policy(env, meta):
    return evaluate.initialize_policy(evaluate.POLICY_MAP[POLICY], meta, env.action_spec())


def reward_parts(env) -> list[float]:
    _, r = env.get_reward_info_and_response()
    return [float(getattr(r, f)) for f in REWARD_FIELDS]


def ahus(env):
    return env.building.simulator._hvac.air_handler._ahus


def vavs(env):
    return env.building.simulator._hvac._vavs


def plant_state(env) -> dict:
    return {
        "ahu_run": [int(a.run_command) for a in ahus(env)],
        "ahu_flow": [float(a.air_flow_rate) for a in ahus(env)],
        "ahu_static_pressure": [float(a.supply_air_static_pressure_sensor) for a in ahus(env)],
        "vav_flow_demand_sum": float(sum(v.flow_rate_demand for v in vavs(env).values())),
        "vav_reheat_mean": float(np.mean([v.reheat_valve_setting for v in vavs(env).values()])),
        "zone_temp_mean": float(np.mean([v.zone_air_temperature for v in vavs(env).values()])),
    }


def step(env, policy, obs, ahu_off: bool = False, n_ahu: int = 0):
    action = evaluate.validate_action(evaluate.get_policy_action(policy, obs), env.action_spec())
    if ahu_off:
        action["discrete_action"][:n_ahu] = 0
    obs, reward, terminated, truncated, _ = env.step(action)
    return np.asarray(obs, dtype=np.float64), float(reward), bool(terminated or truncated)


def rollout(env, policy, n: int, obs=None) -> dict:
    """n steps from the current state (reset first when obs is None)."""
    if obs is None:
        obs, _ = env.reset()
        obs = np.asarray(obs, dtype=np.float64)
    out_obs, out_r, parts = [], [], []
    t0 = time.time()
    for _ in range(n):
        obs, r, done = step(env, policy, obs)
        out_obs.append(obs)
        out_r.append(r)
        parts.append(reward_parts(env))
        if done:
            break
    return {"obs": np.array(out_obs), "reward": np.array(out_r), "parts": np.array(parts),
            "seconds": time.time() - t0, "last_obs": obs}


def max_diff(a: dict, b: dict) -> dict:
    n = min(len(a["reward"]), len(b["reward"]))
    return {
        "steps_compared": n,
        "obs": float(np.max(np.abs(a["obs"][:n] - b["obs"][:n]))) if n else None,
        "reward": float(np.max(np.abs(a["reward"][:n] - b["reward"][:n]))) if n else None,
        "first_obs_diff_step": int(np.argmax(np.any(a["obs"][:n] != b["obs"][:n], axis=1)))
        if n and np.any(a["obs"][:n] != b["obs"][:n]) else None,
    }


# ---------------------------------------------------------------------------- probes

def probe_environment(repo_dir: Path) -> dict:
    import gin
    import pandas as pd

    try:
        commit = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = None
    return {
        "cpu_count": os.cpu_count(),
        "sched_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "gin": getattr(gin, "__version__", None),
        "repo_commit": commit,
    }


def probe_step_time(config_gin: Path, n: int = 288) -> dict:
    t0 = time.time()
    env, meta = load_env(config_gin)
    load_s = time.time() - t0
    run = rollout(env, make_policy(env, meta), n)
    return {
        "scenario": str(config_gin),
        "load_seconds": load_s,
        "steps": len(run["reward"]),
        "seconds_per_step": run["seconds"] / max(len(run["reward"]), 1),
        "projected_minutes_8640_steps": run["seconds"] / max(len(run["reward"]), 1) * 8640 / 60,
        "n_ahu": len(meta["ahu_zone_assignments"]),
        "n_zone": len(meta["zone_temp_indices"]),
        "obs_dim": len(meta["observation_field_names"]),
        "n_discrete": meta["n_discrete_action"],
        "n_continuous": meta["n_continuous_action"],
        "sum_reward": float(run["reward"].sum()),
    }


def probe_determinism(config_gin: Path, n: int = 288) -> dict:
    env, meta = load_env(config_gin)
    a = rollout(env, make_policy(env, meta), n)
    b = rollout(env, make_policy(env, meta), n)  # same env, after reset
    env2, meta2 = load_env(config_gin)
    c = rollout(env2, make_policy(env2, meta2), n)  # fresh env
    return {
        "same_env_after_reset": max_diff(a, b),
        "fresh_env": max_diff(a, c),
        "identical_after_reset": bool(np.array_equal(a["obs"], b["obs"]) and np.array_equal(a["reward"], b["reward"])),
        "identical_fresh_env": bool(np.array_equal(a["obs"], c["obs"]) and np.array_equal(a["reward"], c["reward"])),
    }


def probe_snapshot(config_gin: Path, warm: int = 144, n: int = 144) -> dict:
    env, meta = load_env(config_gin)
    policy = make_policy(env, meta)
    warm_run = rollout(env, policy, warm)
    out: dict = {"warm_steps": len(warm_run["reward"])}
    t0 = time.time()
    env_copy, policy_copy = snapshot(env, policy)
    out["deepcopy_seconds"] = time.time() - t0
    t0 = time.time()
    env_copy2, policy_copy2 = snapshot(env, policy)
    out["second_snapshot_seconds"] = time.time() - t0
    # Size of what a snapshot copies (the shared factorization excluded), as a pickled estimate.
    try:
        lu = env.building.simulator._solver._factorization if hasattr(env.building.simulator, "_solver") else None
        out["shares_factorization"] = lu is not None and (
            env_copy.building.simulator._solver._factorization is lu)
    except Exception as e:  # noqa: BLE001
        out["shares_factorization"] = f"{type(e).__name__}: {e}"
    obs = warm_run["last_obs"]
    a = rollout(env, policy, n, obs=obs)
    b = rollout(env_copy, policy_copy, n, obs=obs.copy())
    c = rollout(env_copy2, policy_copy2, n, obs=obs.copy())
    out["original_vs_copy"] = max_diff(a, b)
    out["identical"] = bool(np.array_equal(a["obs"], b["obs"]) and np.array_equal(a["reward"], b["reward"]))
    out["identical_second_copy"] = bool(np.array_equal(a["obs"], c["obs"]) and np.array_equal(a["reward"], c["reward"]))
    out["branch_seconds_per_step"] = b["seconds"] / max(len(b["reward"]), 1)
    return out


def probe_ahu_off(config_gin: Path, warm: int = 144, n: int = 12) -> dict:
    """Branch at the same state: A keeps the baseline's commands, B forces every AHU off."""
    env, meta = load_env(config_gin)
    policy = make_policy(env, meta)
    n_ahu = len(ahus(env))
    obs = rollout(env, policy, warm)["last_obs"]
    env_b, policy_b = snapshot(env, policy)
    obs_b = obs.copy()
    log = {"on": [], "off": []}
    for _ in range(n):
        obs, _, _ = step(env, policy, obs)
        log["on"].append({**plant_state(env), "reward": dict(zip(REWARD_FIELDS, reward_parts(env)))})
        obs_b, _, _ = step(env_b, policy_b, obs_b, ahu_off=True, n_ahu=n_ahu)
        log["off"].append({**plant_state(env_b), "reward": dict(zip(REWARD_FIELDS, reward_parts(env_b)))})

    def mean(branch, key, sub=None):
        vals = [s[key] if sub is None else s[key][sub] for s in log[branch]]
        return float(np.mean([np.sum(v) for v in vals]))

    summary = {
        b: {
            "ahu_run_sum": mean(b, "ahu_run"),
            "ahu_flow_sum": mean(b, "ahu_flow"),
            "vav_flow_demand_sum": mean(b, "vav_flow_demand_sum"),
            "vav_reheat_mean": mean(b, "vav_reheat_mean"),
            "zone_temp_mean_end": log[b][-1]["zone_temp_mean"],
            "electricity_cost": mean(b, "reward", "electricity_energy_cost"),
            "gas_cost": mean(b, "reward", "natural_gas_energy_cost"),
        }
        for b in ("on", "off")
    }
    return {"warm_steps": warm, "branch_steps": n, "n_ahu": n_ahu, "summary": summary, "log": log}


def probe_generator(repo_dir: Path, shipped_scenarios: Path, out_dir: Path,
                    building: str = "building_train_1", weather_name: str = "atlanta_2") -> dict:
    """Regenerate one shipped scenario and compare it file by file with the shipped copy."""
    t0 = time.time()
    config = sg.load_scenario_from_parts(
        str(repo_dir / "smart_buildings" / "buildings" / f"{building}.yaml"), weather_name,
        output_base_dir=str(out_dir),
    )
    result = sg.generate_scenario_from_config(config)
    seconds = time.time() - t0
    ours, theirs = Path(result["output_dir"]), shipped_scenarios / building / weather_name
    cmp = {}
    for f in ("floor_plan.npy", "zone_map.npy"):
        cmp[f] = bool(np.array_equal(np.load(ours / f), np.load(theirs / f)))

    def gin_lines(p: Path) -> list[str]:
        # Paths differ by output directory only; compare everything else.
        return [l for l in p.read_text().splitlines() if "_filepath" not in l and "metrics_path" not in l]

    ga, gb = gin_lines(ours / "config.gin"), gin_lines(theirs / "config.gin")
    cmp["config.gin_except_paths"] = ga == gb
    if ga != gb:
        cmp["gin_first_diff"] = next(((x, y) for x, y in zip(ga, gb) if x != y), None)
    return {"seconds": seconds, "num_rooms": result["num_rooms"], "num_ahus": result["num_ahus"], "identical": cmp}


def probe_new_weather(repo_dir: Path, packed_root: Path, out_dir: Path, stem: str | None = None,
                      building: str = "building_train_1", start: str = "2023-07-10 06:00", n: int = 288) -> dict:
    packed = weather.Packed(packed_root)
    shipped = set(weather.SHIPPED_STEMS.values())
    if stem is None:
        preferred = "USA_TX_Houston-Bush.Intercontinental.AP.722430"
        stem = preferred if preferred in packed.table else next(
            s for s, r in packed.table.items() if r["country"] == "USA" and s not in shipped)
    weather_dir = repo_dir / "smart_buildings" / "weather"
    yaml_path = weather.write_preset(packed, stem, weather_dir, start)
    config = sg.load_scenario_from_parts(
        str(repo_dir / "smart_buildings" / "buildings" / f"{building}.yaml"), str(yaml_path),
        output_base_dir=str(out_dir),
    )
    result = sg.generate_scenario_from_config(config)
    gin_path = Path(result["gin_config_path"])
    env, meta = load_env(gin_path)
    policy = make_policy(env, meta)
    obs, _ = env.reset()
    obs = np.asarray(obs, dtype=np.float64)
    x100 = packed.x100(stem)
    wc = env.building.simulator.weather_controller
    temp_err, rewards = [], []
    for _ in range(n):
        ts = env.building.current_timestamp
        naive = ts.tz_convert(None) if getattr(ts, "tzinfo", None) else ts
        if naive.minute == 0:
            hour = int((np.datetime64(naive.to_datetime64(), "h") - EPOCH_2023) / np.timedelta64(1, "h"))
            expected_k = (x100[hour, 0] / 100) + 273.15
            temp_err.append(abs(float(wc.get_current_temp(ts)) - expected_k))
        obs, r, done = step(env, policy, obs)
        rewards.append(r)
        if done:
            break
    rewards = np.array(rewards)
    return {
        "stem": stem,
        "station": packed.table[stem]["name"],
        "convection_coefficient": float(yaml_path.read_text().split("convection_coefficient:")[1].split()[0]),
        "steps": len(rewards),
        "rewards_finite": bool(np.all(np.isfinite(rewards))),
        "sum_reward": float(rewards.sum()),
        "hourly_temp_checks": len(temp_err),
        "max_abs_temp_error_k": float(max(temp_err)) if temp_err else None,
    }


# ---------------------------------------------------------------------------- driver

def run_all(repo_dir: Path, shipped_scenarios: Path, packed_root: Path | None, work_dir: Path) -> dict:
    repo_dir, shipped_scenarios, work_dir = Path(repo_dir), Path(shipped_scenarios), Path(work_dir)
    cfg = repo_dir / "smart_buildings" / "training_scenarios" / "building_train_1" / "atlanta_2" / "config.gin"
    probes = [
        ("environment", lambda: probe_environment(repo_dir)),
        ("step_time", lambda: probe_step_time(cfg)),
        ("determinism", lambda: probe_determinism(cfg)),
        ("snapshot", lambda: probe_snapshot(cfg)),
        ("ahu_off", lambda: probe_ahu_off(cfg)),
        ("generator", lambda: probe_generator(repo_dir, shipped_scenarios, work_dir / "regen")),
    ]
    if packed_root is not None:
        probes.append(("new_weather", lambda: probe_new_weather(repo_dir, packed_root, work_dir / "new_weather")))
    report: dict = {}
    for name, fn in probes:
        t0 = time.time()
        print(f"[probe] {name} ...", flush=True)
        try:
            report[name] = fn()
        except Exception as e:  # noqa: BLE001 - one failing probe must not hide the others
            report[name] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        report[name]["probe_seconds"] = time.time() - t0
        print(f"[probe] {name} done in {report[name]['probe_seconds']:.0f}s", flush=True)
    if packed_root is None:
        report["new_weather"] = {"error": "packed weather dataset not attached"}
    return report
