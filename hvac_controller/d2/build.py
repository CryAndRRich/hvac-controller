"""Direction 2: the building model rebuilt from the gateway's metadata alone, for an unseen building.

What the policy is given is exactly what the scenario generator builds a building from: the floor plan and zone map
(the same arrays, stacked floors included), the AHU -> room assignment and the schedule; every equipment constant is
the generator's template and identical across buildings. What it is not given are six envelope numbers and the
station's convection coefficient h. They are set to the centre of the ranges the generator draws from
(smart_control/utils/materials.py; h of the shipped stations spans 8.8-26.7), and the model is corrected online from
the measured zone temperatures (mpc.MPCPolicy's observer).

``model_from_meta`` writes the arrays, generates a gin config with those materials and a placeholder sinusoid weather,
loads that environment, reads the spec and thermal core from it (d1/spec.py, d1/core.py) and builds the ROM
(d1/rom.py). Building a ROM takes ~20 s for a 39-zone building and minutes for the largest, so it is cached per floor
plan: the five weather strips of one building reuse it.

``exogenous`` builds what the model needs per step that no action changes, from the schedule alone: comfort mode and
setpoint windows, the tariff and carbon tables, and the expected occupancy (an occupant is present with the
probability PRESENCE[step of day] on working days). Outside temperature is not known ahead: it is a placeholder here
and is overwritten online with the measured value and a forecast.
"""

from __future__ import annotations

import hashlib
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Centre of each drawn property's range (geometric for the log-sampled ones), materials.PROPERTIES.
H_PRIOR = math.sqrt(8.8 * 26.7)
# Probability that an occupant is present, per 5-minute step of a working day from midnight: the occupancy model
# (randomized_arrival_departure_occupancy.py) averaged over the shipped scenarios' working days.
_HOURLY = [0.0] * 6 + [0.06, 0.34, 0.56, 0.72, 0.84, 0.91, 0.96, 0.94, 0.66, 0.48, 0.35, 0.24, 0.17] + [0.0] * 5
PRESENCE = np.interp(np.arange(288) / 12.0, np.arange(25), _HOURLY + [0.0])
REF_MONDAY = pd.Timestamp("2023-04-03 00:00")  # a Monday with no holiday in the 31 days after it

_CACHE: dict[str, tuple] = {}


def prior_materials() -> dict[str, float]:
    from smart_buildings.smart_control.utils.materials import PROPERTIES

    return {name: (math.sqrt(lo * hi) if scale == "log" else 0.5 * (lo + hi))
            for name, (lo, hi, scale) in PROPERTIES.items()}


def _key(meta: dict, materials: dict, h: float) -> str:
    m = hashlib.sha1()
    m.update(np.ascontiguousarray(meta["floor_plan"]).tobytes())
    m.update(np.ascontiguousarray(meta["zone_map"]).tobytes())
    m.update(repr(sorted(meta["ahu_zone_assignments"].items())).encode())
    m.update(repr((sorted(materials.items()), round(h, 6))).encode())
    return m.hexdigest()


def model_from_meta(meta: dict, materials: dict | None = None, h: float = H_PRIOR, log=None):
    """(spec, rom, env) of the building the metadata describes, with the given materials and h. Cached."""
    materials = materials or prior_materials()
    key = _key(meta, materials, h)
    if key in _CACHE:
        return _CACHE[key]
    from hvac_controller.d1.core import core_from_env
    from hvac_controller.d1.rom import build_rom
    from hvac_controller.d1.spec import BuildingSpec
    from hvac_controller.sim import load_env
    from smart_buildings.smart_control.utils import scenario_generator as sg

    work = Path(tempfile.mkdtemp(prefix="d2_model_"))
    np.save(work / "floor_plan.npy", np.asarray(meta["floor_plan"]))
    np.save(work / "zone_map.npy", np.asarray(meta["zone_map"]))
    ahus = sorted(meta["ahu_zone_assignments"], key=lambda a: int(a.rsplit("_", 1)[1]))
    lo_c, hi_c = (float(v) for v in meta["comfort_temp_window"])
    lo_e, hi_e = (float(v) for v in meta["eco_temp_window"])
    config = sg.ScenarioConfig(
        name="d2_model", write_metrics_dir=False, materials=sg.MaterialsConfig(**materials),
        weather=sg.WeatherConfig(convection_coefficient=float(h)),
        setpoint_schedule=sg.SetpointScheduleConfig(
            morning_start_hour=int(meta["morning_start_hour"]), evening_start_hour=int(meta["evening_start_hour"]),
            heating_setpoint_day=lo_c, cooling_setpoint_day=hi_c, heating_setpoint_night=lo_e,
            cooling_setpoint_night=hi_e, time_zone=str(meta.get("time_zone") or "US/Pacific")),
        simulation=sg.SimulationConfig(start_timestamp=str(REF_MONDAY + pd.Timedelta(hours=6))),
    )
    gin_text = sg._generate_gin_config(str(work / "floor_plan.npy"), str(work / "zone_map.npy"),
                                       [list(meta["ahu_zone_assignments"][a]) for a in ahus], config)
    (work / "config.gin").write_text(gin_text)
    env, _ = load_env(work / "config.gin")
    spec = BuildingSpec.from_env(env)
    env.reset()
    core = core_from_env(env)
    rom, info = build_rom(core, log=log)
    _CACHE[key] = (spec, rom, env)
    return _CACHE[key]


def exogenous(env, spec, start: pd.Timestamp, n_steps: int, occupants_per_zone: float, t_amb: float):
    """Exogenous series (d1/spec.Exogenous) from ``start`` on, from the schedule and expectations alone."""
    from hvac_controller.d1.spec import Exogenous
    from smart_buildings.smart_control.utils import conversion_utils

    hvac = env.building.simulator._hvac
    schedule, rf = hvac.schedule, env.reward_function
    ec, gc = rf._electricity_energy_cost, rf._natural_gas_energy_cost
    dt = pd.Timedelta(spec.dt, unit="s")
    times = [start + k * dt for k in range(n_steps + 2)]
    comfort = np.array([float(hvac.is_comfort_mode(t)) for t in times])
    workday = np.array([float(conversion_utils.is_work_day(t)) for t in times])
    sod = np.array([(t.hour * 60 + t.minute) // 5 for t in times]) % 288
    present = workday * PRESENCE[sod] * occupants_per_zone  # expected occupants per zone
    Z = spec.n_zones
    occ = np.repeat(present[:, None], Z, axis=1)
    price_e, carbon_e, price_g, carbon_g, max_cost, max_carbon = (np.zeros(n_steps) for _ in range(6))
    for k in range(n_steps):
        a, b = times[k + 1], times[k + 2]
        price_e[k], carbon_e[k] = ec.cost(a, b, 1.0), ec.carbon(a, b, 1.0)
        price_g[k], carbon_g[k] = gc.cost(a, b, 1.0), gc.carbon(a, b, 1.0)
        max_cost[k] = ec.cost(a, b, spec.reward["max_elec"]) + gc.cost(a, b, spec.reward["max_gas"])
        max_carbon[k] = ec.carbon(a, b, spec.reward["max_elec"]) + gc.carbon(a, b, spec.reward["max_gas"])
    win = np.array([schedule.get_temperature_window(t) for t in times], dtype=float)
    h = env.building.simulator._weather_controller.get_air_convection_coefficient(times[0])
    return Exogenous(
        t_amb=np.full(n_steps, t_amb), t_amb_next=np.full(n_steps, t_amb), h=np.full(n_steps, float(h)),
        comfort=comfort[:n_steps], win_now=win[:n_steps], win_next=win[1:n_steps + 1],
        occ_step=occ[:n_steps], occ_rew=occ[1:n_steps + 1], price_e=price_e, carbon_e=carbon_e, price_g=price_g,
        carbon_g=carbon_g, max_cost=max_cost, max_carbon=max_carbon, reward_dt=np.full(n_steps, spec.dt),
        obs_exo=np.zeros((n_steps + 1, len(spec.exo_fields))),
    )


def start_from_obs(obs: np.ndarray, field_names: list[str]) -> pd.Timestamp:
    """A timestamp in the reference weeks with the observation's day of week and time of day (5-minute steps)."""
    def angle(prefix):
        s = obs[field_names.index(f"{prefix}_sin_000")]
        c = obs[field_names.index(f"{prefix}_cos_000")]
        return math.atan2(s, c) % (2.0 * math.pi) / (2.0 * math.pi)

    minutes = round(angle("hod") * 24 * 12) * 5 % (24 * 60)
    day = round(angle("dow") * 7) % 7
    return REF_MONDAY + pd.Timedelta(days=day, minutes=minutes)
