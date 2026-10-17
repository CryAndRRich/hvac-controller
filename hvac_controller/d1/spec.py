"""Static description and exogenous inputs of one scenario, read from the organisers' own environment objects.

``BuildingSpec.from_env`` collects everything the plant equations (plant.py) need that does not change during an
episode: the zone / VAV / AHU maps with their floor areas, every equipment constant, the action layout with its
normalisers, the observation layout with its normalisers, and the state the devices are in right after reset.
Nothing is re-parsed from gin: values are read from the objects the scenario's config built, so a scenario with
different constants is picked up as it is.

``Exogenous.from_env`` precomputes, per step, everything that does not depend on the actions: outside temperature,
occupancy, the comfort schedule, tariff and carbon factors and the time features of the observation. Occupancy is a
Markov chain driven by one shared RandomState and advanced only by the order in which the environment asks for it,
so it is regenerated here by asking a pristine copy of the same object in that same order.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from smart_buildings.smart_control.environment import environment as env_lib
from smart_buildings.smart_control.simulator import air_handler as air_handler_py
from smart_buildings.smart_control.simulator import constants as sim_constants
from smart_buildings.smart_control.utils import constants as util_constants
from smart_buildings.smart_control.utils import conversion_utils, regression_building_utils

NON_ZONES = ("exterior_space", "interior_wall")

# Continuous action targets and the discrete ones (run commands).
ACTION_TARGETS = ("ahu_sp", "ahu_p", "ahu_r", "w_sp", "w_dp", "damper", "valve")
DISCRETE_TARGETS = ("ahu_on", "w_on")
_AHU_ACTIONS = {"supervisor_run_command": "ahu_on", "supply_air_temperature_setpoint": "ahu_sp",
                "supply_air_static_pressure_setpoint": "ahu_p", "recirculation_percentage_command": "ahu_r"}
_HWS_ACTIONS = {"supervisor_run_command": "w_on", "supply_water_temperature_setpoint": "w_sp",
                "differential_pressure_setpoint": "w_dp"}
_VAV_ACTIONS = {"supply_air_damper_percentage_command": "damper", "reheat_valve_percentage_command": "valve"}

# Observable measurement -> source key of the raw value (plant.observe builds each source).
_VAV_OBS = {"supply_air_damper_percentage_command": "damper", "reheat_valve_percentage_command": "valve",
            "supply_air_flowrate_setpoint": "vav_max_flow", "zone_air_temperature_sensor": "sens_prev",
            "zone_air_heating_temperature_setpoint_sensor": "vav_heat_sp",
            "zone_air_cooling_temperature_setpoint_sensor": "vav_cool_sp"}
_AHU_OBS = {"cooling_request_count": "ahu_count", "supply_air_static_pressure_setpoint": "ahu_p",
            "supply_air_static_pressure_sensor": "ahu_p", "supply_air_flowrate_setpoint": "ahu_flow",
            "supply_air_flowrate_sensor": "ahu_flow", "supply_air_temperature_sensor": "ahu_sp",
            "supply_air_temperature_setpoint": "ahu_sp", "supply_fan_run_command": "ahu_on",
            "exhaust_fan_run_command": "ahu_on", "supervisor_run_command": "ahu_on",
            "supply_fan_speed_percentage_command": "ahu_speed", "discharge_fan_speed_percentage_command": "ahu_speed",
            "outside_air_flowrate_sensor": "ahu_oa_flow", "recirculation_percentage_command": "ahu_r",
            "outside_air_temperature_sensor": "t_amb_obs"}
_HWS_OBS = {"supply_water_temperature_setpoint": "w_sp", "supply_water_temperature_sensor": "tank",
            "heating_request_count": "w_count", "supervisor_run_command": "w_on", "run_status": "w_on",
            "differential_pressure_setpoint": "w_dp", "differential_pressure_sensor": "w_dp"}
# Width of each source: per VAV, per AHU, or one value.
SOURCE_KIND = {"damper": "vav", "valve": "vav", "vav_max_flow": "vav", "sens_prev": "vav", "vav_heat_sp": "vav",
               "vav_cool_sp": "vav", "ahu_count": "ahu", "ahu_p": "ahu", "ahu_flow": "ahu", "ahu_sp": "ahu",
               "ahu_on": "ahu", "ahu_speed": "ahu", "ahu_oa_flow": "ahu", "ahu_r": "ahu", "w_sp": "one",
               "tank": "one", "w_count": "one", "w_on": "one", "w_dp": "one"}
SOURCES = tuple(SOURCE_KIND)


def _ahus(hvac) -> list:
    ah = hvac.air_handler
    return list(ah.ahus) if isinstance(ah, air_handler_py.AirHandlerSystem) else [ah]


def _split_ahu_field(name: str, single: bool) -> tuple[int, str]:
    if single:
        return 0, name
    m = re.match(r"^ahu_(\d+)_(.+)$", name)
    if not m:
        raise ValueError(f"unexpected air-handler field {name}")
    return int(m.group(1)) - 1, m.group(2)


@dataclass
class BuildingSpec:
    label: str
    dt: float
    zone_names: list[str]
    vav_ids: list[str]
    ahu_ids: list[str]
    zone_cells: np.ndarray  # (Z,) floor area in CVs
    vav_zone_share: np.ndarray  # (Z, V): area_z / area_v for the zones a VAV serves (reheat split, sensing weights)
    vav_ahu: np.ndarray  # (V,) index of each VAV's AHU
    ahu_weight: np.ndarray  # (A, V): area_v / sum of its AHU's VAV areas (recirculation temperature)
    ahu: dict[str, np.ndarray]  # max_flow, max_recirc, min_pressure, fan_eff, cop
    vav: dict[str, np.ndarray]  # max_flow, min_damper, max_discharge, design_pressure, reheat_ff
    hws: dict[str, float]  # header_resistance, pump_eff, boiler constants
    sim: dict[str, float]  # internal gains, occupant heat, cv area
    reward: dict[str, float]
    phys: dict[str, float]  # air/water constants as each module uses them
    # Actions: per continuous column (target, device index, normalised min/max, native min/max); same for discrete.
    cont: list[tuple[str, int, float, float, float, float]]
    disc: list[tuple[str, int, float, float, float, float]]
    # Observations: per field, ("src", source, device index, mean, std) or ("exo", column, 0, 0, 1).
    obs: list[tuple[str, str, int, float, float]]
    exo_fields: list[str]
    occupancy_norm: float
    reset_raw: dict[str, np.ndarray]  # raw value of every observation source right after reset
    tank0: float
    ret0: float
    field_names: list[str] = field(default_factory=list)

    @property
    def n_zones(self) -> int:
        return len(self.zone_names)

    @property
    def n_vavs(self) -> int:
        return len(self.vav_ids)

    @property
    def n_ahus(self) -> int:
        return len(self.ahu_ids)

    @classmethod
    def from_env(cls, env) -> "BuildingSpec":
        sim = env.building.simulator
        hvac = sim._hvac
        grid = sim.building
        room_dict = grid.room_dict
        zone_names = [z for z in room_dict if z not in NON_ZONES and room_dict[z]]
        zidx = {z: i for i, z in enumerate(zone_names)}
        area = {z: len(room_dict[z]) for z in zone_names}
        vav_ids = list(sim._coupled_vavs)
        if vav_ids != [v for v in hvac.vavs if hvac.get_zones_for_vav(v)]:
            raise ValueError("coupled VAVs differ from the VAVs that serve zones")
        ahus = _ahus(hvac)
        single = not isinstance(hvac.air_handler, air_handler_py.AirHandlerSystem)
        Z, V, A = len(zone_names), len(vav_ids), len(ahus)

        share = np.zeros((Z, V))
        vav_area = np.zeros(V)
        vav_ahu = np.zeros(V, dtype=np.int64)
        for j, v in enumerate(vav_ids):
            zs = [z for z in hvac.get_zones_for_vav(v) if z in zidx]
            vav_area[j] = sum(area[z] for z in zs)
            for z in zs:
                share[zidx[z], j] = area[z] / vav_area[j]
            vav_ahu[j] = ahus.index(hvac.vavs[v].air_handler)
        ahu_weight = np.zeros((A, V))
        for i in range(A):
            members = vav_ahu == i
            if not members.any():
                raise ValueError(f"AHU {ahus[i].device_id()} serves no VAV")
            ahu_weight[i, members] = vav_area[members] / vav_area[members].sum()

        vavs = [hvac.vavs[v] for v in vav_ids]
        hws = hvac.hot_water_system
        boiler, pump = hws._heat_source, hws._pump
        if type(boiler).__name__ != "Boiler":
            raise ValueError(f"only a boiler heat source is modelled, not {type(boiler).__name__}")
        rf = env.reward_function

        spec = cls(
            label=str(env._label), dt=float(sim.time_step_sec), zone_names=zone_names, vav_ids=vav_ids,
            ahu_ids=[a.device_id() for a in ahus], zone_cells=np.array([area[z] for z in zone_names], dtype=float),
            vav_zone_share=share, vav_ahu=vav_ahu, ahu_weight=ahu_weight,
            ahu={"max_flow": np.array([a.max_air_flow_rate for a in ahus], dtype=float),
                 "max_recirc": np.array([a._max_recirculation for a in ahus], dtype=float),
                 "min_pressure": np.array([a._min_supply_air_static_pressure for a in ahus], dtype=float),
                 "fan_eff": np.array([a.fan_efficiency for a in ahus], dtype=float),
                 "cop": np.array([a._cooling_cop for a in ahus], dtype=float)},
            vav={"max_flow": np.array([v.max_air_flow_rate for v in vavs], dtype=float),
                 "min_damper": np.array([v.min_damper_setting for v in vavs], dtype=float),
                 "max_discharge": np.array([v._max_discharge_air_temperature for v in vavs], dtype=float),
                 "design_pressure": np.array([v._max_air_flow_static_pressure for v in vavs], dtype=float),
                 "reheat_ff": np.array([v._reheat_max_water_flow_factor for v in vavs], dtype=float)},
            hws={"header_resistance": float(hws._header_resistance), "pump_eff": float(pump._water_pump_efficiency),
                 "heating_rate": float(boiler._heating_rate), "cooling_rate": float(boiler._cooling_rate),
                 "has_tank": float(boiler._has_tank), "water_capacity": float(boiler._water_capacity),
                 "tank_length": float(boiler._tank_length), "tank_radius": float(boiler._tank_radius),
                 "ins_k": float(boiler._insulation_conductivity), "ins_t": float(boiler._insulation_thickness),
                 "conv": float(boiler._convection_coefficient)},
            sim={"gain_w_per_m2": float(sim._internal_gain_w_per_m2),
                 "standby_fraction": float(sim._internal_gain_standby_fraction),
                 "cv_area_m2": (grid.cv_size_cm / 100.0) ** 2,
                 "occupant_w": float(sim._occupant_sensible_heat_w if sim._occupancy is not None else 0.0)},
            reward={"w_prod": float(rf._productivity_weight), "w_cost": float(rf._energy_cost_weight),
                    "w_carbon": float(rf._carbon_emission_weight),
                    "max_pph": float(rf._max_productivity_personhour_usd),
                    "min_pph": float(rf._min_productivity_personhour_usd),
                    "dev_ref": float(rf._productivity_deviation_ref),
                    "max_elec": float(rf._max_electricity_rate), "max_gas": float(rf._max_natural_gas_rate)},
            phys={"air_rho": sim_constants.AIR_DENSITY, "air_cp": sim_constants.AIR_HEAT_CAPACITY,
                  "water_rho": sim_constants.WATER_DENSITY, "water_cp": sim_constants.WATER_HEAT_CAPACITY,
                  "gravity": sim_constants.GRAVITY, "pa_per_bar": util_constants.PASCALS_PER_BAR,
                  "boiler_water_rho": util_constants.WATER_DENSITY, "boiler_water_cp": util_constants.WATER_HEAT_CAPACITY},
            cont=[], disc=[], obs=[], exo_fields=[],
            occupancy_norm=float(env.occupancy_normalization_constant), reset_raw={},
            tank0=float(boiler._current_temperature), ret0=float(boiler._return_water_temperature_sensor),
            field_names=list(env.field_names),
        )
        spec._read_actions(env, hvac, single)
        spec._read_observations(env, hvac, single)
        spec._read_reset(hvac, ahus, vavs)
        return spec

    def _read_actions(self, env, hvac, single: bool) -> None:
        hws_id, ahs_id = hvac.hot_water_system.device_id(), hvac.air_handler.device_id()
        vidx = {v: i for i, v in enumerate(self.vav_ids)}
        seen = set()
        for field_id in env.action_names:
            device_id, setpoint = env._id_map.inv[field_id]
            norm = env.action_normalizers[setpoint]
            if device_id == ahs_id:
                i, name = _split_ahu_field(setpoint, single)
                target = _AHU_ACTIONS[name]
            elif device_id == hws_id:
                i, target = 0, _HWS_ACTIONS[setpoint]
            elif device_id in vidx:
                i, target = vidx[device_id], _VAV_ACTIONS[setpoint]
            else:
                raise ValueError(f"action on unknown device {device_id}")
            row = (target, i, float(norm.min_normalized_value), float(norm.max_normalized_value),
                   float(norm.min_native_value), float(norm.max_native_value))
            (self.disc if env_lib.is_discrete_setpoint(setpoint) else self.cont).append(row)
            seen.add((target, i))
        need = {(t, i) for t in ("ahu_on", "ahu_sp", "ahu_p", "ahu_r") for i in range(self.n_ahus)}
        need |= {(t, 0) for t in ("w_on", "w_sp", "w_dp")}
        need |= {(t, i) for t in ("damper", "valve") for i in range(self.n_vavs)}
        if need - seen:
            raise ValueError(f"actions missing (the local thermostat loop is not modelled): {sorted(need - seen)[:5]}")

    def _read_observations(self, env, hvac, single: bool) -> None:
        hws_id, ahs_id = hvac.hot_water_system.device_id(), hvac.air_handler.device_id()
        vidx = {v: i for i, v in enumerate(self.vav_ids)}
        normalizer = env.observation_normalizer
        aux = list(env._auxiliary_features)
        self.exo_fields = aux + ["t_amb_obs"]
        for field_id in env.field_names:
            if field_id in aux:
                self.obs.append(("exo", field_id, self.exo_fields.index(field_id), 0.0, 1.0))
                continue
            device_id, measurement = env._id_map.inv[field_id]
            consts = normalizer._get_normalization_constants(measurement)
            var = float(consts.sample_variance)
            mean, std = float(consts.sample_mean), (float(np.sqrt(var)) if var > 0.0 else 0.0)
            if device_id == ahs_id:
                i, name = _split_ahu_field(measurement, single)
                source = _AHU_OBS[name]
            elif device_id == hws_id:
                i, source = 0, _HWS_OBS[measurement]
            elif device_id in vidx:
                i, source = vidx[device_id], _VAV_OBS[measurement]
            else:
                raise ValueError(f"observation of unknown device {device_id}")
            if source == "t_amb_obs":
                self.obs.append(("exo_dev", source, self.exo_fields.index("t_amb_obs"), mean, std))
            else:
                self.obs.append(("src", source, i, mean, std))

    def _read_reset(self, hvac, ahus, vavs) -> None:
        """Raw observation sources right after reset, read from attributes (sensor reads have side effects)."""
        hws = hvac.hot_water_system
        boiler, pump = hws._heat_source, hws._pump
        r = self.reset_raw
        r["damper"] = np.array([v.damper_setting for v in vavs], dtype=float)
        r["valve"] = np.array([v.reheat_valve_setting for v in vavs], dtype=float)
        r["vav_max_flow"] = self.vav["max_flow"].copy()
        r["sens_prev"] = np.array([v.zone_air_temperature for v in vavs], dtype=float)
        r["vav_heat_sp"] = np.array([v.zone_air_heating_temperature_setpoint for v in vavs], dtype=float)
        r["vav_cool_sp"] = np.array([v.zone_air_cooling_temperature_setpoint for v in vavs], dtype=float)
        r["ahu_count"] = np.array([a.cooling_request_count for a in ahus], dtype=float)
        r["ahu_p"] = np.array([a.supply_air_static_pressure_setpoint for a in ahus], dtype=float)
        r["ahu_flow"] = np.array([a.air_flow_rate for a in ahus], dtype=float)
        r["ahu_sp"] = np.array([a.supply_air_temperature_setpoint for a in ahus], dtype=float)
        r["ahu_on"] = np.array([int(a.run_command) for a in ahus], dtype=float)
        r["ahu_speed"] = np.array([a.supply_fan_speed_percentage for a in ahus], dtype=float)
        r["ahu_oa_flow"] = np.array([a.ambient_flow_rate for a in ahus], dtype=float)
        r["ahu_r"] = np.array([a.recirculation for a in ahus], dtype=float)
        r["w_sp"] = np.array([boiler._supply_water_temperature_setpoint], dtype=float)
        r["tank"] = np.array([boiler._current_temperature], dtype=float)
        r["w_count"] = np.array([hws.heating_request_count], dtype=float)
        r["w_on"] = np.array([int(hws.run_command)], dtype=float)
        r["w_dp"] = np.array([pump.differential_pressure_setpoint], dtype=float)


@dataclass
class Exogenous:
    """Per-step inputs that no action changes. Step k runs from t_k to t_{k+1}; observation k+1 is taken at t_{k+1}
    (row 0 of obs_exo is the reset observation at t_0)."""

    t_amb: np.ndarray  # (n,) at t_k: mixing air and the thermal step
    t_amb_next: np.ndarray  # (n,) at t_{k+1}: billed cooling, boiler losses
    h: np.ndarray  # (n,)
    comfort: np.ndarray  # (n,) comfort mode at t_k (internal gains)
    win_now: np.ndarray  # (n, 2) heating/cooling setpoints at t_k (VAV setpoint sensors)
    win_next: np.ndarray  # (n, 2) at t_{k+1} (reward)
    occ_step: np.ndarray  # (n, Z) occupancy over [t_k, t_{k+1}) (occupant heat)
    occ_rew: np.ndarray  # (n, Z) over [t_{k+1}, t_{k+2}) (reward)
    price_e: np.ndarray  # (n,) USD per W of electricity over the reward interval
    carbon_e: np.ndarray
    price_g: np.ndarray
    carbon_g: np.ndarray
    max_cost: np.ndarray  # (n,) cost and carbon at the configured maximum rates
    max_carbon: np.ndarray
    reward_dt: np.ndarray  # (n,) seconds
    obs_exo: np.ndarray  # (n + 1, len(exo_fields))
    times: np.ndarray  # (n + 1,) t_k as int64 ns (naive)

    @property
    def n_steps(self) -> int:
        return len(self.t_amb)

    def arrays(self) -> dict[str, np.ndarray]:
        return dict(self.__dict__)

    @classmethod
    def from_env(cls, env, spec: BuildingSpec, n_steps: int, occupancy=None) -> "Exogenous":
        """occupancy: a pristine copy of the scenario's occupancy object, taken before env.reset() asked it anything.
        Defaults to a deep copy of the env's own object, which is only pristine if the env has not been reset."""
        sim = env.building.simulator
        hvac = sim._hvac
        schedule = hvac.schedule
        wc = sim._weather_controller
        occ = copy.deepcopy(occupancy if occupancy is not None else env.building._occupancy)
        rf = env.reward_function
        dt = pd.Timedelta(spec.dt, unit="s")
        t0 = sim.start_timestamp
        times = [t0 + k * dt for k in range(n_steps + 2)]
        obs_zones = [z.zone_id for z in env.building.zones]
        zones = spec.zone_names
        five = pd.Timedelta(5, unit="minute")

        def n_occupants(ts):
            return int(sum(occ.average_zone_occupancy(z, ts - five, ts) for z in obs_zones))

        n_occ_obs = [n_occupants(times[0])]  # env.reset() observation
        occ_step, occ_rew = np.zeros((n_steps, len(zones))), np.zeros((n_steps, len(zones)))
        for k in range(n_steps):
            t, t1, t2 = times[k], times[k + 1], times[k + 2]
            occ_step[k] = [occ.average_zone_occupancy(z, t, t1) for z in zones]  # execute_step_sim
            n_occ_obs.append(n_occupants(t1))  # observation
            occ_rew[k] = [occ.average_zone_occupancy(z, t1, t2) for z in zones]  # reward

        ec, gc = rf._electricity_energy_cost, rf._natural_gas_energy_cost
        price_e, carbon_e, price_g, carbon_g, max_cost, max_carbon, reward_dt = ([] for _ in range(7))
        for k in range(n_steps):
            start = conversion_utils.proto_to_pandas_timestamp(conversion_utils.pandas_to_proto_timestamp(times[k + 1]))
            end = conversion_utils.proto_to_pandas_timestamp(conversion_utils.pandas_to_proto_timestamp(times[k + 2]))
            price_e.append(ec.cost(start, end, 1.0))
            carbon_e.append(ec.carbon(start, end, 1.0))
            price_g.append(gc.cost(start, end, 1.0))
            carbon_g.append(gc.carbon(start, end, 1.0))
            max_cost.append(ec.cost(start, end, spec.reward["max_elec"]) + gc.cost(start, end, spec.reward["max_gas"]))
            max_carbon.append(ec.carbon(start, end, spec.reward["max_elec"])
                              + gc.carbon(start, end, spec.reward["max_gas"]))
            reward_dt.append((end - start).total_seconds())

        obs_exo = np.zeros((n_steps + 1, len(spec.exo_fields)))
        for k in range(n_steps + 1):
            ts = times[k]
            vals = {}
            for label, n_feat, interval in (
                    (env_lib.HOD_LABEL, env._num_hod_features, conversion_utils.TimeIntervalEnum.HOUR_OF_DAY),
                    (env_lib.DOW_LABEL, env._num_dow_features, conversion_utils.TimeIntervalEnum.DAY_OF_WEEK)):
                rad = conversion_utils.get_radian_time(ts, interval)
                for name, value in regression_building_utils.expand_time_features(n_feat, rad, label).items():
                    vals[f"{name[0]}_{name[1]}"] = value
            vals[env_lib.COMFORT_MODE_NOW] = float(env.building.is_comfort_mode(ts))
            vals[env_lib.COMFORT_MODE_SOON] = float(env.building.is_comfort_mode(ts + pd.Timedelta(60, unit="minute")))
            vals[env_lib.NUM_OCCUPANTS] = (n_occ_obs[k] - spec.occupancy_norm) / (spec.occupancy_norm + 1)
            vals["t_amb_obs"] = wc.get_current_temp(ts)
            obs_exo[k] = [vals[f] for f in spec.exo_fields]

        return cls(
            t_amb=np.array([wc.get_current_temp(times[k]) for k in range(n_steps)]),
            t_amb_next=np.array([wc.get_current_temp(times[k + 1]) for k in range(n_steps)]),
            h=np.array([wc.get_air_convection_coefficient(times[k]) for k in range(n_steps)]),
            comfort=np.array([float(hvac.is_comfort_mode(times[k])) for k in range(n_steps)]),
            win_now=np.array([schedule.get_temperature_window(times[k]) for k in range(n_steps)], dtype=float),
            win_next=np.array([schedule.get_temperature_window(times[k + 1]) for k in range(n_steps)], dtype=float),
            occ_step=occ_step, occ_rew=occ_rew,
            price_e=np.array(price_e), carbon_e=np.array(carbon_e), price_g=np.array(price_g),
            carbon_g=np.array(carbon_g), max_cost=np.array(max_cost), max_carbon=np.array(max_carbon),
            reward_dt=np.array(reward_dt), obs_exo=obs_exo,
            times=np.array([t.value for t in times[:n_steps + 1]], dtype=np.int64),
        )
