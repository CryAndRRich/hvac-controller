"""Ground truth for step C: run the organisers' environment and record every quantity plant.py recomputes.

``Tracer`` wraps, on one freshly reset environment:

- the solver (core.CoreRecorder, log only): heat per zone, air conductance and supply temperature per VAV;
- execute_step_sim: the state before the step (zone temperatures, boiler tank, return water) and the devices after
  it (native settings, flows, hot water);
- the occupancy object: what it returned in the step and in the reward;
- the reward function: the RewardInfo it was given and the RewardResponse it returned.

``ExcitePolicy`` is a structured random policy: actions held for a few steps, a share of them at the bounds, and
AHU / hot-water run commands switched off in blocks of hours, so every branch of the equipment equations is visited.
"""

from __future__ import annotations

import numpy as np

from hvac_controller.d1.core import CoreRecorder
from hvac_controller.d1.spec import NON_ZONES, _ahus
from smart_buildings import evaluate


class ExcitePolicy:
    def __init__(self, action_spec, seed: int = 0, p_switch: float = 1 / 8,
                 p_bound: float = 0.3, mean_off_steps: int = 24, mean_on_steps: int = 72):
        self.rng = np.random.default_rng(seed)
        self.n_c = int(action_spec["continuous_action"].shape[0])
        self.n_d = int(action_spec["discrete_action"].shape[0])
        self.p_switch, self.p_bound = p_switch, p_bound
        self.p_off, self.p_on = 1.0 / mean_on_steps, 1.0 / mean_off_steps
        self.c = self._draw(self.n_c)
        self.d = np.ones(self.n_d, dtype=np.int32)

    def _draw(self, n):
        x = self.rng.uniform(-1.0, 1.0, n)
        bound = self.rng.random(n) < self.p_bound
        return np.where(bound, np.sign(x), x)

    def get_action(self, obs):
        switch = self.rng.random(self.n_c) < self.p_switch
        self.c = np.where(switch, self._draw(self.n_c), self.c)
        u = self.rng.random(self.n_d)
        self.d = np.where(self.d == 1, (u >= self.p_off).astype(np.int32), (u < self.p_on).astype(np.int32))
        return {"continuous_action": self.c.astype(np.float32), "discrete_action": self.d.astype(np.int32)}


def make_policy(name: str, env, meta, seed: int = 0):
    if name == "excite":
        return ExcitePolicy(env.action_spec(), seed=seed)
    return evaluate.initialize_policy(evaluate.POLICY_MAP[name], meta, env.action_spec())


class Tracer:
    def __init__(self, env):
        self.env = env
        sim = env.building.simulator
        self.sim = sim
        hvac = sim._hvac
        self.hvac = hvac
        self.zone_names = [z for z in sim.building.room_dict if z not in NON_ZONES and sim.building.room_dict[z]]
        self.vavs = [hvac.vavs[v] for v in sim._coupled_vavs]
        self.ahus = _ahus(hvac)
        self.hws = hvac.hot_water_system
        self.boiler, self.pump = self.hws._heat_source, self.hws._pump
        self.core = CoreRecorder(env, verify=False)
        self.rows: list[dict] = []
        self._row: dict = {}
        self._phase = None
        self._occ: dict = {}
        # execute_step_sim: state before and devices after.
        self._orig_step = sim.execute_step_sim
        sim.execute_step_sim = self._step
        # occupancy: values returned per phase.
        occ = env.building._occupancy
        self._orig_occ = occ.average_zone_occupancy
        occ.average_zone_occupancy = self._occupancy
        # reward: info and response.
        self._orig_reward_info = sim.reward_info
        sim.reward_info = self._reward_info
        rf = env.reward_function
        self._orig_compute = rf.compute_reward
        rf.compute_reward = self._compute_reward

    def detach(self):
        self.core.detach()
        self.sim.execute_step_sim = self._orig_step
        self.env.building._occupancy.average_zone_occupancy = self._orig_occ
        self.sim.reward_info = self._orig_reward_info
        self.env.reward_function.compute_reward = self._orig_compute

    # ---- wrappers ---------------------------------------------------------------------------------------------
    def _zone_temps(self):
        t = self.sim.building.get_zone_average_temps()
        return np.array([t[z] for z in self.zone_names], dtype=np.float64)

    def _occupancy(self, zone_id, start, end):
        value = self._orig_occ(zone_id, start, end)
        if self._phase is not None:
            self._occ.setdefault(self._phase, {})[zone_id] = value
        return value

    def _step(self, *args, **kwargs):
        r = self._row
        b = self.boiler
        r["zone_pre"] = self._zone_temps()
        r["tank_pre"] = b._current_temperature
        r["ret_pre"] = b._return_water_temperature_sensor
        # Native settings as the devices hold them after request_action.
        r["nat_ahu_on"] = np.array([float(int(a.run_command)) for a in self.ahus])
        r["nat_ahu_sp"] = np.array([a._supply_air_temperature_setpoint for a in self.ahus])
        r["nat_ahu_p"] = np.array([a._fan_static_pressure for a in self.ahus])
        r["nat_ahu_r"] = np.array([a.recirculation for a in self.ahus])
        r["nat_w_on"] = float(int(self.hws.run_command))
        r["nat_w_sp"] = b._supply_water_temperature_setpoint
        r["nat_w_head"] = self.pump._water_pump_differential_head
        r["nat_damper_cmd"] = np.array([np.nan if v.damper_command is None else v.damper_command for v in self.vavs])
        r["nat_valve_cmd"] = np.array([np.nan if v.reheat_valve_command is None else v.reheat_valve_command
                                       for v in self.vavs])
        self._occ = {}
        self._phase = "step"
        out = self._orig_step(*args, **kwargs)
        self._phase = None
        r["occ_step"] = np.array([self._occ.get("step", {}).get(z, np.nan) for z in self.zone_names])
        r["sens_pre"] = np.array([v.zone_air_temperature for v in self.vavs])
        r["damper"] = np.array([v.damper_setting for v in self.vavs])
        r["valve"] = np.array([v.reheat_valve_setting for v in self.vavs])
        r["flow"] = np.array([v.flow_rate_demand for v in self.vavs])
        r["ratio"] = np.array([v.supply_air_flow_ratio for v in self.vavs])
        r["ahu_raw"] = np.array([a._air_flow_rate for a in self.ahus])
        r["ahu_flow"] = np.array([a.air_flow_rate for a in self.ahus])
        r["ahu_p_eff"] = np.array([a.supply_air_static_pressure_setpoint for a in self.ahus])
        r["ff_sum"] = self.hws.flow_factor_sum
        r["q_water"] = self.hws.total_flow_rate
        r["ret"] = b._return_water_temperature_sensor
        r["zone_post"] = self._zone_temps()
        return out

    def _reward_info(self, occupancy_function):
        self._phase = "reward"
        try:
            return self._orig_reward_info(occupancy_function)
        finally:
            self._phase = None

    def _compute_reward(self, info):
        response = self._orig_compute(info)
        r = self._row
        zi = info.zone_reward_infos
        r["rw_zone_temp"] = np.array([zi[z].zone_air_temperature for z in self.zone_names])
        r["rw_heat_sp"] = np.array([zi[z].heating_setpoint_temperature for z in self.zone_names])
        r["rw_cool_sp"] = np.array([zi[z].cooling_setpoint_temperature for z in self.zone_names])
        r["rw_occ"] = np.array([zi[z].average_occupancy for z in self.zone_names])
        r["occ_rew"] = np.array([self._occ.get("reward", {}).get(z, np.nan) for z in self.zone_names])
        ah = list(info.air_handler_reward_infos.values())
        r["rw_blower"] = sum(a.blower_electrical_energy_rate for a in ah)
        r["rw_cool"] = sum(a.air_conditioning_electrical_energy_rate for a in ah)
        bo = list(info.boiler_reward_infos.values())
        r["rw_gas"] = sum(x.natural_gas_heating_energy_rate for x in bo)
        r["rw_pump"] = sum(x.pump_electrical_energy_rate for x in bo)
        for name in ("agent_reward_value", "productivity_reward", "electricity_energy_cost", "natural_gas_energy_cost",
                     "carbon_emitted", "total_occupancy", "normalized_productivity_regret", "normalized_energy_cost",
                     "normalized_carbon_emission"):
            r[f"rr_{name}"] = getattr(response, name)
        return response

    # ---- rollout ----------------------------------------------------------------------------------------------
    def run(self, policy, obs0, n_steps: int) -> dict[str, np.ndarray]:
        """Steps the environment n_steps times; obs0 is what env.reset() returned."""
        obs = np.asarray(obs0)
        out = {"obs0": obs.copy()}
        for _ in range(n_steps):
            action = evaluate.validate_action(evaluate.get_policy_action(policy, obs), self.env.action_spec())
            self._row = {"a_cont": action["continuous_action"].copy(),
                         "a_disc": action["discrete_action"].astype(np.float64)}
            obs, reward, terminated, truncated, _ = self.env.step(action)
            obs = np.asarray(obs)
            r = self._row
            r["obs"] = obs.copy()
            r["reward"] = float(reward)
            r["tank"] = self.boiler._current_temperature
            r["d_tank"] = self.boiler._step_tank_temperature_change
            self.rows.append(r)
            if terminated or truncated:
                break
        core = self.core.arrays()
        for k in self.rows[0]:
            out[k] = np.stack([np.asarray(row[k], dtype=np.float64) for row in self.rows])
        for k in ("q_zone", "g", "ts", "t_amb", "h", "sensed_env"):
            out[f"core_{k}"] = np.asarray(core[k], dtype=np.float64)
        return out
