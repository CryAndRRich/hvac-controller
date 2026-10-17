"""Equipment, reward and observation of one building as pure batched functions, on NumPy or PyTorch (ops.Ops).

Mirrors what one env.step of the organisers' simulator computes around its thermal solve
(simulator_flexible_floor_plan.execute_step_sim, the device models, SetpointEnergyCarbonRegretFunction and
Environment._get_observation):

- ``decode``: normalised action -> native setpoints (protobuf float32) -> device clamps and floors.
- ``pre``: temperatures at t -> supply air, VAV flows, hot water, reheat -> heat per zone q, air conductance g and
  supply temperature Ts per VAV, which the thermal ROM takes.
- ``post``: temperatures at t+dt -> return water, boiler tank, the energy the reward bills, reward, observation.

Facts the order of operations encodes (read from the organisers' code, checked against traces in env_check):

- The observed zone temperature is the one at the *start* of the step: VAV.zone_air_temperature is only written in
  update_settings, before the solve.
- Reading the boiler's supply-water sensor moves its tank toward the setpoint by the time elapsed since the last
  action. The reads inside the step see zero elapsed time, so the tank moves once per step, in the observation;
  the reheat of step k uses the tank temperature before that move.
- The reward bills cooling from the zone temperatures and outside temperature *after* the step, so the billed
  cooling is not the cooling the zones received.

``exact=True`` keeps the organisers' branches. ``exact=False`` relaxes the two discrete switches (AHU and hot-water
run commands, given in [0, 1]) to products, for gradients. ``emulate_f32`` rounds wherever the organisers store a
value in a protobuf float: native actions, the reward's inputs and output, and observations.
"""

from __future__ import annotations

import math

import numpy as np

from hvac_controller.d1.ops import Ops
from hvac_controller.d1.spec import SOURCE_KIND, SOURCES, BuildingSpec

TINY = 1e-30  # inside square roots, so their gradient stays finite at zero; never changes a forward value


class Plant:
    def __init__(self, spec: BuildingSpec, ops: Ops, exact: bool = True, emulate_f32: bool = True):
        o = ops
        self.o, self.spec, self.exact, self.f32 = o, spec, exact, emulate_f32
        Z, V, A = spec.n_zones, spec.n_vavs, spec.n_ahus
        self.Z, self.V, self.A = Z, V, A
        self.share = o.asarray(spec.vav_zone_share)  # (Z, V)
        self.ahu_w = o.asarray(spec.ahu_weight)  # (A, V)
        self.vav_ahu = o.index(spec.vav_ahu)
        onehot = np.zeros((V, A))
        onehot[np.arange(V), spec.vav_ahu] = 1.0
        self.onehot = o.asarray(onehot)
        self.n_vav_per_ahu = o.asarray(onehot.sum(axis=0))
        ah, vv, p = spec.ahu, spec.vav, spec.phys
        self.ahu_max, self.max_recirc = o.asarray(ah["max_flow"]), o.asarray(ah["max_recirc"])
        self.p_min, self.fan_eff, self.cop = o.asarray(ah["min_pressure"]), o.asarray(ah["fan_eff"]), o.asarray(ah["cop"])
        self.v_max, self.min_damper = o.asarray(vv["max_flow"]), o.asarray(vv["min_damper"])
        self.max_dis, self.design_p = o.asarray(vv["max_discharge"]), o.asarray(vv["design_pressure"])
        self.reheat_ff = o.asarray(vv["reheat_ff"])
        sm = spec.sim
        self.gain_on = o.asarray(sm["gain_w_per_m2"] * sm["cv_area_m2"] * 1.0 * spec.zone_cells)
        self.gain_off = o.asarray(sm["gain_w_per_m2"] * sm["cv_area_m2"] * sm["standby_fraction"] * spec.zone_cells)
        self.air_cap = p["air_cp"] * p["air_rho"]  # C_air = (c_p rho) flow, as Vav._reheat_effectiveness
        self.water_cap = p["water_cp"] * p["water_rho"]
        self.boiler_cap = p["boiler_water_cp"] * p["boiler_water_rho"]
        self.head_per_bar = p["gravity"] * p["water_rho"] / p["pa_per_bar"]
        hw = spec.hws
        r1 = hw["tank_radius"]
        r2 = r1 + hw["ins_t"]
        self.tank_dynamic = hw["cooling_rate"] > 0.0 and hw["heating_rate"] > 0.0 and hw["has_tank"] > 0.0
        self.diss_den = (math.log(r2 / r1) / hw["ins_k"] + 1.0 / hw["conv"] / r2) if hw["has_tank"] else 1.0
        # Actions.
        cont, disc = spec.cont, spec.disc
        self.c_nmin, self.c_nmax = o.asarray([c[2] for c in cont]), o.asarray([c[3] for c in cont])
        self.c_lo, self.c_hi = o.asarray([c[4] for c in cont]), o.asarray([c[5] for c in cont])
        self.d_nmin, self.d_nmax = o.asarray([c[2] for c in disc]), o.asarray([c[3] for c in disc])
        self.d_lo, self.d_hi = o.asarray([c[4] for c in disc]), o.asarray([c[5] for c in disc])

        def columns(rows, target, n):
            idx = [-1] * n
            for j, row in enumerate(rows):
                if row[0] == target:
                    idx[row[1]] = j
            assert min(idx) >= 0, target
            return o.index(idx)

        self.i_ahu_sp, self.i_ahu_p, self.i_ahu_r = (columns(cont, t, A) for t in ("ahu_sp", "ahu_p", "ahu_r"))
        self.i_w_sp, self.i_w_dp = (columns(cont, t, 1) for t in ("w_sp", "w_dp"))
        self.i_damper, self.i_valve = (columns(cont, t, V) for t in ("damper", "valve"))
        self.i_ahu_on, self.i_w_on = columns(disc, "ahu_on", A), columns(disc, "w_on", 1)
        # Observations: one gather into [sources | exogenous] followed by the per-field normalisation.
        width = {"vav": V, "ahu": A, "one": 1}
        offset, start = {}, 0
        for s in SOURCES:
            offset[s] = start
            start += width[SOURCE_KIND[s]]
        self.n_src = start
        gather, mean, std, plain = [], [], [], []
        for kind, name, i, m, sd in spec.obs:
            if kind == "src":
                gather.append(offset[name] + i)
            else:
                gather.append(self.n_src + i)
            plain.append(kind == "exo")
            mean.append(m)
            std.append(sd)
        self.obs_gather = o.index(gather)
        self.obs_mean, self.obs_std = o.asarray(mean), o.asarray(std)
        self.obs_std_safe = o.asarray([s if s > 0.0 else 1.0 for s in std])
        self.obs_plain = o.asarray(plain) > 0.5
        self.obs_zero = o.asarray([(not p) and s <= 0.0 for p, s in zip(plain, std)]) > 0.5
        self.n_obs = len(gather)

    # ------------------------------------------------------------------------------------------------- helpers
    def _r32(self, x):
        return self.o.round_f32(x) if self.f32 else x

    def _switch(self, on, a, b):
        """a where the device runs, else b: exact branch, or the relaxation on * a + (1 - on) * b."""
        if self.exact:
            return self.o.where(on > 0.5, a, b)
        return on * a + (1.0 - on) * b

    # -------------------------------------------------------------------------------------------------- decode
    def decode(self, a_cont, a_disc) -> dict:
        """Hybrid action (continuous in [-1, 1], discrete in {0, 1}; [0, 1] when relaxed) -> device settings."""
        o = self
        nat_c = (a_cont - o.c_nmin) / (o.c_nmax - o.c_nmin) * (o.c_hi - o.c_lo) + o.c_lo
        nat_d = ((2.0 * a_disc - 1.0) - o.d_nmin) / (o.d_nmax - o.d_nmin) * (o.d_hi - o.d_lo) + o.d_lo
        nat_c, nat_d = self._r32(nat_c), self._r32(nat_d)
        return self.settings(
            ahu_on=nat_d[:, o.i_ahu_on], w_on=nat_d[:, o.i_w_on][:, 0],
            ahu_sp=nat_c[:, o.i_ahu_sp], ahu_p=nat_c[:, o.i_ahu_p], ahu_r=nat_c[:, o.i_ahu_r],
            w_sp=nat_c[:, o.i_w_sp][:, 0], w_dp=nat_c[:, o.i_w_dp][:, 0],
            damper=nat_c[:, o.i_damper], valve=nat_c[:, o.i_valve])

    def settings(self, ahu_on, w_on, ahu_sp, ahu_p, ahu_r, w_sp, w_dp, damper, valve) -> dict:
        """Native commands -> what the devices hold (set_action and the property setters)."""
        x = self.o
        if self.exact:
            ahu_on = x.where(ahu_on == 1.0, 1.0, 0.0)
            w_on = x.where(w_on == 1.0, 1.0, 0.0)
        head = w_dp / self.head_per_bar  # pump stores the head; the pressure getter converts it back
        return {
            "ahu_on": ahu_on, "ahu_sp": ahu_sp, "ahu_p_cmd": ahu_p, "ahu_r": x.clip(ahu_r, 0.0, self.max_recirc),
            "w_on": w_on, "w_sp": w_sp, "w_head": head,
            "damper": x.maximum(x.clip(damper, 0.0, 1.0), self.min_damper), "valve": x.clip(valve, 0.0, 1.0),
        }

    # ----------------------------------------------------------------------------------------------- the step
    def _supply(self, dev, recirc, t_amb):
        x = self.o
        r = dev["ahu_r"]
        mixed = r * recirc + (1.0 - r) * t_amb[:, None]
        sp, on = dev["ahu_sp"], dev["ahu_on"]
        if self.exact:
            return mixed, x.where((on > 0.5) & (mixed > sp), sp, mixed)
        return mixed, mixed - on * x.relu(mixed - sp)

    def pre(self, y, tank, dev: dict, t_amb, comfort, occ_step) -> dict:
        """Everything before the thermal solve. y: (B, V + Z) sensed and zone temperatures at t."""
        x, V = self.o, self.V
        sens = y[:, :V]
        recirc = sens @ self.ahu_w.T
        mixed, supply = self._supply(dev, recirc, t_amb)
        on = dev["ahu_on"]
        p_eff = self._switch(on, x.maximum(dev["ahu_p_cmd"], self.p_min), 0.0)
        p_v = p_eff[:, self.vav_ahu]
        demand = x.maximum(dev["damper"] * self.v_max * x.sqrt(x.maximum(p_v / self.design_p, TINY)), 1e-5)
        demanded = demand @ self.onehot
        ratio = x.minimum(1.0, self.ahu_max / demanded)
        flow = demand * ratio[:, self.vav_ahu]
        ahu_raw = x.minimum(flow @ self.onehot, self.ahu_max)  # AirHandler._air_flow_rate, capped as it accumulates
        ahu_flow = self._switch(on, ahu_raw, 0.0)  # the gated air_flow_rate the fans and sensors see
        ff = dev["valve"] * self.reheat_ff
        ff_sum = x.sum(ff, 1)
        dp = self._switch(dev["w_on"], self.head_per_bar * dev["w_head"], 0.0)
        root = x.where(dp > 0.0, x.sqrt(x.maximum(dp / (1.0 + ff_sum ** 2 * self.spec.hws["header_resistance"]),
                                                  TINY)), 0.0)
        q_water = ff_sum * root / 3600.0
        ff_safe = x.where(ff_sum > 0.0, ff_sum, 1.0)
        water_share = x.where(ff_sum[:, None] > 0.0, q_water[:, None] * (ff / ff_safe[:, None]), 0.0)
        c_air = self.air_cap * flow
        c_water = self.water_cap * water_share
        eff0 = x.minimum(1.0, c_water / c_air)
        ts = supply[:, self.vav_ahu]
        t_w = tank[:, None]
        gap = t_w - ts
        headroom = self.max_dis - ts
        capped = x.minimum(eff0, headroom / x.where(gap > 0.0, gap, 1.0))
        eff = x.where(gap <= 0.0, eff0, x.where(headroom <= 0.0, 0.0, capped))
        reheat = eff * c_air * (t_w - ts)
        gains = x.where(comfort[:, None] > 0.5, self.gain_on, self.gain_off)
        q_zone = reheat @ self.share.T + gains + occ_step * self.spec.sim["occupant_w"]
        g = flow * self.spec.phys["air_rho"] * self.spec.phys["air_cp"]
        return {"sens": sens, "recirc": recirc, "mixed": mixed, "supply": supply, "p_eff": p_eff, "demand": demand,
                "ratio": ratio, "flow": flow, "ahu_raw": ahu_raw, "ahu_flow": ahu_flow, "ff_sum": ff_sum, "dp": dp,
                "q_water": q_water, "eff": eff, "reheat": reheat, "q_zone": q_zone, "g": g, "ts": ts,
                "tank_used": tank}

    def post(self, y_new, ret, dev: dict, aux: dict, t_amb_next, win_now, win_next, occ_rew, price_e, carbon_e,
             price_g, carbon_g, max_cost, max_carbon, reward_dt, obs_exo_next) -> dict:
        """Everything after the thermal solve. y_new: (B, V + Z) at t+dt."""
        x, V = self.o, self.V
        tank = aux["tank_used"]
        # Return water from the loop's energy balance (held when no water flows).
        reheat_sum = x.sum(aux["reheat"], 1)
        w_rate = self.water_cap * aux["q_water"]
        raw_ret = tank - reheat_sum / x.where(w_rate > 0.0, w_rate, 1.0)
        air_t = x.where(reheat_sum >= 0.0, x.amin(aux["ts"], 1), x.amax(aux["ts"], 1))
        low, high = x.minimum(tank, air_t), x.maximum(tank, air_t)
        ret_new = x.where(w_rate > 0.0, x.minimum(high, x.maximum(low, raw_ret)), ret)
        # The boiler tank moves toward its setpoint when the observation reads it.
        sp = dev["w_sp"]
        if self.tank_dynamic:
            step = self.spec.dt / 60.0
            up = x.minimum(tank + self.spec.hws["heating_rate"] * step, sp)
            down = x.maximum(tank - self.spec.hws["cooling_rate"] * step, sp)
            tank_new = x.where(sp > tank, up, x.where(sp < tank, down, sp))
            d_tank = tank_new - tank
        else:
            tank_new, d_tank = sp, 0.0 * sp
        # Billed energy, from the state after the step.
        recirc_n = y_new[:, :V] @ self.ahu_w.T
        mixed_n, supply_n = self._supply(dev, recirc_n, t_amb_next)
        cool = x.sum(aux["ahu_raw"] * self.spec.phys["air_rho"] * self.spec.phys["air_cp"] * (supply_n - mixed_n)
                     / self.cop, 1)
        p_eff, ahu_flow, r = aux["p_eff"], aux["ahu_flow"], dev["ahu_r"]
        blower = x.sum(ahu_flow * p_eff / self.fan_eff, 1) + x.sum(ahu_flow * (1.0 - r) * p_eff / self.fan_eff, 1)
        head = self._switch(dev["w_on"], dev["w_head"], 0.0)
        pump = aux["q_water"] * self.spec.phys["water_rho"] * self.spec.phys["gravity"] * head / self.spec.hws["pump_eff"]
        sw = x.where(sp > ret_new, sp, ret_new)
        flow_heat = self.boiler_cap * aux["q_water"] * (sw - ret_new)
        hw = self.spec.hws
        diss = x.where(sw < t_amb_next, 0.0, hw["tank_length"] * 2.0 * np.pi * (sw - t_amb_next) / self.diss_den)
        tank_heat = self.boiler_cap * hw["water_capacity"] * d_tank / self.spec.dt
        gas = x.relu(flow_heat + diss + tank_heat)
        # Reward (inputs and output pass through protobuf floats).
        rw = self.spec.reward
        zone_n = self._r32(y_new[:, V:])
        heat_sp, cool_sp = self._r32(win_next[:, 0:1]), self._r32(win_next[:, 1:2])
        occ = self._r32(occ_rew)
        blower32, cool32, gas32, pump32 = (self._r32(v) for v in (blower, cool, gas, pump))
        dev_k = x.maximum(x.maximum(0.0, zone_n - cool_sp), heat_sp - zone_n)
        ref = rw["dev_ref"]
        shortfall = x.where(dev_k <= ref, dev_k / ref, 1.0 + ((dev_k - ref) / ref) ** 2.0)
        hours = reward_dt / 3600.0
        actual = x.sum(rw["max_pph"] * (1.0 - shortfall) * occ * reward_dt[:, None] / 3600.0, 1)
        total_occ = x.sum(occ, 1)
        max_p = rw["max_pph"] * total_occ * hours
        min_p = rw["min_pph"] * total_occ * hours
        regret = x.where(total_occ > 0.0, (actual - min_p) / x.where(total_occ > 0.0, max_p - min_p, 1.0) - 1.0, 0.0)
        elec = blower32 + x.abs(cool32) + pump32
        cost = price_e * x.abs(elec) + price_g * gas32
        carbon = carbon_e * x.abs(elec) + carbon_g * gas32
        norm_cost, norm_carbon = cost / max_cost, carbon / max_carbon
        raw_reward = regret * rw["w_prod"] - norm_cost * rw["w_cost"] - norm_carbon * rw["w_carbon"]
        reward = self._r32(raw_reward / (rw["w_prod"] + rw["w_cost"] + rw["w_carbon"]))
        src = {
            "damper": dev["damper"], "valve": dev["valve"], "vav_max_flow": self.v_max + 0.0 * aux["sens"],
            "sens_prev": aux["sens"], "vav_heat_sp": win_now[:, 0:1] + 0.0 * aux["sens"],
            "vav_cool_sp": win_now[:, 1:2] + 0.0 * aux["sens"], "ahu_count": self.n_vav_per_ahu + 0.0 * p_eff,
            "ahu_p": p_eff, "ahu_flow": ahu_flow, "ahu_sp": dev["ahu_sp"], "ahu_on": dev["ahu_on"],
            "ahu_speed": aux["ahu_raw"] / self.ahu_max, "ahu_oa_flow": (1.0 - r) * aux["ahu_raw"], "ahu_r": r,
            "w_sp": sp[:, None], "tank": tank_new[:, None], "w_count": float(V) + 0.0 * sp[:, None],
            "w_on": dev["w_on"][:, None], "w_dp": aux["dp"][:, None],
        }
        return {"ret": ret_new, "tank": tank_new, "d_tank": d_tank, "cool": cool, "blower": blower, "pump": pump,
                "gas": gas, "elec": elec, "regret": regret, "norm_cost": norm_cost, "norm_carbon": norm_carbon,
                "productivity": actual, "total_occ": total_occ, "reward": reward,
                "obs": self.observe(src, obs_exo_next)}

    def observe(self, src: dict, obs_exo):
        """Raw sources (B, width) and exogenous columns (B, n_exo) -> the observation vector the policy sees."""
        x = self.o
        raw = x.concatenate([src[s] for s in SOURCES] + [obs_exo], 1)[:, self.obs_gather]
        if self.f32:
            raw = x.round_f32(raw)
        norm = (raw - self.obs_mean) / self.obs_std_safe
        out = x.where(self.obs_plain, raw, x.where(self.obs_zero, 0.0, norm))
        return self._r32(out)

    def reset_obs(self, batch: int, obs_exo0):
        x = self.o
        src = {s: x.asarray(np.repeat(self.spec.reset_raw[s][None, :], batch, axis=0)) for s in SOURCES}
        return self.observe(src, obs_exo0)
