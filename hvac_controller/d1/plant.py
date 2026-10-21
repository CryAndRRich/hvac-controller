"""Equipment, reward and observation of a building as pure batched functions, on NumPy or PyTorch (ops.Ops).

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

Several buildings in one batch (``specs`` of different buildings, one per scenario, rows grouped by scenario): every
device dimension is padded to the largest building's, and each row's constants, index maps and action and
observation layouts are its scenario's. A row's observation holds its building's fields first, in order, then zeros;
its action is read from its building's columns. Padding is inert: a padded VAV has no flow capacity and no reheat
and belongs to no AHU (it reads its pressure and supply from AHU 0, a real one), a padded AHU serves no VAV, a
padded zone has no floor area and no occupants. With one building nothing is padded and every operation is the
one-building one.
"""

from __future__ import annotations

import math
import pickle

import numpy as np

from hvac_controller.d1.ops import Ops
from hvac_controller.d1.spec import SOURCE_KIND, SOURCES, BuildingSpec

TINY = 1e-30  # inside square roots, so their gradient stays finite at zero; never changes a forward value
# What differs between buildings (per scenario in a padded batch); everything else must be equal.
BUILDING_FIELDS = ("zone_names", "vav_ids", "ahu_ids", "zone_cells", "vav_zone_share", "vav_ahu", "ahu_weight", "ahu",
                   "vav", "cont", "disc", "obs", "reset_raw", "tank0", "ret0", "field_names")
SHARED_FIELDS = ("dt", "hws", "sim", "phys", "exo_fields", "occupancy_norm")
SHARED_REWARD = ("w_prod", "w_cost", "w_carbon", "max_pph", "min_pph", "dev_ref")  # max_elec and max_gas scale with
# the AHU count, but only through the exogenous max_cost and max_carbon
# Padding values of the per-device constants: inert, and no division by zero.
PAD_AHU = {"max_flow": 1.0, "max_recirc": 0.7, "min_pressure": 0.0, "fan_eff": 1.0, "cop": 1.0}
PAD_VAV = {"max_flow": 0.0, "min_damper": 0.1, "max_discharge": 400.0, "design_pressure": 1.0, "reheat_ff": 0.0}


def building_key(spec: BuildingSpec) -> bytes:
    return pickle.dumps([getattr(spec, f) for f in BUILDING_FIELDS + SHARED_FIELDS + ("reward",)])


def _pad(v, shape, fill=0.0) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    out = np.full(shape, fill, dtype=np.float64)
    out[tuple(slice(0, n) for n in v.shape)] = v
    return out


class Plant:
    def __init__(self, spec: BuildingSpec | list[BuildingSpec], ops: Ops, exact: bool = True,
                 emulate_f32: bool = True, n_rows: int | None = None):
        """``spec``: one building, or one spec per scenario (rows grouped: row i is scenario i // (n_rows / S))."""
        o = ops
        specs = spec if isinstance(spec, list) else [spec]
        sp0 = specs[0]
        self.o, self.spec, self.exact, self.f32 = o, sp0, exact, emulate_f32
        self.grouped = len({building_key(s) for s in specs}) > 1
        for s in specs:
            for f in SHARED_FIELDS:
                if pickle.dumps(getattr(s, f)) != pickle.dumps(getattr(sp0, f)):
                    raise ValueError(f"{s.label} and {sp0.label} differ in {f}: one batch needs it equal")
            if any(s.reward[k] != sp0.reward[k] for k in SHARED_REWARD):
                raise ValueError(f"{s.label} and {sp0.label} differ in the reward's weights or productivity constants")
        S = len(specs)
        Z, V, A = (max(getattr(s, n) for s in specs) for n in ("n_zones", "n_vavs", "n_ahus"))
        self.S, self.Z, self.V, self.A = S, Z, V, A
        if self.grouped:
            if n_rows is None or n_rows % S:
                raise ValueError(f"{n_rows} rows for {S} scenarios")
            row_s = np.arange(n_rows) // (n_rows // S)
        n_cont, n_disc = max(len(s.cont) for s in specs), max(len(s.disc) for s in specs)
        n_obs = max(len(s.obs) for s in specs)

        def per_row(make, shape, fill=0.0):
            """A constant per device: (shape) for one building, (rows, shape) for several."""
            if not self.grouped:
                return o.asarray(make(sp0))
            return o.asarray(np.stack([_pad(make(s), shape, fill) for s in specs])[row_s])

        def per_scenario(make, shape):
            """A small matrix: as is for one building, (S, shape) for several (applied by _mm)."""
            if not self.grouped:
                return o.asarray(make(sp0))
            return o.asarray(np.stack([_pad(make(s), shape) for s in specs]))

        def index(make, n):
            """Columns or devices to take: (n,) for one building, (rows, n) padded with 0 for several."""
            if not self.grouped:
                return o.index(make(sp0))
            return o.index(np.stack([_pad(make(s), (n,)) for s in specs]).astype(np.int64)[row_s])

        def onehot(s, padded_to_ahu0: bool):
            m = np.zeros((V if self.grouped else s.n_vavs, A if self.grouped else s.n_ahus))
            m[np.arange(s.n_vavs), s.vav_ahu] = 1.0
            if padded_to_ahu0:
                m[s.n_vavs:, 0] = 1.0
            return m

        self.share_t = per_scenario(lambda s: s.vav_zone_share.T, (V, Z))  # (V, Z): reheat split
        self.ahu_w_t = per_scenario(lambda s: s.ahu_weight.T, (V, A))  # (V, A): recirculation temperature
        self.to_ahu = per_scenario(lambda s: onehot(s, False), (V, A))  # VAV -> its AHU, summed
        self.vav_ahu = o.index(sp0.vav_ahu)
        self.from_ahu = per_scenario(lambda s: onehot(s, True).T, (A, V)) if self.grouped else None
        self.n_vav_per_ahu = per_row(lambda s: onehot(s, False).sum(axis=0), (A,))
        self.n_vav = per_row(lambda s: [float(s.n_vavs)], (1,))
        ah = {k: per_row(lambda s, k=k: s.ahu[k], (A,), PAD_AHU[k]) for k in PAD_AHU}
        vv = {k: per_row(lambda s, k=k: s.vav[k], (V,), PAD_VAV[k]) for k in PAD_VAV}
        self.ahu_max, self.max_recirc, self.p_min = ah["max_flow"], ah["max_recirc"], ah["min_pressure"]
        self.fan_eff, self.cop = ah["fan_eff"], ah["cop"]
        self.v_max, self.min_damper = vv["max_flow"], vv["min_damper"]
        self.max_dis, self.design_p, self.reheat_ff = vv["max_discharge"], vv["design_pressure"], vv["reheat_ff"]
        sm, p = sp0.sim, sp0.phys
        self.gain_on = per_row(lambda s: sm["gain_w_per_m2"] * sm["cv_area_m2"] * 1.0 * s.zone_cells, (Z,))
        self.gain_off = per_row(lambda s: sm["gain_w_per_m2"] * sm["cv_area_m2"] * sm["standby_fraction"] * s.zone_cells,
                                (Z,))
        self.air_cap = p["air_cp"] * p["air_rho"]  # C_air = (c_p rho) flow, as Vav._reheat_effectiveness
        self.water_cap = p["water_cp"] * p["water_rho"]
        self.boiler_cap = p["boiler_water_cp"] * p["boiler_water_rho"]
        self.head_per_bar = p["gravity"] * p["water_rho"] / p["pa_per_bar"]
        hw = sp0.hws
        r1 = hw["tank_radius"]
        r2 = r1 + hw["ins_t"]
        self.tank_dynamic = hw["cooling_rate"] > 0.0 and hw["heating_rate"] > 0.0 and hw["has_tank"] > 0.0
        self.diss_den = (math.log(r2 / r1) / hw["ins_k"] + 1.0 / hw["conv"] / r2) if hw["has_tank"] else 1.0
        # Actions (padded columns: an inert unit range).
        self.c_nmin = per_row(lambda s: [c[2] for c in s.cont], (n_cont,), -1.0)
        self.c_nmax = per_row(lambda s: [c[3] for c in s.cont], (n_cont,), 1.0)
        self.c_lo = per_row(lambda s: [c[4] for c in s.cont], (n_cont,), 0.0)
        self.c_hi = per_row(lambda s: [c[5] for c in s.cont], (n_cont,), 1.0)
        self.d_nmin = per_row(lambda s: [c[2] for c in s.disc], (n_disc,), -1.0)
        self.d_nmax = per_row(lambda s: [c[3] for c in s.disc], (n_disc,), 1.0)
        self.d_lo = per_row(lambda s: [c[4] for c in s.disc], (n_disc,), 0.0)
        self.d_hi = per_row(lambda s: [c[5] for c in s.disc], (n_disc,), 1.0)
        self.n_cont, self.n_disc = n_cont, n_disc

        def columns(rows, target, n):
            idx = [-1] * n
            for j, row in enumerate(rows):
                if row[0] == target:
                    idx[row[1]] = j
            assert min(idx) >= 0, target
            return idx

        self.i_ahu_sp, self.i_ahu_p, self.i_ahu_r = (index(lambda s, t=t: columns(s.cont, t, s.n_ahus), A)
                                                     for t in ("ahu_sp", "ahu_p", "ahu_r"))
        self.i_w_sp, self.i_w_dp = (index(lambda s, t=t: columns(s.cont, t, 1), 1) for t in ("w_sp", "w_dp"))
        self.i_damper, self.i_valve = (index(lambda s, t=t: columns(s.cont, t, s.n_vavs), V) for t in ("damper", "valve"))
        self.i_ahu_on = index(lambda s: columns(s.disc, "ahu_on", s.n_ahus), A)
        self.i_w_on = index(lambda s: columns(s.disc, "w_on", 1), 1)
        # Observations: one gather into [sources | exogenous] followed by the per-field normalisation.
        width = {"vav": V, "ahu": A, "one": 1}
        offset, start = {}, 0
        for s in SOURCES:
            offset[s] = start
            start += width[SOURCE_KIND[s]]
        self.n_src = start

        def obs_layout(s):
            gather, mean, std, plain = [], [], [], []
            for kind, name, i, m, sd in s.obs:
                gather.append(offset[name] + i if kind == "src" else self.n_src + i)
                plain.append(float(kind == "exo"))
                mean.append(m)
                std.append(sd)
            zero = [float((not p) and sd <= 0.0) for p, sd in zip(plain, std)]
            return gather, mean, std, plain, zero

        layouts = [obs_layout(s) for s in specs]
        if not self.grouped:
            gather, mean, std, plain, zero = layouts[0]
            self.obs_gather = o.index(gather)
            self.obs_mean = o.asarray(mean)
            self.obs_std_safe = o.asarray([s if s > 0.0 else 1.0 for s in std])
            self.obs_plain, self.obs_zero = o.asarray(plain) > 0.5, o.asarray(zero) > 0.5
        else:  # padded fields: zero (not plain, "zero variance")
            stack = (lambda j, fill: np.stack([_pad(lay[j], (n_obs,), fill) for lay in layouts])[row_s])
            self.obs_gather = o.index(stack(0, 0.0).astype(np.int64))
            self.obs_mean, std = o.asarray(stack(1, 0.0)), stack(2, 1.0)
            self.obs_std_safe = o.asarray(np.where(std > 0.0, std, 1.0))
            self.obs_plain, self.obs_zero = o.asarray(stack(3, 0.0)) > 0.5, o.asarray(stack(4, 1.0)) > 0.5
        self.n_obs = n_obs
        self.reset_raw = {k: per_row(lambda s, k=k: s.reset_raw[k], (width[SOURCE_KIND[k]],)) for k in SOURCES}
        self.max_g = np.stack([_pad(self._max_air_conductance(s), (V,)) for s in specs])  # (S, V)

    @staticmethod
    def _max_air_conductance(sp: BuildingSpec) -> np.ndarray:
        """Upper bound of g = flow * rho * c_p per VAV over every valid action: full damper, the highest static
        pressure the action range allows (1 % over it: the organisers reject actions beyond a 1e-5 tolerance), and
        no AHU capacity cap."""
        p_max = 1.01 * max(max(row[5] for row in sp.cont if row[0] == "ahu_p"), float(np.max(sp.ahu["min_pressure"])))
        flow = sp.vav["max_flow"] * np.sqrt(p_max / sp.vav["design_pressure"])
        return sp.phys["air_rho"] * sp.phys["air_cp"] * flow

    def max_air_conductance(self) -> np.ndarray:
        """(V,) for one building, (S, V) per scenario when padded."""
        return self.max_g if self.grouped else self.max_g[0]

    # ------------------------------------------------------------------------------------------------- helpers
    def _r32(self, x):
        return self.o.round_f32(x) if self.f32 else x

    def _switch(self, on, a, b):
        """a where the device runs, else b: exact branch, or the relaxation on * a + (1 - on) * b."""
        if self.exact:
            return self.o.where(on > 0.5, a, b)
        return on * a + (1.0 - on) * b

    def _mm(self, x, m):
        """x (B, n) times a small matrix: (n, k), or per scenario (S, n, k) over the grouped rows."""
        if not self.grouped:
            return x @ m
        return (x.reshape(self.S, -1, x.shape[1]) @ m).reshape(x.shape[0], -1)

    def _take(self, x, idx):
        """Columns of x: idx (n,) for one building, (B, n) per row."""
        return x[:, idx] if not self.grouped else self.o.take(x, idx)

    def _at_vav(self, x):
        """A per-AHU quantity at each VAV's AHU."""
        return x[:, self.vav_ahu] if not self.grouped else self._mm(x, self.from_ahu)

    # -------------------------------------------------------------------------------------------------- decode
    def decode(self, a_cont, a_disc) -> dict:
        """Hybrid action (continuous in [-1, 1], discrete in {0, 1}; [0, 1] when relaxed) -> device settings."""
        o, t = self, self._take
        nat_c = (a_cont - o.c_nmin) / (o.c_nmax - o.c_nmin) * (o.c_hi - o.c_lo) + o.c_lo
        nat_d = ((2.0 * a_disc - 1.0) - o.d_nmin) / (o.d_nmax - o.d_nmin) * (o.d_hi - o.d_lo) + o.d_lo
        nat_c, nat_d = self._r32(nat_c), self._r32(nat_d)
        return self.settings(
            ahu_on=t(nat_d, o.i_ahu_on), w_on=t(nat_d, o.i_w_on)[:, 0],
            ahu_sp=t(nat_c, o.i_ahu_sp), ahu_p=t(nat_c, o.i_ahu_p), ahu_r=t(nat_c, o.i_ahu_r),
            w_sp=t(nat_c, o.i_w_sp)[:, 0], w_dp=t(nat_c, o.i_w_dp)[:, 0],
            damper=t(nat_c, o.i_damper), valve=t(nat_c, o.i_valve))

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
        recirc = self._mm(sens, self.ahu_w_t)
        _, supply = self._supply(dev, recirc, t_amb)
        on = dev["ahu_on"]
        p_eff = self._switch(on, x.maximum(dev["ahu_p_cmd"], self.p_min), 0.0)
        p_v = self._at_vav(p_eff)
        demand = x.maximum(dev["damper"] * self.v_max * x.sqrt(x.maximum(p_v / self.design_p, TINY)), 1e-5)
        demanded = self._mm(demand, self.to_ahu)
        ratio = x.minimum(1.0, self.ahu_max / x.where(demanded > 0.0, demanded, 1.0))  # 0 only on a padded AHU
        flow = demand * self._at_vav(ratio)
        ahu_raw = x.minimum(self._mm(flow, self.to_ahu), self.ahu_max)  # AirHandler._air_flow_rate, capped as it accumulates
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
        ts = self._at_vav(supply)
        t_w = tank[:, None]
        gap = t_w - ts
        headroom = self.max_dis - ts
        capped = x.minimum(eff0, headroom / x.where(gap > 0.0, gap, 1.0))
        eff = x.where(gap <= 0.0, eff0, x.where(headroom <= 0.0, 0.0, capped))
        reheat = eff * c_air * (t_w - ts)
        gains = x.where(comfort[:, None] > 0.5, self.gain_on, self.gain_off)
        q_zone = self._mm(reheat, self.share_t) + gains + occ_step * self.spec.sim["occupant_w"]
        g = flow * self.spec.phys["air_rho"] * self.spec.phys["air_cp"]
        return {"sens": sens, "p_eff": p_eff, "ratio": ratio, "flow": flow, "ahu_raw": ahu_raw, "ahu_flow": ahu_flow,
                "ff_sum": ff_sum, "dp": dp, "q_water": q_water, "reheat": reheat, "q_zone": q_zone, "g": g, "ts": ts,
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
        recirc_n = self._mm(y_new[:, :V], self.ahu_w_t)
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
            "w_sp": sp[:, None], "tank": tank_new[:, None], "w_count": self.n_vav + 0.0 * sp[:, None],
            "w_on": dev["w_on"][:, None], "w_dp": aux["dp"][:, None],
        }
        return {"ret": ret_new, "tank": tank_new, "d_tank": d_tank, "cool": cool, "blower": blower, "pump": pump,
                "gas": gas, "elec": elec, "regret": regret, "norm_cost": norm_cost, "norm_carbon": norm_carbon,
                "reward": reward, "obs": self.observe(src, obs_exo_next)}

    def observe(self, src: dict, obs_exo):
        """Raw sources (B, width) and exogenous columns (B, n_exo) -> the observation vector the policy sees."""
        x = self.o
        raw = self._take(x.concatenate([src[s] for s in SOURCES] + [obs_exo], 1), self.obs_gather)
        if self.f32:
            raw = x.round_f32(raw)
        norm = (raw - self.obs_mean) / self.obs_std_safe
        out = x.where(self.obs_plain, raw, x.where(self.obs_zero, 0.0, norm))
        return self._r32(out)

    def reset_obs(self, batch: int, obs_exo0):
        x = self.o
        if self.grouped:
            src = {s: self.reset_raw[s] + 0.0 for s in SOURCES}
        else:
            src = {s: x.asarray(np.repeat(self.spec.reset_raw[s][None, :], batch, axis=0)) for s in SOURCES}
        return self.observe(src, obs_exo0)
