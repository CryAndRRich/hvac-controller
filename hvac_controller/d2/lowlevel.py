"""Direction 2: the low-level controller the planner drives, batched, in NumPy.

Every row (one candidate plan inside a rollout, or the live building) runs K10's per-VAV PI law towards a zone
target. Six supervisory decisions per row, each in [0, 1], set everything else:

- ``run``: every AHU and the hot-water pump on when > 0.5; off otherwise (an AHU that is off moves no air, so the
  zones then float and the PI integrals are reset);
- ``target``: the zones' PI target, a position inside the comfort band. Only the comfort band matters for the score:
  occupants are present only on working days between 06:00 and 19:00, when the comfort band applies;
- ``sat``: the AHUs' supply-air temperature setpoint across its native range. The AHU only cools, so a high setpoint
  passes the mixed air through and a low one buys cooling;
- ``recirc``: the share of return air, 0 to the 0.7 cap;
- ``pressure``: the AHUs' static pressure from the 200 Pa floor to the top of its range;
- ``water``: the boiler's supply-water setpoint across its native range; the pump runs at full differential
  pressure whenever it is on.

With ``per_ahu`` the first five are given per AHU, as ``[run (A) | target (A) | sat (A) | recirc (A) | pressure (A) |
water]``: each AHU runs, aims its zones and sets its air on its own, and the pump runs when any AHU does. Otherwise
one vector of six drives every AHU alike.

The PI law is K10's (hvac_controller/baseline_k10.py, ZonePIPolicy._command_zones): u = clip(Kp e + Ki I, -1, 1)
with conditional integration, damper = min + (1 - min) |u|, reheat valve = max(0, -u).

With ``heat_seq`` the heating side is sequenced instead, as a "dual maximum" VAV is: the first half of the heating
demand opens the reheat valve at minimum airflow, and only the second half opens the damper. Every extra m3/s of
supply air carries at least 30 % outside air that the reheat must also warm, so K10's law, which opens the damper
with the heating demand, buys heat with ventilation load.

With ``deadband`` (kelvin) each zone runs two one-sided PI loops instead of one: heating towards the band floor plus
``deadband``, cooling (damper) towards the planner's target, and between the two the zone floats at no cost. One
target makes a zone that is a little under it call for reheat while its AHU is cooling the air: cooling paid for
twice. Heating is sequenced as with ``heat_seq``. The integrals are then (B, 2V): cooling, then heating.

NumPy rather than torch: with a few dozen rows, a step is a few hundred tiny array operations, and NumPy's per-call
overhead is a fraction of torch's (2026-10-24: the replica step at 48 rows takes 0.47 ms in float32 NumPy against
0.89 ms in torch).
"""

from __future__ import annotations

import numpy as np

from hvac_controller.d1.policy import Layout

DEC = ("run", "target", "sat", "recirc", "pressure", "water")
N_DEC = len(DEC)


class LowLevel:
    def __init__(self, meta: dict, kp: float = 0.1, ki: float = 0.005, min_damper: float = 0.1,
                 max_recirc: float = 0.7, p_floor: float = 200.0, dtype=np.float32, per_ahu: bool = False,
                 heat_seq: bool = False, deadband: float = 0.0):
        lay = Layout(meta)
        A, V = lay.A, lay.V
        self.per_ahu, self.heat_seq, self.deadband = per_ahu, heat_seq, deadband
        self.n_int = 2 * lay.V if deadband > 0.0 else lay.V  # width of the PI integral state
        self.n_dec = 5 * A + 1 if per_ahu else N_DEC
        # Columns of each decision in a decision vector.
        if per_ahu:
            self.cols = {name: list(range(i * A, (i + 1) * A)) for i, name in enumerate(DEC[:5])}
            self.cols["water"] = [5 * A]
        else:
            self.cols = {name: [i] for i, name in enumerate(DEC)}
        self.vav_ahu = np.asarray(lay.vav_ahu, dtype=np.int64)
        self.lay, self.A, self.V, self.dtype = lay, A, V, dtype
        self.kp, self.ki, self.min_damper = kp, ki, min_damper
        self.max_recirc, self.p_floor = max_recirc, p_floor
        pad = lay.padded(V, A, lay.n_cont, lay.n_disc)

        def t(v):
            return np.asarray(v, dtype=dtype)

        self.i_temp = np.asarray(lay.i_temp, dtype=np.int64)
        self.t_mean, self.t_std = t(lay.c_temp[:, 0]), t(lay.c_temp[:, 1])
        self.inv_cont, self.inv_disc = pad["inv_cont"], pad["inv_disc"]
        self.lo, self.hi = t(pad["native_lo"]), t(pad["native_hi"])
        self.nlo, self.nhi = t(pad["norm_lo"]), t(pad["norm_hi"])
        self.band = tuple(float(v) for v in meta["comfort_temp_window"])
        # Native ranges of the commands the decisions span: [sp (A) | p (A) | r (A) | w_sp, w_dp | damper | valve].
        lo, hi = pad["native_lo"], pad["native_hi"]
        self.sat_range = (float(lo[0]), float(hi[0]))
        self.p_top = float(hi[A])
        self.w_range = (float(lo[3 * A]), float(hi[3 * A]))
        self.dp_max = float(hi[3 * A + 1])

    def zone_temps(self, obs: np.ndarray) -> np.ndarray:
        """Kelvin zone temperatures (B, V) from normalised observations (B, n_obs)."""
        return obs[:, self.i_temp] * self.t_std + self.t_mean

    def default_decision(self, active: np.ndarray) -> np.ndarray:
        """K10's choices as decisions, (n, n_dec): units on when the comfort mode is on or due (active), the band
        centre, supply 288 K, recirculation 0.67, the pressure floor, water 330 K."""
        values = {"run": np.where(active > 0.5, 1.0, 0.0), "target": 0.5,
                  "sat": (288.0 - self.sat_range[0]) / (self.sat_range[1] - self.sat_range[0]),
                  "recirc": 0.67 / self.max_recirc, "pressure": 0.0,
                  "water": (330.0 - self.w_range[0]) / (self.w_range[1] - self.w_range[0])}
        d = np.empty((len(active), self.n_dec))
        for name, cols in self.cols.items():
            d[:, cols] = np.asarray(values[name], dtype=float).reshape(-1, 1) if np.ndim(values[name]) else values[name]
        return np.clip(d, 0.0, 1.0)

    def _per_ahu(self, dec: np.ndarray, name: str) -> np.ndarray:
        """(B, A) values of a decision, broadcast from one column when it is not per AHU."""
        v = dec[:, self.cols[name]]
        return v if v.shape[1] == self.A else np.broadcast_to(v, (dec.shape[0], self.A))

    def act(self, obs: np.ndarray, integral: np.ndarray, dec: np.ndarray):
        """obs (B, n_obs), PI integrals (B, V), decisions (B, 6) in [0, 1] -> continuous (B, n_cont) and discrete
        (B, n_disc) actions, integrals'."""
        B, A, dt = obs.shape[0], self.A, self.dtype
        dec = dec.astype(dt, copy=False)
        run_a = (self._per_ahu(dec, "run") > 0.5).astype(dt)  # (B, A)
        run = run_a[:, self.vav_ahu]  # (B, V): each zone follows its AHU
        b_lo, b_hi = self.band
        target = b_lo + (b_hi - b_lo) * self._per_ahu(dec, "target")[:, self.vav_ahu]
        if self.deadband > 0.0:
            damper, valve, integral = self._two_loops(self.zone_temps(obs), target, integral, run)
        else:
            damper, valve, integral = self._one_loop(self.zone_temps(obs) - target, integral, run)
        sat = self.sat_range[0] + (self.sat_range[1] - self.sat_range[0]) * self._per_ahu(dec, "sat")
        recirc = self.max_recirc * self._per_ahu(dec, "recirc")
        pressure = self.p_floor + (self.p_top - self.p_floor) * self._per_ahu(dec, "pressure")
        water = self.w_range[0] + (self.w_range[1] - self.w_range[0]) * dec[:, self.cols["water"]]
        native = np.concatenate([sat, pressure, recirc, water, np.full((B, 1), self.dp_max, dtype=dt),
                                 damper, valve], 1)
        a = (native - self.lo) / (self.hi - self.lo) * (self.nhi - self.nlo) + self.nlo
        a_c = np.clip(a[:, self.inv_cont], -1.0, 1.0)
        a_d = np.concatenate([run_a, run_a.max(1, keepdims=True)], 1)[:, self.inv_disc]
        return a_c, a_d, integral.astype(dt, copy=False)

    def _sequenced(self, heat, cool):
        """Damper and valve from heating and cooling demands in [0, 1]: the valve takes the first half of the heating
        demand at minimum air, the damper the rest; cooling opens the damper."""
        valve = np.minimum(2.0 * heat, 1.0)
        opening = np.maximum(np.clip(2.0 * heat - 1.0, 0.0, 1.0), cool)
        return self.min_damper + (1.0 - self.min_damper) * opening, valve

    def _one_loop(self, err, integral, run):
        """K10's single PI loop on err = temperature - target."""
        u_raw = self.kp * err + self.ki * integral
        # Conditional integration (K10): integrate unless saturated and pushing further; reset while off.
        grow = (np.abs(u_raw) < 1.0) | (np.sign(err) != np.sign(u_raw))
        integral = np.where(grow, np.clip(integral + err, -1.0 / self.ki, 1.0 / self.ki), integral) * run
        u = np.clip(self.kp * err + self.ki * integral, -1.0, 1.0) * run
        if self.heat_seq:
            damper, valve = self._sequenced(np.maximum(-u, 0.0), np.maximum(u, 0.0))
        else:
            damper, valve = self.min_damper + (1.0 - self.min_damper) * np.abs(u), np.maximum(-u, 0.0)
        return damper, valve, integral

    def _pi_up(self, err, integral, run):
        """One-sided PI: demand in [0, 1] growing with err > 0, its integral kept in [0, 1 / ki]."""
        u_raw = self.kp * err + self.ki * integral
        stuck = ((u_raw <= 0.0) & (err < 0.0)) | ((u_raw >= 1.0) & (err > 0.0))
        integral = np.where(stuck, integral, np.clip(integral + err, 0.0, 1.0 / self.ki)) * run
        return np.clip(self.kp * err + self.ki * integral, 0.0, 1.0) * run, integral

    def _two_loops(self, temps, target, integral, run):
        """Cooling towards the target, heating towards the band floor + deadband, floating in between."""
        V = self.V
        t_heat = self.band[0] + self.deadband
        cool, i_cool = self._pi_up(temps - np.maximum(target, t_heat + 0.3), integral[:, :V], run)
        heat, i_heat = self._pi_up(t_heat - temps, integral[:, V:], run)
        damper, valve = self._sequenced(heat, cool)
        return damper, valve, np.concatenate([i_cool, i_heat], 1)
