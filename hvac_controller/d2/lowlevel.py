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

The PI law is K10's (hvac_controller/baseline_k10.py, ZonePIPolicy._command_zones): u = clip(Kp e + Ki I, -1, 1)
with conditional integration, damper = min + (1 - min) |u|, reheat valve = max(0, -u).

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
                 max_recirc: float = 0.7, p_floor: float = 200.0, dtype=np.float32):
        lay = Layout(meta)
        A, V = lay.A, lay.V
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
        """K10's choices as decisions, (n, 6): units on when the comfort mode is on or due (active), the band
        centre, supply 288 K, recirculation 0.67, the pressure floor, water 330 K."""
        d = np.empty((len(active), N_DEC))
        d[:, 0] = np.where(active > 0.5, 1.0, 0.0)
        d[:, 1] = 0.5
        d[:, 2] = (288.0 - self.sat_range[0]) / (self.sat_range[1] - self.sat_range[0])
        d[:, 3] = 0.67 / self.max_recirc
        d[:, 4] = 0.0
        d[:, 5] = (330.0 - self.w_range[0]) / (self.w_range[1] - self.w_range[0])
        return np.clip(d, 0.0, 1.0)

    def act(self, obs: np.ndarray, integral: np.ndarray, dec: np.ndarray):
        """obs (B, n_obs), PI integrals (B, V), decisions (B, 6) in [0, 1] -> continuous (B, n_cont) and discrete
        (B, n_disc) actions, integrals'."""
        B, A, dt = obs.shape[0], self.A, self.dtype
        dec = dec.astype(dt, copy=False)
        run = (dec[:, 0:1] > 0.5).astype(dt)
        b_lo, b_hi = self.band
        target = b_lo + (b_hi - b_lo) * dec[:, 1:2]
        err = self.zone_temps(obs) - target
        u_raw = self.kp * err + self.ki * integral
        # Conditional integration (K10): integrate unless saturated and pushing further; reset while off.
        grow = (np.abs(u_raw) < 1.0) | (np.sign(err) != np.sign(u_raw))
        integral = np.where(grow, np.clip(integral + err, -1.0 / self.ki, 1.0 / self.ki), integral) * run
        u = np.clip(self.kp * err + self.ki * integral, -1.0, 1.0) * run
        damper = self.min_damper + (1.0 - self.min_damper) * np.abs(u)
        valve = np.maximum(-u, 0.0)
        sat = self.sat_range[0] + (self.sat_range[1] - self.sat_range[0]) * dec[:, 2:3]
        recirc = self.max_recirc * dec[:, 3:4]
        pressure = self.p_floor + (self.p_top - self.p_floor) * dec[:, 4:5]
        water = self.w_range[0] + (self.w_range[1] - self.w_range[0]) * dec[:, 5:6]
        native = np.concatenate([np.broadcast_to(sat, (B, A)), np.broadcast_to(pressure, (B, A)),
                                 np.broadcast_to(recirc, (B, A)), water, np.full((B, 1), self.dp_max, dtype=dt),
                                 damper, valve], 1)
        a = (native - self.lo) / (self.hi - self.lo) * (self.nhi - self.nlo) + self.nlo
        a_c = np.clip(a[:, self.inv_cont], -1.0, 1.0)
        a_d = np.broadcast_to(run, (B, A + 1))[:, self.inv_disc]
        return a_c, a_d, integral.astype(dt, copy=False)
