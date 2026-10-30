"""Direction 2: the student's view of a building, one fixed-length vector whatever the building's size.

The teacher (an MPC that plans on the exact model) labels states with supervisory decisions; the student learns to
make them from what the gateway gives it at every step, with no model and no planning. Its input must therefore
mean the same thing in a 30-zone building as in an 85-zone one, so zones enter only through statistics. Everything
here comes from the observation and the metadata, and the same ``FeatureBuilder`` runs when labelling (teacher.py)
and when acting (student.py), so the two cannot drift apart.

Groups (``NAMES`` lists them in order):
- time: hour of day and day of week (cyclic), the comfort flags, hours until the next comfort period starts and
  until the current one ends;
- people: occupants per zone;
- outside air: now, its 1/6/24 h averages and its range over the last day;
- zones: mean, min, max and spread of the zone temperatures; how far the coldest and warmest zones are inside the
  comfort band; the share of zones outside it; their change over the last hour and six hours;
- plant: the hot-water tank temperature;
- building: size (log zones), air handlers per zone, floors.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from hvac_controller.d1.policy import Layout

NAMES = (
    "hod_sin", "hod_cos", "dow_sin", "dow_cos", "comfort_now", "comfort_soon", "h_to_start", "h_to_end",
    "occ_per_zone",
    "oat", "oat_1h", "oat_6h", "oat_24h", "oat_day_min", "oat_day_max",
    "t_mean", "t_min", "t_max", "t_std", "margin_lo", "margin_hi", "share_cold", "share_warm",
    "dt_1h_mean", "dt_1h_min", "dt_1h_max", "dt_6h_mean",
    "tank",
    "log_zones", "ahus_per_zone", "floors",
)
N_FEATURES = len(NAMES)
STEPS_PER_HOUR = 12
_EMA = {"oat_1h": 12, "oat_6h": 72, "oat_24h": 288}  # time constants in steps


class FeatureBuilder:
    def __init__(self, meta: dict):
        lay = Layout(meta)
        names = list(meta["observation_field_names"])
        self.lay = lay
        self.i_temp, self.c_temp = np.asarray(lay.i_temp), lay.c_temp
        self.i_oat, self.c_oat = lay.i_temps[0], lay.c_temps[0]
        self.i_tank, self.c_tank = lay.i_temps[1], lay.c_temps[1]
        self.i = {n: names.index(n) for n in ("hod_sin_000", "hod_cos_000", "dow_sin_000", "dow_cos_000",
                                              "comfort_mode_now", "comfort_mode_soon", "num_occupants")}
        self.lo, self.hi = (float(v) for v in meta["comfort_temp_window"])
        self.start_h, self.end_h = float(meta["morning_start_hour"]), float(meta["evening_start_hour"])
        V, A = lay.V, lay.A
        self.static = np.array([math.log(V) / 4.0, A / V * 5.0, float(meta.get("num_floors") or 1) / 4.0])
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.ema = {k: 0.0 for k in _EMA}
        self.oat_day = deque(maxlen=288)
        self.temps = deque(maxlen=73)  # zone temperature vectors of the last 6 h, for the changes

    # ---- schedule ----------------------------------------------------------------------------------------------
    def _hours(self, hod: float, dow: int) -> tuple[float, float]:
        """Hours until the next comfort period starts and until the current one ends (0 outside one), working days
        Monday to Friday."""
        in_comfort = dow < 5 and self.start_h <= hod < self.end_h
        to_end = self.end_h - hod if in_comfort else 0.0
        to_start = 0.0
        for d in range(8):
            day = (dow + d) % 7
            if day >= 5:
                continue
            t = d * 24.0 + self.start_h - hod
            if t > 0.0:
                to_start = t
                break
        return min(to_start, 72.0), to_end

    # ---- one step ----------------------------------------------------------------------------------------------
    def update(self, obs: np.ndarray) -> np.ndarray:
        """The features of this observation; call once per step, in order (the averages and changes carry over)."""
        obs = np.asarray(obs, dtype=np.float64)
        temps = obs[self.i_temp] * self.c_temp[:, 1] + self.c_temp[:, 0]
        oat = float(obs[self.i_oat] * self.c_oat[1] + self.c_oat[0])
        tank = float(obs[self.i_tank] * self.c_tank[1] + self.c_tank[0])
        hs, hc = obs[self.i["hod_sin_000"]], obs[self.i["hod_cos_000"]]
        ds, dc = obs[self.i["dow_sin_000"]], obs[self.i["dow_cos_000"]]
        hod = (math.atan2(hs, hc) % (2 * math.pi)) / (2 * math.pi) * 24.0
        dow = int(round((math.atan2(ds, dc) % (2 * math.pi)) / (2 * math.pi) * 7)) % 7
        to_start, to_end = self._hours(hod, dow)
        count = float(obs[self.i["num_occupants"]])
        count = count if count >= 0.0 else count * 126.0 + 125.0  # the environment reports the raw count
        self.n += 1
        for k, tau in _EMA.items():
            a = max(1.0 / tau, 1.0 / self.n)
            self.ema[k] += a * (oat - self.ema[k])
        self.oat_day.append(oat)
        self.temps.append(temps)
        t1 = self.temps[-13] if len(self.temps) > 12 else self.temps[0]
        t6 = self.temps[0]
        d1 = temps - t1
        lo, hi = self.lo, self.hi
        f = np.array([
            hs, hc, ds, dc, obs[self.i["comfort_mode_now"]], obs[self.i["comfort_mode_soon"]], to_start / 24.0,
            to_end / 12.0,
            count / len(temps) / 2.0,
            (oat - 295.0) / 10.0, (self.ema["oat_1h"] - 295.0) / 10.0, (self.ema["oat_6h"] - 295.0) / 10.0,
            (self.ema["oat_24h"] - 295.0) / 10.0, (min(self.oat_day) - 295.0) / 10.0,
            (max(self.oat_day) - 295.0) / 10.0,
            (temps.mean() - 295.5) / 2.0, (temps.min() - 295.5) / 2.0, (temps.max() - 295.5) / 2.0, temps.std(),
            temps.min() - lo, hi - temps.max(), float((temps < lo).mean()), float((temps > hi).mean()),
            d1.mean(), d1.min(), d1.max(), (temps - t6).mean() / 3.0,
            (tank - 330.0) / 20.0,
            *self.static,
        ], dtype=np.float32)
        return f
