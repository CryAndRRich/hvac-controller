"""Direction 2: the planner as a linear program on the building's exact linear thermal model.

The CEM planner (mpc.py) searches six building-wide knobs. This plans what the score is made of instead: how much
heat each zone gets or loses in every half hour of the next day, and when the air handlers run, by a linear program
on the same model (the modal ROM rebuilt from the metadata, build.py), whose thermal core is exactly linear.

Planning model (``LPModel``): the ROM's exact free response from the current state, plus a compact realisation of
its forced response, stepped in periods of ``period`` steps with the inputs held: x' = a^p x + (1 - a^p) / (1 - a)
* B w. The HVAC heat enters where the simulator puts it: as air power at each VAV (the ROM's air-power inputs).

Plant, linearised around a zone at 295 K, with the supply air of the period's regime (``LPPlanner._regime``):

- heating regime (outside below T_COOLING): the coil is idle, supply = mixed air at 70 % recirculation. Minimum airflow brings
  air power ``p0`` (negative: cold air). Heat comes first from the reheat valve at minimum airflow (``h1``, up to the
  308 K discharge cap, billed as gas one for one), then from more airflow (``h2``, dearer: the extra air must be
  heated from supply to zone temperature too);
- cooling regime: supply 285 K; extra airflow removes ``c``, billed at the coil as c * kappa / COP with
  kappa = (mixed - supply) / (zone - supply) (the outside-air share is cooled too; economiser below 295 K), plus fan.
  Reheat (from 285 K) stays available;
- running the air handlers (``o``, relaxed to [0, 1]) costs the minimum-airflow fans and coil; no heat or cooling
  without them.

Objective: the competition's own reward, linearised: energy at the period's tariff and carbon normalisers, and
0.6 x (degrees outside the comfort band) / 5 per zone, over 1/Z of the zones, in every step the building is
expected to be occupied (the reward's occupant-weighted mean, with occupants spread evenly).

The solution's zone temperatures at the end of each period are the targets the low-level controller tracks.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

T_ZONE = 295.0  # linearisation point of the zone temperature
T_COOL_SUPPLY = 285.0  # supply air in the cooling regime: the coldest the action range allows
T_DISCHARGE_MAX = 308.0  # VAV discharge cap (spec.vav["max_discharge"])
RECIRC = 0.7  # recirculation in the heating regime and above 295 K outside
P_COOL = 500.0  # Pa: the most static pressure planned for extra airflow
T_COOLING = 293.0  # outside air above this: cooling regime (coldest supply); below, the coil idles


class LPModel:
    """The planning model of one building's ROM. Heat inputs are in kW (the ROM takes W).

    The zone temperatures split exactly into a free response (the current state decaying, ``free_offset``, every
    mode kept) and a forced response from zero (the planned inputs), which the LP carries as its state. The forced
    response is realised compactly: modes that settle within ``fast_steps`` act as a static gain over the period;
    the others are grouped by time constant (bins ``ratio`` wide in log scale) and each group's input-output map is
    cut to its leading singular directions. Every mode's effect stays in, so responses keep their sign: a truncation
    that drops modes can make cooling now warm the zones later, which a linear program then exploits.
    """

    def __init__(self, rom, period: int, ratio: float = 1.4, max_rank: int = 12, rel_tol: float = 1e-3,
                 fast_steps: int = 3):
        a, b, c = rom.a, rom.b * 1000.0, rom.c  # b: per kW
        b[:, -1] /= 1000.0  # the outside-temperature column is in kelvin, not a heat input
        Z, V = rom.n_zones, rom.n_vavs
        self.Z, self.V, self.t_ref, self.period = Z, V, float(rom.t_ref), period
        self.a_all, self.C_all = a, c[V:]
        tau = 1.0 / np.maximum(1.0 - a, 1e-12)
        fast = tau <= fast_steps
        g = 1.0 / (1.0 - a[fast])  # steady state of x' = a x + b w
        cf = c[V:, fast]
        D = cf @ (g[:, None] * b[fast])
        self.Dq, self.Dp, self.Da = D[:, :Z], D[:, Z:Z + V], D[:, Z + V]
        a_list, sb_list, c_list = [], [], []
        log_tau = np.log(tau[~fast])
        idx = np.flatnonzero(~fast)
        if len(idx):
            edges = np.arange(log_tau.min(), log_tau.max() + np.log(ratio), np.log(ratio))
            which = np.clip(np.digitize(log_tau, edges) - 1, 0, len(edges) - 1)
            for k in np.unique(which):
                sel = idx[which == k]
                tau_k = float(np.exp(np.mean(np.log(tau[sel]))))
                a_k = 1.0 - 1.0 / tau_k
                M = c[V:, sel] @ b[sel]  # (Z, Z + V + 1)
                U, s, Wt = np.linalg.svd(M, full_matrices=False)
                keep = min(max_rank, int((s > rel_tol * s[0]).sum())) if s[0] > 0 else 0
                for j in range(keep):
                    a_list.append(a_k)
                    sb_list.append(s[j] * Wt[j])
                    c_list.append(U[:, j])
        a_r = np.array(a_list)
        B_r = np.array(sb_list).reshape(len(a_list), Z + V + 1)
        self.A = a_r ** period
        S = (1.0 - self.A) / (1.0 - a_r)  # sum_{j < period} a^j
        SB = S[:, None] * B_r
        self.SBq, self.SBp, self.SBa = SB[:, :Z], SB[:, Z:Z + V], SB[:, Z + V]
        self.Cz = np.array(c_list).reshape(len(c_list), Z).T

    def free_offset(self, x0: np.ndarray, P: int) -> np.ndarray:
        """(P, Z): the zone temperatures (deviation from t_ref) of the current state decaying freely, at the end of
        each period, every mode kept."""
        k = self.period * np.arange(1, P + 1)
        return (self.a_all[None, :] ** k[:, None] * np.asarray(x0)[None, :]) @ self.C_all.T

    @property
    def order(self) -> int:
        return len(self.A)


class LPPlanner:
    def __init__(self, spec, rom, plant, period: int = 12, n_periods: int = 24, time_limit: float = 20.0,
                 runs: str = "round", round_at: float = 0.25, **model_kw):
        self.spec, self.period, self.P, self.time_limit = spec, period, n_periods, time_limit
        self.runs, self.round_at = runs, round_at
        self.m = LPModel(rom, period, **model_kw)
        ph = spec.phys
        self.rho_cp = ph["air_rho"] * ph["air_cp"]
        v, ah = spec.vav, spec.ahu
        p_min = float(np.max(ah["min_pressure"]))
        p_top = max(row[5] for row in spec.cont if row[0] == "ahu_p")
        self.p_min, self.p_top = p_min, p_top
        self.f_min = v["min_damper"] * v["max_flow"] * np.sqrt(p_min / v["design_pressure"])  # (V,) m3/s
        # Extra airflow is planned at no more than P_COOL static pressure: fan power is flow x pressure and the
        # pressure a flow needs grows with its square, so the last few m3/s are the dearest by far.
        self.p_cool = min(P_COOL, p_top)
        self.f_max = np.minimum(v["max_flow"] * np.sqrt(self.p_cool / v["design_pressure"]), v["max_flow"])
        self.cop, self.fan_eff = float(np.mean(ah["cop"])), float(np.mean(ah["fan_eff"]))
        self.gain_on = np.asarray(plant.gain_on, dtype=float).reshape(-1)
        self.gain_off = np.asarray(plant.gain_off, dtype=float).reshape(-1)
        self.occ_w = spec.sim["occupant_w"]
        rw = spec.reward
        self.w = (rw["w_prod"], rw["w_cost"], rw["w_carbon"])
        self.w_sum = sum(self.w)

    # ---- per-period inputs ----------------------------------------------------------------------------------
    def _regime(self, t_out: np.ndarray) -> dict:
        """Supply temperature, coil factor and the minimum-airflow terms of each period, from outside air (P,)."""
        heating = t_out < T_COOLING
        mixed_heat = RECIRC * T_ZONE + (1.0 - RECIRC) * t_out
        mixed_cool = np.where(t_out < T_ZONE, t_out, RECIRC * T_ZONE + (1.0 - RECIRC) * t_out)  # economiser
        mixed = np.where(heating, mixed_heat, mixed_cool)
        supply = np.where(heating, mixed_heat, np.minimum(mixed_cool, T_COOL_SUPPLY))
        kappa = np.where(heating, 0.0, np.maximum(mixed - supply, 0.0) / (T_ZONE - supply))
        recirc = np.where(heating | (t_out >= T_ZONE), RECIRC, 0.0)
        return {"heating": heating, "mixed": mixed, "supply": supply, "kappa": kappa, "recirc": recirc}

    def _per_period(self, x: np.ndarray, P: int) -> np.ndarray:
        """Mean of each period of a per-step series (n_steps, ...) -> (P, ...)."""
        return x[:P * self.period].reshape((P, self.period) + x.shape[1:]).mean(1)

    def _sum_period(self, x: np.ndarray, P: int) -> np.ndarray:
        return x[:P * self.period].reshape((P, self.period) + x.shape[1:]).sum(1)

    # ---- the program ------------------------------------------------------------------------------------------
    def plan(self, x0: np.ndarray, ex: dict, margin: float = 0.2, offset: np.ndarray | None = None) -> dict:
        """x0: the tracked ROM state (full order). ex: per-step exogenous series from now on (numpy), with keys
        t_amb, comfort, occ_step, occ_rew, price_e, carbon_e, price_g, carbon_g, max_cost, max_carbon, win (n, 2).
        ``offset`` (Z,): measured minus modelled zone temperatures now, carried over the horizon (offset-free MPC:
        the plan starts from the zones as measured, whatever the model's materials get wrong).
        Returns per-period zone temperatures, heat, cooling and air-handler runs, and the regime."""
        m, p = self.m, self.period
        P = min(self.P, len(ex["t_amb"]) // p)
        Z, V, r = m.Z, m.V, m.order
        t_out = self._per_period(ex["t_amb"], P)
        occupied = ex["occ_rew"][:P * p].sum(1) > 1e-3
        rg = self._regime(t_out)
        comfort = self._per_period(ex["comfort"], P)
        q = (np.where(comfort[:, None] > 0.5, self.gain_on, self.gain_off)
             + self._per_period(ex["occ_step"], P) * self.occ_w) / 1000.0  # (P, Z) kW
        dT = t_out - m.t_ref
        # Normalised reward per kW and per step, summed over each period's steps.
        el = (self.w[1] * ex["price_e"] / ex["max_cost"] + self.w[2] * ex["carbon_e"] / ex["max_carbon"]) * 1000.0
        gas = (self.w[1] * ex["price_g"] / ex["max_cost"] + self.w[2] * ex["carbon_g"] / ex["max_carbon"]) * 1000.0
        el, gas = self._sum_period(el, P) / self.w_sum, self._sum_period(gas, P) / self.w_sum
        w_comfort = self._sum_period(occupied.astype(float), P) * self.w[0] / self.w_sum / 5.0 / Z  # per K per zone
        # The band a period's end must meet: the tightest window over its occupied steps.
        win = ex["win"][:P * p]
        w_lo = np.where(occupied, win[:, 0], -np.inf).reshape(P, p).max(1)
        w_hi = np.where(occupied, win[:, 1], np.inf).reshape(P, p).min(1)
        lo, hi = w_lo + margin, w_hi - margin
        # Plant terms per period (kW).
        supply = rg["supply"]
        p0 = self.f_min[None, :] * self.rho_cp * (supply[:, None] - T_ZONE) / 1000.0  # (P, V)
        h1_max = self.f_min[None, :] * self.rho_cp * (T_DISCHARGE_MAX - supply[:, None]) / 1000.0
        extra = (self.f_max - self.f_min)[None, :]
        h2_max = extra * self.rho_cp * (T_DISCHARGE_MAX - T_ZONE) / 1000.0 * np.ones((P, 1))
        h2_cost = (T_DISCHARGE_MAX - supply) / (T_DISCHARGE_MAX - T_ZONE)  # gas per kW delivered
        c_max = extra * self.rho_cp * np.maximum(T_ZONE - supply[:, None], 0.5) / 1000.0
        fan_per_flow = self.p_min * (2.0 - rg["recirc"]) / self.fan_eff / 1000.0  # kW per m3/s at minimum flow
        fan_cool = self.p_cool * (2.0 - rg["recirc"]) / self.fan_eff / 1000.0  # kW per m3/s of extra flow
        fan_per_c = fan_cool / (self.rho_cp * np.maximum(T_ZONE - supply, 0.5) / 1000.0)  # kW fan per kW cooling
        f_tot = float(self.f_min.sum())
        on_cost = el * (f_tot * fan_per_flow + f_tot * self.rho_cp * np.maximum(rg["mixed"] - supply, 0.0) / 1000.0
                        / self.cop)

        # Variable layout.
        nX, nU = P * r, P * V
        iX, iH1, iH2, iC = 0, nX, nX + nU, nX + 2 * nU
        iO = nX + 3 * nU
        iSL = iO + P
        iSH = iSL + P * Z
        n = iSH + P * Z

        def blk(i0, t, width):
            return i0 + t * width

        # Equalities: x_t - A x_{t-1} - SBp (h1 + h2 - c + p0 o) = SBq q + SBa dT (+ A x0 at t = 0).
        rows, cols, vals, beq = [], [], [], np.empty(nX)
        x0s = np.zeros(r)  # the LP state is the forced response, from zero; the state x0 enters as free_offset
        eye = np.arange(r)
        sbp = m.SBp
        sbp_r, sbp_c = np.nonzero(np.abs(sbp) > 0.0)
        sbp_v = sbp[sbp_r, sbp_c]
        for t in range(P):
            r0 = t * r
            rows.append(r0 + eye), cols.append(blk(iX, t, r) + eye), vals.append(np.ones(r))
            if t > 0:
                rows.append(r0 + eye), cols.append(blk(iX, t - 1, r) + eye), vals.append(-m.A)
            for i0, sign in ((iH1, -1.0), (iH2, -1.0), (iC, 1.0)):
                rows.append(r0 + sbp_r), cols.append(blk(i0, t, V) + sbp_c), vals.append(sign * sbp_v)
            rows.append(r0 + eye), cols.append(np.full(r, iO + t)), vals.append(-(sbp @ p0[t]))
            beq[r0:r0 + r] = m.SBq @ q[t] + m.SBa * dT[t] + (m.A * x0s if t == 0 else 0.0)
        A_eq = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nX, n))

        # Inequalities: comfort in occupied periods, capacity.
        rows, cols, vals, bub = [], [], [], []
        nrow = 0
        cz_r, cz_c = np.nonzero(np.abs(m.Cz) > 1e-12)
        cz_v = m.Cz[cz_r, cz_c]
        dp_r, dp_c = np.nonzero(np.abs(m.Dp) > 1e-12)
        dp_v = m.Dp[dp_r, dp_c]
        zr = np.arange(Z)
        free = m.free_offset(x0, P)
        if offset is not None:
            free = free + np.asarray(offset, dtype=float)[None, :]
        for t in range(P):
            if w_comfort[t] <= 0.0:
                continue
            base = m.t_ref + m.Dq @ q[t] + m.Da * dT[t] + free[t]  # (Z,)
            dpo = m.Dp @ p0[t]
            for sign, bound, i_s in ((-1.0, -(lo[t] - base), iSL), (1.0, hi[t] - base, iSH)):
                rows.append(nrow + cz_r), cols.append(blk(iX, t, r) + cz_c), vals.append(sign * cz_v)
                for i0, s2 in ((iH1, 1.0), (iH2, 1.0), (iC, -1.0)):
                    rows.append(nrow + dp_r), cols.append(blk(i0, t, V) + dp_c), vals.append(sign * s2 * dp_v)
                rows.append(nrow + zr), cols.append(np.full(Z, iO + t)), vals.append(sign * dpo)
                rows.append(nrow + zr), cols.append(blk(i_s, t, Z) + zr), vals.append(-np.ones(Z))
                bub.append(bound)
                nrow += Z
        vr = np.arange(V)
        for t in range(P):
            for i0, cap in ((iH1, h1_max[t]), (iH2, h2_max[t]), (iC, c_max[t])):
                rows.append(nrow + vr), cols.append(blk(i0, t, V) + vr), vals.append(np.ones(V))
                rows.append(nrow + vr), cols.append(np.full(V, iO + t)), vals.append(-cap)
                bub.append(np.zeros(V))
                nrow += V
        A_ub = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nrow, n))
        b_ub = np.concatenate(bub)

        # Objective.
        cost = np.zeros(n)
        cost[iH1:iH2] = np.repeat(gas, V)
        cost[iH2:iC] = np.repeat(gas * h2_cost, V)
        cost[iC:iO] = np.repeat(el * (rg["kappa"] / self.cop + fan_per_c), V)
        cost[iO:iSL] = on_cost
        cost[iSL:iSH] = np.repeat(w_comfort, Z)
        cost[iSH:] = np.repeat(w_comfort, Z)
        bounds = np.zeros((n, 2))
        bounds[:, 1] = np.inf
        bounds[iX:iH1, 0] = -np.inf
        bounds[iO:iSL, 1] = 1.0
        # The air handlers run or not: a fraction would buy capacity at a fraction of its fixed cost (fans and
        # outside air at minimum flow), which no real run can. ``runs`` = "milp" solves it exactly; "round" solves
        # the relaxation, rounds the runs up from ``round_at`` and solves again with them fixed (two LPs, much
        # faster, and in the cases checked within a few per cent of the MILP).
        def solve(bnds, integer):
            integrality = np.zeros(n, dtype=np.int64)
            if integer:
                integrality[iO:iSL] = 1
            res = None
            for presolve in (True, False):  # HiGHS's presolve sometimes ends in "numerical difficulties" here
                res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=beq, bounds=bnds, method="highs",
                              integrality=integrality if integer else None,
                              options={"time_limit": self.time_limit, "mip_rel_gap": 1e-3, "presolve": presolve})
                if res.x is not None:
                    break
            return res

        if self.runs == "milp":
            res = solve(bounds, True)
        else:
            res = solve(bounds, False)
            if res.x is not None:
                fixed = bounds.copy()
                on = (res.x[iO:iSL] > self.round_at).astype(float)
                fixed[iO:iSL, 0] = fixed[iO:iSL, 1] = on
                res = solve(fixed, False)
        if res.x is None:
            return {"ok": False, "message": res.message, "P": P}
        z = res.x
        X = z[iX:iH1].reshape(P, r)
        h = z[iH1:iH2].reshape(P, V) + z[iH2:iC].reshape(P, V)
        c = z[iC:iO].reshape(P, V)
        o = z[iO:iSL]
        u = h - c + p0 * o[:, None]
        temps = X @ m.Cz.T + u @ m.Dp.T + q @ m.Dq.T + np.outer(dT, m.Da) + free + m.t_ref  # (P, Z)
        return {"ok": True, "P": P, "temps": temps, "heat": h, "cool": c, "run": o, "regime": rg, "t_out": t_out,
                "objective": float(res.fun), "occupied": w_comfort > 0.0, "flow_needed":
                self.f_min[None, :] + c / (self.rho_cp * np.maximum(T_ZONE - supply[:, None], 0.5) / 1000.0)}
