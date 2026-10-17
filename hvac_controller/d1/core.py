"""Step A of direction 1: isolate the simulator's thermal core and check that it is the linear system we think.

The organisers' step (simulator_flexible_floor_plan.execute_step_sim) is: equipment equations -> per-zone heat
``input_q`` and per-VAV air-side conductance ``g`` and supply temperature ``Ts`` -> one backward-Euler solve
(FiniteDifferenceSolver.solve, exact air-side Woodbury) -> a random intra-room swap (apply_convection) -> next step.

``CoreRecorder`` wraps the solver's ``solve`` on one environment. At every call it

1. reduces ``input_q`` to one number per zone (Building.apply_thermal_power_zone spreads a zone's heat evenly over
   every CV of the zone; only the VAV air stream lands on the diffusers, through the solver's ``_applied``) and checks
   that the per-zone numbers rebuild the field exactly;
2. re-runs the step with our own pure function from the environment's own start state (``one_step`` error: zero
   if we understand the core);
3. advances a free-running replay that never sees the swap (``free_run`` error: the swap's accumulated effect on what
   the VAVs sense, or zero when the swap is switched off);
4. logs the per-step inputs and sensed outputs, so a reduced model can later be replayed on exactly the same inputs.

``save_core`` writes the matrices step B needs. Nothing here changes the simulation itself.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse

from hvac_controller.sim import load_env
from smart_buildings import evaluate

NON_ZONES = ("exterior_space", "interior_wall")
POLICY = "pid_water"


def solver_of(env):
    return env.building.simulator._solver


def grid_of(env):
    return env.building.simulator.building


def zone_spread_matrix(env) -> tuple[scipy.sparse.csc_matrix, list[str]]:
    """U (n_solved x n_zones): column z holds 1 / n_z on each of zone z's n_z CVs, indexed by solved position.

    This is how apply_thermal_power_zone spreads a zone's heat (input_q[rows, cols] = power / rows.size).
    """
    solver, grid = solver_of(env), grid_of(env)
    n_cells = grid.temp.size
    position = np.full(n_cells, -1, dtype=np.int64)
    position[solver._solved] = np.arange(len(solver._solved))
    n_cols = grid.temp.shape[1]
    rows, cols, vals, names = [], [], [], []
    for name, coords in grid.room_dict.items():
        if name in NON_ZONES or not coords:
            continue
        r, c = np.transpose(np.asarray(coords))
        pos = position[r * n_cols + c]
        if np.any(pos < 0):
            raise ValueError(f"zone {name} has a CV pinned to ambient")
        rows.append(pos)
        cols.append(np.full(len(pos), len(names)))
        vals.append(np.full(len(pos), 1.0 / len(pos)))
        names.append(name)
    d = scipy.sparse.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(solver._solved), len(names)),
    )
    return d, names


def core_step(solver, temp_solved, mass, q_solved, t_amb, h, g, ts):
    """FiniteDifferenceSolver.solve written as a pure function of (state, inputs) -> (state', mass')."""
    lu = solver._factorization
    exposure = solver.ambient_conductance(h)
    forcing = (solver._capacitance_rate * temp_solved + q_solved
               + t_amb * (solver._dirichlet_conductance + exposure))
    forcing = forcing + solver._mass_coupling * mass
    solved = lu.solve(forcing)
    if g is not None:
        solved = solver._add_air_side(lu, solved, g, ts)
    mass = mass * solver._mass_retained + solver._mass_gained * solved
    return solved, mass


def core_from_env(env):
    """The thermal core of a freshly reset environment (before its first step), as rom.Core; no rollout needed."""
    from hvac_controller.d1.rom import Core

    sim = env.building.simulator
    s = solver_of(env)
    h = float(sim._weather_controller.get_air_convection_coefficient(sim.current_timestamp))
    zones, names = zone_spread_matrix(env)
    return Core(
        a=s.matrix(h).tocsc(), cap=s._capacitance_rate.copy(), amb=s._dirichlet_conductance + s.ambient_conductance(h),
        mass_coupling=s._mass_coupling.copy(), mass_retained=s._mass_retained.copy(),
        mass_gained=s._mass_gained.copy(), applied=s._applied.tocsc(), sensed=s._sensed.tocsr(), zones=zones,
        temp0=np.asarray(grid_of(env).temp, dtype=np.float64).reshape(-1)[s._solved].copy(),
        mass0=s._mass_temp.copy(), zone_names=names, h=h,
    )


class CoreRecorder:
    """Wraps the solver's solve. verify=False only logs the per-step inputs and sensed outputs (no extra solves)."""

    def __init__(self, env, verify: bool = True):
        self.env = env
        self.verify = verify
        self.solver = solver_of(env)
        self.zones, self.zone_names = zone_spread_matrix(env)
        # Indicator of zone membership: summing a zone's CVs gives back its power (columns of U sum to one).
        self._member_t = self.zones.T.astype(bool).astype(np.float64).tocsr()
        self.log: dict[str, list] = {k: [] for k in (
            "q_zone", "g", "ts", "t_amb", "h", "sensed_env", "sensed_free",
            "one_step_err", "q_rebuild_err", "free_run_err")}
        self._free = None  # (temp_solved, mass) of the free-running replay
        self._original = self.solver.solve
        self.solver.solve = self._wrapped
        self.initial = None

    def detach(self):
        self.solver.solve = self._original

    def _sensed(self, solved):
        return np.asarray(self.solver._sensed @ solved).ravel()

    def _wrapped(self, temp, input_q, ambient_temperature, convection_coefficient,
                 air_side_conductance=None, supply_temperature=None):
        s = self.solver
        idx = s._solved
        temp_solved = np.asarray(temp, dtype=np.float64).reshape(-1)[idx].copy()
        mass_before = s._mass_temp.copy()
        q_field = np.asarray(input_q, dtype=np.float64).reshape(-1)[idx]
        q_zone = np.asarray(self._member_t @ q_field).ravel()
        q_rebuilt = np.asarray(self.zones @ q_zone).ravel()
        g = None if air_side_conductance is None else np.asarray(air_side_conductance, dtype=np.float64).copy()
        ts = None if supply_temperature is None else np.asarray(supply_temperature, dtype=np.float64).copy()
        h = float(convection_coefficient)
        t_amb = float(ambient_temperature)

        out = self._original(temp, input_q, ambient_temperature, convection_coefficient,
                             air_side_conductance, supply_temperature)
        out_solved = np.asarray(out, dtype=np.float64).reshape(-1)[idx]
        if not self.verify:
            lg = self.log
            lg["q_zone"].append(q_zone)
            lg["g"].append(np.zeros(0) if g is None else g)
            lg["ts"].append(np.zeros(0) if ts is None else ts)
            lg["t_amb"].append(t_amb)
            lg["h"].append(h)
            lg["sensed_env"].append(self._sensed(out_solved))
            return out

        if self._free is None:
            self._free = (temp_solved.copy(), mass_before.copy())
            self.initial = {"temp_solved": temp_solved.copy(), "mass": mass_before.copy()}
        mine, _ = core_step(s, temp_solved, mass_before, q_rebuilt, t_amb, h, g, ts)
        free_t, free_m = core_step(s, *self._free, q_rebuilt, t_amb, h, g, ts)
        self._free = (free_t, free_m)

        sensed_env = self._sensed(out_solved)
        sensed_free = self._sensed(free_t)
        lg = self.log
        lg["q_zone"].append(q_zone)
        lg["g"].append(np.zeros(0) if g is None else g)
        lg["ts"].append(np.zeros(0) if ts is None else ts)
        lg["t_amb"].append(t_amb)
        lg["h"].append(h)
        lg["sensed_env"].append(sensed_env)
        lg["sensed_free"].append(sensed_free)
        lg["one_step_err"].append(float(np.max(np.abs(mine - out_solved))))
        lg["q_rebuild_err"].append(float(np.max(np.abs(q_rebuilt - q_field))))
        lg["free_run_err"].append(float(np.max(np.abs(sensed_free - sensed_env))))
        return out

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.log.items()}


def _at(series: np.ndarray, steps: list[int]) -> dict[str, float]:
    return {f"{k}_steps": float(series[min(k, len(series)) - 1]) for k in steps if len(series)}


def run_core_check(config_gin: Path, n_steps: int, swap: bool, out_dir: Path | None = None) -> dict:
    """Roll the baseline for n_steps with the recorder attached; report exactness and the swap's effect."""
    overrides = None if swap else {"StochasticConvectionSimulator.p": 0.0}
    t0 = time.time()
    env, meta = load_env(config_gin, gin_overrides=overrides)
    policy = evaluate.initialize_policy(evaluate.POLICY_MAP[POLICY], meta, env.action_spec())
    obs, _ = env.reset()
    rec = CoreRecorder(env)  # attach after reset: reset() does not call solve
    obs = np.asarray(obs)
    for _ in range(n_steps):
        action = evaluate.validate_action(evaluate.get_policy_action(policy, obs), env.action_spec())
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    rec.detach()
    a = rec.arrays()
    seconds = time.time() - t0
    free = np.abs(a["sensed_free"] - a["sensed_env"])  # (steps, V)
    checkpoints = [12, 288, len(free)]
    report = {
        "config": str(config_gin),
        "swap": swap,
        "steps": int(len(free)),
        "seconds": seconds,
        "n_solved": int(len(rec.solver._solved)),
        "n_zones": len(rec.zone_names),
        "n_vavs": int(rec.solver._sensed.shape[0]),
        "h": float(a["h"][0]) if len(a["h"]) else None,
        "max_one_step_err_k": float(a["one_step_err"].max()),
        "max_q_rebuild_err_w": float(a["q_rebuild_err"].max()),
        "free_run_max_err_k": _at(np.maximum.accumulate(free.max(axis=1)), checkpoints),
        "free_run_mean_err_k": _at(free.mean(axis=1), checkpoints),
        "sensed_range_k": [float(a["sensed_env"].min()), float(a["sensed_env"].max())],
        "t_amb_range_k": [float(a["t_amb"].min()), float(a["t_amb"].max())],
        "share_steps_air_on": float(np.mean([gg.sum() > 0 for gg in a["g"]])) if len(a["g"]) else None,
    }
    if out_dir is not None:
        save_core(env, rec, a, Path(out_dir))
        (Path(out_dir) / "report.json").write_text(json.dumps(report, indent=1))
    return report


def save_core(env, rec: CoreRecorder, a: dict[str, np.ndarray], out_dir: Path) -> None:
    """Matrices and logged inputs/outputs that step B (reduced model) needs, for one (building, h)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    s = rec.solver
    h = float(a["h"][0])
    scipy.sparse.save_npz(out_dir / "matrix.npz", s.matrix(h).tocsc())
    scipy.sparse.save_npz(out_dir / "applied.npz", s._applied.tocsc())
    scipy.sparse.save_npz(out_dir / "sensed.npz", s._sensed.tocsr())
    scipy.sparse.save_npz(out_dir / "zones.npz", rec.zones.tocsc())
    np.savez_compressed(
        out_dir / "vectors.npz",
        solved=s._solved, shape=np.array(s._shape), h=np.array(h),
        capacitance_rate=s._capacitance_rate, dirichlet=s._dirichlet_conductance,
        exposure=s.ambient_conductance(h), mass_coupling=s._mass_coupling,
        mass_retained=s._mass_retained, mass_gained=s._mass_gained,
        temp0=rec.initial["temp_solved"], mass0=rec.initial["mass"],
        zone_names=np.array(rec.zone_names),
    )
    np.savez_compressed(
        out_dir / "sequence.npz",
        q_zone=a["q_zone"], g=np.stack(a["g"]), ts=np.stack(a["ts"]), t_amb=a["t_amb"], h=a["h"],
        sensed_env=a["sensed_env"], sensed_free=a["sensed_free"],
    )
