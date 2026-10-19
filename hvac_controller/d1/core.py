"""The simulator's thermal core, as the linear system step B reduces (step A, passed 2026-10-18; D1 doc §5).

The organisers' step (simulator_flexible_floor_plan.execute_step_sim) is: equipment equations -> per-zone heat
``input_q`` and per-VAV air-side conductance ``g`` and supply temperature ``Ts`` -> one backward-Euler solve
(FiniteDifferenceSolver.solve, exact air-side Woodbury) -> a random intra-room swap (apply_convection) -> next step.
Step A showed that a pure function of (state, per-zone heat, g, Ts, outside temperature) reproduces that solve to
1e-12 K and that ignoring the swap costs at most 1.5e-3 K over 4 days.

- ``core_from_env``: the core of a freshly reset environment (matrices and initial state), which rom.build_rom reduces.
- ``CoreRecorder``: wraps the solver's ``solve`` and logs its per-step inputs (heat per zone, g, Ts, outside
  temperature, convection coefficient), which env_check compares with the plant equations.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse

from hvac_controller.d1.spec import NON_ZONES


def zone_spread_matrix(env) -> tuple[scipy.sparse.csc_matrix, list[str]]:
    """U (n_solved x n_zones): column z holds 1 / n_z on each of zone z's n_z CVs, indexed by solved position.

    This is how apply_thermal_power_zone spreads a zone's heat (input_q[rows, cols] = power / rows.size).
    """
    solver, grid = env.building.simulator._solver, env.building.simulator.building
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


def core_from_env(env):
    """The thermal core of a freshly reset environment (before its first step), as rom.Core; no rollout needed."""
    from hvac_controller.d1.rom import Core

    sim = env.building.simulator
    s = sim._solver
    h = float(sim._weather_controller.get_air_convection_coefficient(sim.current_timestamp))
    zones, names = zone_spread_matrix(env)
    return Core(
        a=s.matrix(h).tocsc(), cap=s._capacitance_rate.copy(), amb=s._dirichlet_conductance + s.ambient_conductance(h),
        mass_coupling=s._mass_coupling.copy(), mass_retained=s._mass_retained.copy(),
        mass_gained=s._mass_gained.copy(), applied=s._applied.tocsc(), sensed=s._sensed.tocsr(), zones=zones,
        temp0=np.asarray(sim.building.temp, dtype=np.float64).reshape(-1)[s._solved].copy(),
        mass0=s._mass_temp.copy(), zone_names=names, h=h,
    )


class CoreRecorder:
    """Wraps the solver's solve and logs its inputs per step, reduced to one heat value per zone."""

    def __init__(self, env):
        self.solver = env.building.simulator._solver
        zones, _ = zone_spread_matrix(env)
        # Indicator of zone membership: summing a zone's CVs gives back its power (columns of U sum to one).
        self._member_t = zones.T.astype(bool).astype(np.float64).tocsr()
        self.log: dict[str, list] = {k: [] for k in ("q_zone", "g", "ts", "t_amb", "h")}
        self._original = self.solver.solve
        self.solver.solve = self._wrapped

    def detach(self):
        self.solver.solve = self._original

    def _wrapped(self, temp, input_q, ambient_temperature, convection_coefficient,
                 air_side_conductance=None, supply_temperature=None):
        q_field = np.asarray(input_q, dtype=np.float64).reshape(-1)[self.solver._solved]
        lg = self.log
        lg["q_zone"].append(np.asarray(self._member_t @ q_field).ravel())
        lg["g"].append(np.zeros(0) if air_side_conductance is None
                       else np.asarray(air_side_conductance, dtype=np.float64).copy())
        lg["ts"].append(np.zeros(0) if supply_temperature is None
                        else np.asarray(supply_temperature, dtype=np.float64).copy())
        lg["t_amb"].append(float(ambient_temperature))
        lg["h"].append(float(convection_coefficient))
        return self._original(temp, input_q, ambient_temperature, convection_coefficient,
                              air_side_conductance, supply_temperature)

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.log.items()}
