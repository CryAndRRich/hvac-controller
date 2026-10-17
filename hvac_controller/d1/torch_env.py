"""The differentiable copy of one scenario: ROM (step B) + plant (plant.py), batched, on NumPy or PyTorch.

``BuildingEnv`` follows the interface SHAC-style learners use (DiffRL, rewarped): ``num_envs``, ``num_obs``,
``num_actions``, ``reset``, ``step(action) -> obs, reward, done, extras``, ``initialize_trajectory``,
``clear_grad`` and ``get_checkpoint``. The action is the competition's hybrid action: continuous values in [-1, 1]
and discrete run commands in {0, 1} (any value in [0, 1] when ``exact=False``).

One environment holds one building (one ROM). Its batch runs B copies of it, each with its own actions; the
exogenous series is shared and every copy sits at the same step. With the NumPy backend it is the validation
replica that env_check compares with the organisers' simulator.
"""

from __future__ import annotations

import numpy as np

from hvac_controller.d1.ops import Ops
from hvac_controller.d1.plant import Plant
from hvac_controller.d1.rom import LinearRom
from hvac_controller.d1.spec import BuildingSpec, Exogenous

EXO_STEP = ("t_amb", "t_amb_next", "comfort", "win_now", "win_next", "occ_step", "occ_rew", "price_e", "carbon_e",
            "price_g", "carbon_g", "max_cost", "max_carbon", "reward_dt")


class ThermalRom:
    """LinearRom.step for a batch: x' = A x + B w, y' = C x' + D w, with the VAV air power solved implicitly."""

    def __init__(self, rom: LinearRom, ops: Ops):
        o = ops
        Z, V = rom.n_zones, rom.n_vavs
        self.o, self.V = o, V
        self.diag = rom.a.ndim == 1
        self.a = o.asarray(rom.a)
        self.b_q, self.b_p, self.b_a = o.asarray(rom.b[:, :Z]), o.asarray(rom.b[:, Z:Z + V]), o.asarray(rom.b[:, Z + V])
        self.c = o.asarray(rom.c)
        self.d_q, self.d_a = o.asarray(rom.d[:, :Z]), o.asarray(rom.d[:, Z + V])
        self.n_p = o.asarray(rom._np)  # (V + Z, V): output response to air power
        self.m = o.asarray(rom._m)  # (V, V): reduced Q^T A^-1 P
        self.eye = o.eye(V)
        self.t_ref = float(rom.t_ref)
        self.x0 = o.asarray(rom.x0)
        self.y0 = o.asarray(rom.extra["y0"])
        self.h = float(rom.extra.get("h", np.nan))

    def step(self, x, q_zone, t_amb, g, ts):
        da = (t_amb - self.t_ref)[:, None]
        ax = self.a * x if self.diag else x @ self.a.T
        x_pre = ax + q_zone @ self.b_q.T + da * self.b_a
        y_pre = x_pre @ self.c.T + q_zone @ self.d_q.T + da * self.d_a
        lhs = self.eye + g[:, :, None] * self.m
        p = self.o.solve(lhs, g * ((ts - self.t_ref) - y_pre[:, :self.V]))
        return x_pre + p @ self.b_p.T, y_pre + p @ self.n_p.T + self.t_ref, p


class BuildingEnv:
    def __init__(self, spec: BuildingSpec, rom: LinearRom, exo: Exogenous, num_envs: int = 1, ops: Ops | None = None,
                 exact: bool = True, emulate_f32: bool = True, episode_steps: int | None = None):
        o = ops or Ops("numpy")
        self.o, self.spec = o, spec
        self.plant = Plant(spec, o, exact=exact, emulate_f32=emulate_f32)
        self.rom = ThermalRom(rom, o)
        if np.isfinite(self.rom.h) and np.max(np.abs(exo.h - self.rom.h)) > 0.0:
            raise ValueError("the ROM was built for another convection coefficient than the weather gives")
        self.exo = {k: o.asarray(getattr(exo, k)) for k in EXO_STEP}
        self.obs_exo = o.asarray(exo.obs_exo)
        self.num_envs = num_envs
        self.num_obs = self.plant.n_obs
        self.num_actions = (len(spec.cont), len(spec.disc))
        self.episode_steps = min(episode_steps or exo.n_steps, exo.n_steps)
        self.k = 0
        self.state = None

    # ---- episode ----------------------------------------------------------------------------------------------
    def reset(self):
        o, B = self.o, self.num_envs
        ones = o.asarray(np.ones((B, 1)))
        self.state = {"x": ones * self.rom.x0, "y": ones * self.rom.y0,
                      "tank": o.full((B,), self.spec.tank0), "ret": o.full((B,), self.spec.ret0)}
        self.k = 0
        self.obs = self.plant.reset_obs(B, ones * self.obs_exo[0])
        return self.obs

    def step(self, a_cont, a_disc):
        o, k, s = self.o, self.k, self.state
        a_cont, a_disc = o.asarray(a_cont), o.asarray(a_disc)
        e = {name: v[k] for name, v in self.exo.items()}

        dev = self.plant.decode(a_cont, a_disc)
        ones = o.asarray(np.ones(self.num_envs))  # exogenous values are shared; broadcast them over the batch
        t_amb = ones * e["t_amb"]
        aux = self.plant.pre(s["y"], s["tank"], dev, t_amb, ones * e["comfort"], ones[:, None] * e["occ_step"])
        x_new, y_new, p_air = self.rom.step(s["x"], aux["q_zone"], t_amb, aux["g"], aux["ts"])
        out = self.plant.post(
            y_new, s["ret"], dev, aux, ones * e["t_amb_next"], ones[:, None] * e["win_now"],
            ones[:, None] * e["win_next"], ones[:, None] * e["occ_rew"], ones * e["price_e"], ones * e["carbon_e"],
            ones * e["price_g"], ones * e["carbon_g"], ones * e["max_cost"], ones * e["max_carbon"],
            ones * e["reward_dt"], ones[:, None] * self.obs_exo[k + 1])
        self.state = {"x": x_new, "y": y_new, "tank": out["tank"], "ret": out["ret"]}
        self.k = k + 1
        self.obs = out["obs"]
        done = self.k >= self.episode_steps
        return out["obs"], out["reward"], done, {"dev": dev, "pre": aux, "post": out, "y": y_new, "p_air": p_air}

    # ---- gradient windows (SHAC) --------------------------------------------------------------------------------
    def _detach(self, v):
        return v.detach().clone() if self.o.is_torch else np.array(v, copy=True)

    def get_checkpoint(self, detach: bool = False) -> dict:
        state = {k: (self._detach(v) if detach else v) for k, v in self.state.items()}
        return {"state": state, "k": self.k, "obs": self._detach(self.obs) if detach else self.obs}

    def load_checkpoint(self, checkpoint: dict) -> None:
        self.state = dict(checkpoint["state"])
        self.k = checkpoint["k"]
        self.obs = checkpoint["obs"]

    def clear_grad(self) -> None:
        """Cut the graph between the current state and everything before it."""
        self.state = {k: self._detach(v) for k, v in self.state.items()}
        self.obs = self._detach(self.obs)

    def initialize_trajectory(self):
        self.clear_grad()
        return self.obs
