"""The differentiable copy of one scenario: ROM (step B) + plant (plant.py), batched, on NumPy or PyTorch.

``BuildingEnv`` follows the batched interface SHAC-style learners use (DiffRL, rewarped): ``num_envs``, ``num_obs``,
``num_actions``, ``reset``, ``step(action) -> obs, reward, done, extras`` and ``clear_grad`` (cut the graph between
windows). The action is the competition's hybrid action: continuous values in [-1, 1] and discrete run commands in
{0, 1} (any value in [0, 1] when ``exact=False``).

One environment holds one building (one ROM). Its batch runs B copies of it, each with its own actions; the
exogenous series is shared and every copy sits at the same step. With the NumPy backend it is the validation
replica that env_check compares with the organisers' simulator.

``replica_from_env`` builds the three inputs (spec, ROM, exogenous series) from a freshly loaded environment.
"""

from __future__ import annotations

import copy
import time

import numpy as np

from hvac_controller.d1.core import core_from_env
from hvac_controller.d1.ops import Ops
from hvac_controller.d1.plant import Plant
from hvac_controller.d1.rom import LinearRom, build_rom
from hvac_controller.d1.spec import BuildingSpec, Exogenous

EXO_STEP = ("t_amb", "t_amb_next", "comfort", "win_now", "win_next", "occ_step", "occ_rew", "price_e", "carbon_e",
            "price_g", "carbon_g", "max_cost", "max_carbon", "reward_dt")


def replica_from_env(env, n_steps: int, log=None) -> tuple[BuildingSpec, LinearRom, Exogenous, np.ndarray, dict]:
    """Spec, ROM and exogenous series of a freshly loaded environment, which this resets.

    Returns them with the reset observation and the build timings. The occupancy object is copied before
    ``env.reset()`` asks it anything, so the exogenous occupancy replays the environment's own draws.
    """
    t0 = time.time()
    spec = BuildingSpec.from_env(env)
    occupancy = copy.deepcopy(env.building._occupancy)
    obs0 = env.reset()[0]
    t1 = time.time()
    rom, info = build_rom(core_from_env(env), log=log)
    t2 = time.time()
    exo = Exogenous.from_env(env, spec, n_steps, occupancy)
    timings = {"spec_s": t1 - t0, "rom_s": t2 - t1, "exo_s": time.time() - t2, "rom": info}
    return spec, rom, exo, obs0, timings


class ThermalRom:
    """LinearRom.step for a batch: x' = a * x + B w, y' = C x', with the VAV air power solved implicitly."""

    def __init__(self, rom: LinearRom, ops: Ops):
        o = ops
        Z, V = rom.n_zones, rom.n_vavs
        self.o, self.V = o, V
        self.a = o.asarray(rom.a)
        self.b_q, self.b_p, self.b_a = o.asarray(rom.b[:, :Z]), o.asarray(rom.b[:, Z:Z + V]), o.asarray(rom.b[:, Z + V])
        self.c = o.asarray(rom.c)
        self.n_p = o.asarray(rom._np)  # (V + Z, V): output response to air power
        self.m = o.asarray(rom._m)  # (V, V): reduced Q^T A^-1 P
        self.eye = o.eye(V)
        self.t_ref = float(rom.t_ref)
        self.x0 = o.asarray(rom.x0)
        self.y0 = o.asarray(rom.extra["y0"])
        self.h = float(rom.extra.get("h", np.nan))

    def step(self, x, q_zone, t_amb, g, ts):
        x_pre = self.a * x + q_zone @ self.b_q.T + (t_amb - self.t_ref)[:, None] * self.b_a
        y_pre = x_pre @ self.c.T
        lhs = self.eye + g[:, :, None] * self.m
        p = self.o.solve(lhs, g * ((ts - self.t_ref) - y_pre[:, :self.V]))
        return x_pre + p @ self.b_p.T, y_pre + p @ self.n_p.T + self.t_ref


class BuildingEnv:
    def __init__(self, spec: BuildingSpec, rom: LinearRom, exo: Exogenous, num_envs: int = 1, ops: Ops | None = None,
                 exact: bool = True, emulate_f32: bool = True, episode_steps: int | None = None,
                 compile_step: bool = False):
        o = ops or Ops("numpy")
        self.o, self.spec = o, spec
        self.plant = Plant(spec, o, exact=exact, emulate_f32=emulate_f32)
        self.rom = ThermalRom(rom, o)
        if np.isfinite(self.rom.h) and np.max(np.abs(exo.h - self.rom.h)) > 0.0:
            raise ValueError("the ROM was built for another convection coefficient than the weather gives")
        self.exo = {k: o.asarray(getattr(exo, k)) for k in EXO_STEP}
        self.obs_exo = o.asarray(exo.obs_exo)
        self.num_envs = num_envs
        self.ones = o.asarray(np.ones(num_envs))  # exogenous values are shared; broadcast them over the batch
        self.num_obs = self.plant.n_obs
        self.num_actions = (len(spec.cont), len(spec.disc))
        self.episode_steps = min(episode_steps or exo.n_steps, exo.n_steps)
        self.k = 0
        self.state = None
        if compile_step:  # torch only: the step body as one compiled graph (fuses the plant's elementwise ops)
            self._advance = o.torch.compile(self._advance)

    # ---- episode ----------------------------------------------------------------------------------------------
    def reset(self):
        o, B = self.o, self.num_envs
        ones = self.ones[:, None]
        self.state = {"x": ones * self.rom.x0, "y": ones * self.rom.y0,
                      "tank": o.full((B,), self.spec.tank0), "ret": o.full((B,), self.spec.ret0)}
        self.k = 0
        self.obs = self.plant.reset_obs(B, ones * self.obs_exo[0])
        return self.obs

    def step(self, a_cont, a_disc):
        o, k = self.o, self.k
        e = {name: v[k] for name, v in self.exo.items()}
        self.state, out = self._advance(self.state, o.asarray(a_cont), o.asarray(a_disc), e, self.obs_exo[k + 1])
        self.k = k + 1
        self.obs = out["obs"]
        return out["obs"], out["reward"], self.k >= self.episode_steps, {"post": out, "y": self.state["y"]}

    def _advance(self, s: dict, a_cont, a_disc, e: dict, obs_exo_next) -> tuple[dict, dict]:
        """One step as a pure function of arrays: state, actions and this step's exogenous row -> state', outputs."""
        dev = self.plant.decode(a_cont, a_disc)
        ones = self.ones
        t_amb = ones * e["t_amb"]
        aux = self.plant.pre(s["y"], s["tank"], dev, t_amb, ones * e["comfort"], ones[:, None] * e["occ_step"])
        x_new, y_new = self.rom.step(s["x"], aux["q_zone"], t_amb, aux["g"], aux["ts"])
        out = self.plant.post(
            y_new, s["ret"], dev, aux, ones * e["t_amb_next"], ones[:, None] * e["win_now"],
            ones[:, None] * e["win_next"], ones[:, None] * e["occ_rew"], ones * e["price_e"], ones * e["carbon_e"],
            ones * e["price_g"], ones * e["carbon_g"], ones * e["max_cost"], ones * e["max_carbon"],
            ones * e["reward_dt"], ones[:, None] * obs_exo_next)
        return {"x": x_new, "y": y_new, "tank": out["tank"], "ret": out["ret"]}, out

    def clear_grad(self) -> None:
        """Cut the graph between the current state and everything before it (start of a gradient window)."""
        if self.o.is_torch:
            self.state = {k: v.detach() for k, v in self.state.items()}
            self.obs = self.obs.detach()
