"""Direction 2: model predictive control on the replica (d1/torch_env.BuildingEnv) as the building model.

Every ``rp`` steps the planner searches the next ``h`` steps of supervisory decisions (lowlevel.DEC, one vector per
block; blocks are short near the present and longer further out) by the cross-entropy method: ``n`` candidate plans
are rolled out in a batched float32 NumPy copy of the model from the current state, each driven by the same low-level
controller the live building runs, and scored with the competition's own reward. The best plan seen is executed
until the next replan, which warm-starts from it shifted in time. The default plan (K10's schedule) is always a
candidate, so in the model the executed plan is never worse than K10's.

The model's state is tracked by a one-row float64 replica stepped with every action actually sent, from the episode's
initial state (the whole grid at 294 K). Where the model comes from (preset ``source``):

- ``oracle``: a replica cache (d1/cache.py) of the scenario itself: true materials, convection coefficient, weather and
  occupancy. Not a submission: it measures what planning can win when the model is right (D2 doc §3.6).
- ``meta``: rebuilt from the metadata (build.py) with prior materials and h; expected occupancy; outside temperature
  measured each step and forecast by persistence of the last day. The tracked state is corrected each step towards
  the measured zone temperatures (gain ``obs`` per mille), and occupants per zone are estimated from the measured
  building occupancy.
"""

from __future__ import annotations

import math
import re
import time

import numpy as np

from hvac_controller.d1.ops import Ops
from hvac_controller.d1.torch_env import BuildingEnv
from hvac_controller.d2.lowlevel import LowLevel

_BASE = {"n": 48, "it": 3, "el": 6, "h": 144, "rp": 12, "blocks": (12, 12, 12, 12, 24, 24, 48), "sigma": 0.2,
         "sigma_run": 0.35, "min_sigma": 0.03, "seed": 0, "threads": 1, "obs": 100, "ridge": 100, "occ": 150,
         "ep": 8640, "lag": 0, "cw": 100, "guard": 0, "pa": 0, "hs": 0, "db": 0}
PRESETS = {
    # 12 h ahead in blocks of 1, 1, 1, 1, 2, 2, 4 h; replan every hour.
    "oracle": dict(_BASE, source="oracle"),
    "meta": dict(_BASE, source="meta"),
    # Diagnostic: how far planning alone can go. The true model, decisions per AHU, a day ahead, a large search.
    "heavy": dict(_BASE, source="oracle", pa=1, n=128, it=5, el=12, h=288, blocks=(12, 12, 12, 12, 24, 24, 48, 48, 96)),
}
# Integer overrides: obs = observer gain per mille, ridge = its regularisation per mille of the mean sensor gain,
# occ = prior occupants per zone x 100, ep = episode steps, lag = steps between a plan's request and its use (0:
# plan synchronously; > 0: on the planner thread, overlapping the simulator), cw = weight of the comfort term in
# the planner's objective x 100 (above 100: plans keep a margin against model and forecast error), guard = comfort
# guard margin in mK (0: off; see MPCPolicy._guard), pa = decisions per AHU (1) or one for all (0), hs = sequenced
# heating in the low-level controller (lowlevel.LowLevel, heat_seq), db = its heating/cooling deadband: heat to the
# band floor + db mK (0: one PI loop, as K10).
SHORT = ("n", "it", "el", "h", "rp", "seed", "threads", "obs", "ridge", "occ", "ep", "lag", "cw", "guard", "pa", "hs", "db")
OCC_NORM = 125.0  # evaluate.py / sim.load_env: the observation's occupancy normalisation constant


def parse_config(name: str) -> dict:
    """``oracle`` or ``meta.n64.it4.h288``: a preset with integer overrides (keys in SHORT)."""
    preset, *overrides = name.split(".")
    if preset not in PRESETS:
        raise ValueError(f"unknown MPC preset {preset!r}: {sorted(PRESETS)}")
    cfg = dict(PRESETS[preset], preset=preset)
    for o in overrides:
        m = re.fullmatch(r"([a-z]+)(\d+)", o)
        if not m or m.group(1) not in SHORT:
            raise ValueError(f"bad override {o!r} in {name!r}")
        cfg[m.group(1)] = int(m.group(2))
    if cfg["h"] != sum(cfg["blocks"]):  # stretch or cut the block pattern to the horizon
        blocks, left = [], cfg["h"]
        for b in cfg["blocks"] * math.ceil(cfg["h"] / sum(cfg["blocks"])):
            if left <= 0:
                break
            blocks.append(min(b, left))
            left -= blocks[-1]
        cfg["blocks"] = tuple(blocks)
    return cfg


def _active(comfort: np.ndarray, lead: int = 12) -> np.ndarray:
    """Comfort mode on now or within the next hour, per step (what K10 runs its units in)."""
    out = comfort.astype(float).copy()
    for j in range(1, lead + 1):
        out[:-j] = np.maximum(out[:-j], comfort[j:])
    return out


def _set_exo(env: BuildingEnv, name: str, k0: int, values) -> None:
    """Overwrite an exogenous column of a one-scenario model from step k0 on (values clipped to the series)."""
    values = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    n = min(len(values), env.n_steps - k0)
    if n <= 0:
        return
    col = env.exo_cols[name]
    block = env.exo[k0:k0 + n]
    v = env.o.asarray(values[:n] if isinstance(col, slice) else values[:n, 0])
    block[:, col] = v


class MPCPolicy:
    def __init__(self, meta: dict, action_spec, cfg: dict, spec, rom, exo=None, model_ctx=None):
        """``exo``: the scenario's own series (oracle). Without it (meta), the series is built from ``model_ctx``'s
        schedule when the first observation gives the time of day and day of week."""
        self.cfg, self.meta, self.action_spec = cfg, meta, action_spec
        self.spec, self.rom, self.model_ctx = spec, rom, model_ctx
        self.ll = LowLevel(meta, per_ahu=bool(cfg["pa"]), heat_seq=bool(cfg["hs"]), deadband=cfg["db"] / 1000.0)
        self.field_names = list(meta["observation_field_names"])
        lay = self.ll.lay
        self.i_oat, self.c_oat = lay.i_temps[0], lay.c_temps[0]
        self.i_occ = self.field_names.index("num_occupants")
        self.occ_per_zone = cfg["occ"] / 100.0
        self.integral = np.zeros((1, self.ll.n_int), dtype=np.float32)
        self.rng = np.random.default_rng(cfg["seed"])
        self.k, self.plan, self.plan_k = 0, None, 0
        self._pending, self._requested, self._executor, self._model_occ_scale = None, 0, None, 1.0
        self.oat: list[float] = []
        self.stats = {"replans": 0, "plan_s": 0.0, "gain": [], "obs_err": []}
        self.track = None
        if exo is not None:
            self._init_models(exo)

    def _init_models(self, exo) -> None:
        self.exo = exo
        self.n_steps = exo.n_steps
        self.track = BuildingEnv(self.spec, self.rom, exo, 1, Ops("numpy"))
        self.track.reset()
        self.model = BuildingEnv(self.spec, self.rom, exo, self.cfg["n"], Ops("numpy", dtype=np.float32),
                                 emulate_f32=False)
        self.model.reset()
        self.active = _active(np.asarray(exo.comfort))
        # Observer: a ridge-regularised state change towards the measured sensed temperatures. An exact (least-norm)
        # one is unstable: it builds large components that cancel on the sensors now but not after one step of
        # modes with different decays.
        c = np.asarray(self.track.rom.c_t[0], dtype=np.float64)  # (r, V + Z)
        cs = c[:, :self.spec.n_vavs]
        gram = cs.T @ cs
        ridge = self.cfg["ridge"] / 1000.0 * np.trace(gram) / len(gram)
        self.obs_x = np.linalg.solve(gram + ridge * np.eye(len(gram)), cs.T)  # (V, r)
        self.obs_y = self.obs_x @ c  # (V, V + Z)

    # ---- online model updates (meta) -------------------------------------------------------------------------
    def _measure(self, obs: np.ndarray) -> float:
        oat = float(obs[self.i_oat] * self.c_oat[1] + self.c_oat[0])
        self.oat.append(oat)
        return oat

    def _start(self, obs: np.ndarray) -> None:
        from hvac_controller.d2.build import exogenous, start_from_obs

        start = start_from_obs(obs, self.field_names)
        n = self.cfg["ep"] + self.cfg["h"] + 2
        self._init_models(exogenous(self.model_ctx, self.spec, start, n, self.occ_per_zone, self.oat[-1]))
        self.presence = np.asarray(self.exo.occ_rew[:, 0]) / self.occ_per_zone  # expected share present

    def _correct(self, obs: np.ndarray) -> None:
        """Pull the tracked state towards the measured zone temperatures (both lag one step the same way)."""
        lay = self.ll.lay
        measured = obs[lay.i_temp] * lay.c_temp[:, 1] + lay.c_temp[:, 0]
        predicted = self.track.obs[0][lay.i_temp] * lay.c_temp[:, 1] + lay.c_temp[:, 0]
        err = measured - predicted
        self.stats["obs_err"].append(float(np.abs(err).mean()))
        g = self.cfg["obs"] / 1000.0
        if g > 0.0:
            self.track.state["x"] = self.track.state["x"] + g * err[None] @ self.obs_x
            self.track.state["y"] = self.track.state["y"] + g * err[None] @ self.obs_y

    def _update_occupancy(self, obs: np.ndarray) -> None:
        """Occupants per zone from the measured count, once enough people are expected to be in."""
        share = self.presence[max(self.k - 1, 0)]
        if share < 0.5:
            return
        v = float(obs[self.i_occ])
        count = v if v >= 0.0 else v * (OCC_NORM + 1.0) + OCC_NORM  # the environment reports the raw count
        est = count / (share * self.spec.n_zones)
        new = float(np.clip(0.8 * self.occ_per_zone + 0.2 * est, 0.5, 3.0))
        scale = new / self.occ_per_zone
        if abs(scale - 1.0) > 1e-3:
            for name in ("occ_step", "occ_rew"):
                col = self.track.exo_cols[name]
                self.track.exo[self.k:, col] = self.track.exo[self.k:, col] * scale
            self._model_occ_scale *= scale  # the planner's model takes it with the next snapshot
            self.occ_per_zone = new

    def _forecast(self, H: int) -> np.ndarray:
        """Outside temperature for the next H steps: yesterday's curve, offset to now, the offset fading over 6 h."""
        now, hist = self.oat[-1], self.oat
        if len(hist) <= 288:
            return np.full(H, now)
        j = np.arange(H)
        k = len(hist) - 1
        base = np.array([hist[k - 288 + i] if k - 288 + i < k else hist[k - 288 + i - 288] for i in j])
        return base + (now - hist[k - 288]) * np.exp(-j / 72.0)

    # ---- planning --------------------------------------------------------------------------------------------
    def _lengths(self, H: int) -> list[int]:
        out, left = [], H
        for b in self.cfg["blocks"]:
            if left <= 0:
                break
            out.append(min(b, left))
            left -= out[-1]
        return out

    @staticmethod
    def _to_blocks(steps: np.ndarray, lengths: list[int]) -> np.ndarray:
        edges = np.cumsum([0] + lengths)
        return np.stack([steps[a:b].mean(0) for a, b in zip(edges[:-1], edges[1:])])

    def _rollout(self, cand: np.ndarray, lengths: list[int], snap: dict) -> np.ndarray:
        """Total model reward of each candidate (N, blocks, 6) from the snapshot's state."""
        m, N = self.model, cand.shape[0]
        m.state = {name: np.repeat(np.asarray(v, dtype=np.float32), N, axis=0) for name, v in snap["state"].items()}
        m.k = np.full(N, snap["k"], dtype=np.int64)
        m.obs = np.repeat(snap["obs"], N, axis=0)
        integral = np.repeat(snap["integral"], N, axis=0)
        steps = np.repeat(cand, lengths, axis=1).astype(np.float32)  # (N, H, 6)
        extra = (self.cfg["cw"] / 100.0 - 1.0) * self.spec.reward["w_prod"]
        total = np.zeros(N)
        for t in range(steps.shape[1]):
            a_c, a_d, integral = self.ll.act(m.obs, integral, steps[:, t])
            _, reward, _, info = m.step(a_c, a_d)
            total += reward
            if extra:
                total += extra * info["post"]["regret"]
        return total

    def _snapshot(self, obs: np.ndarray) -> dict:
        """Everything a plan starts from, copied, so the planner can run while the live building moves on."""
        H = min(self.cfg["h"], self.n_steps - self.k)
        snap = {"k": self.k, "H": H, "obs": obs.copy(), "integral": self.integral.copy(),
                "state": {name: np.array(v, copy=True) for name, v in self.track.state.items()},
                "plan": self.plan, "plan_k": self.plan_k, "occ_scale": self._model_occ_scale}
        self._model_occ_scale = 1.0
        if self.cfg["source"] == "meta":
            snap["forecast"] = self._forecast(H + 1)
        return snap

    def _replan(self, obs: np.ndarray) -> None:
        """Plan now and execute it from now on."""
        self.plan, self.plan_k = self._plan(self._snapshot(obs))

    def _plan(self, snap: dict) -> tuple[np.ndarray, int]:
        """The best plan (per step, from snap['k']) the cross-entropy search finds in the model. Touches only the
        model, the random generator and the stats, so it may run on the planner thread."""
        cfg, t0, k, H = self.cfg, time.perf_counter(), snap["k"], snap["H"]
        if snap["occ_scale"] != 1.0:
            for name in ("occ_step", "occ_rew"):
                col = self.model.exo_cols[name]
                self.model.exo[k:, col] = self.model.exo[k:, col] * snap["occ_scale"]
        if "forecast" in snap:
            f = snap["forecast"]
            _set_exo(self.model, "t_amb", k, f[:H])
            _set_exo(self.model, "t_amb_next", k, f[1:H + 1])
        lengths = self._lengths(H)
        default = self.ll.default_decision(self.active[k:k + H])
        if self.active[k:k + H].max() < 0.5:  # nobody due within the horizon (weekend, holiday): all off, no search
            self.stats["skipped"] = self.stats.get("skipped", 0) + 1
            return default, k
        warm = default.copy()
        if snap["plan"] is not None:  # the last plan, shifted to now, then the default
            rest = snap["plan"][k - snap["plan_k"]:][:H]
            warm[:len(rest)] = rest
        mean = self._to_blocks(warm, lengths)
        fixed = [mean.copy(), self._to_blocks(default, lengths)]
        sigma = np.full_like(mean, cfg["sigma"])
        sigma[:, self.ll.cols["run"]] = cfg["sigma_run"]
        best, best_r, default_r = None, -math.inf, None
        for _ in range(cfg["it"]):
            cand = np.clip(mean + sigma * self.rng.standard_normal((cfg["n"],) + mean.shape), 0.0, 1.0)
            for j, f in enumerate(fixed + ([best] if best is not None else [])):
                cand[j] = f
            r = np.nan_to_num(self._rollout(cand, lengths, snap), nan=-1e9, posinf=-1e9, neginf=-1e9)
            if default_r is None:
                default_r = float(r[1])
            i = int(np.argmax(r))
            if r[i] > best_r:
                best, best_r = cand[i].copy(), float(r[i])
            elite = cand[np.argsort(r)[-cfg["el"]:]]
            mean, sigma = elite.mean(0), np.maximum(elite.std(0, ddof=1), cfg["min_sigma"])
        self.stats["replans"] += 1
        self.stats["plan_s"] += time.perf_counter() - t0
        self.stats["gain"].append((best_r - default_r) / H)  # model reward per step over the default plan
        return np.repeat(best, lengths, axis=0), k

    # ---- acting ----------------------------------------------------------------------------------------------
    def _schedule_plans(self, o: np.ndarray) -> None:
        """Replan every rp steps. With ``lag`` > 0 a plan is searched on the planner thread from the state at the
        request and put in force exactly ``lag`` steps later (waiting for it if needed), while the live building
        keeps executing the previous plan: the search overlaps the simulator's own steps, and the outcome does not
        depend on timing."""
        lag = self.cfg["lag"]
        if self.plan is None or lag == 0:
            if self.plan is None or self.k - self.plan_k >= self.cfg["rp"]:
                self._replan(o)
            return
        if self._pending is not None and self.k >= self._pending[0]:
            self.plan, self.plan_k = self._pending[1].result()
            self._pending = None
        if self._pending is None and self.k - self._requested >= self.cfg["rp"]:
            if self._executor is None:
                from concurrent.futures import ThreadPoolExecutor

                self._executor = ThreadPoolExecutor(1, thread_name_prefix="d2-planner")
            self._requested = self.k
            self._pending = (self.k + lag, self._executor.submit(self._plan, self._snapshot(o)))

    def _guard(self, obs: np.ndarray, dec: np.ndarray) -> np.ndarray:
        """While people are in, a zone closer than ``guard`` mK to a comfort limit gets the plant now rather than at
        the next replan: units on, the band centre as target, and at least K10's cooling (supply 288 K) when a zone
        is warm or its hot water (330 K) when one is cold. Comfort is 60 % of the score and unbounded; a plan is only
        as good as the model and the forecast behind it."""
        margin = self.cfg["guard"] / 1000.0
        if margin <= 0.0 or float(obs[self.i_occ]) <= 0.5:  # the raw occupant count
            return dec
        temps = self.ll.zone_temps(obs[None])[0]
        lo, hi = self.ll.band
        warm, cold = bool((temps > hi - margin).any()), bool((temps < lo + margin).any())
        if not (warm or cold):
            return dec
        self.stats["guard"] = self.stats.get("guard", 0) + 1
        safe = self.ll.default_decision(np.ones(1))[0]
        c = self.ll.cols
        out = dec.copy()
        out[c["run"]], out[c["target"]] = 1.0, 0.5
        if warm:
            out[c["sat"]] = np.minimum(out[c["sat"]], safe[c["sat"]])
        if cold:
            out[c["water"]] = np.maximum(out[c["water"]], safe[c["water"]])
        return out

    def get_action(self, obs) -> dict:
        obs = np.asarray(obs, dtype=np.float32)
        live = self.track is None or self.k < self.n_steps
        if self.cfg["source"] == "meta" and live:
            oat = self._measure(obs)
            if self.track is None:
                self._start(obs)
            elif self.k > 0:
                self._correct(obs)
                self._update_occupancy(obs)
            _set_exo(self.track, "t_amb", self.k, [oat])
            _set_exo(self.track, "t_amb_next", self.k, [oat])
        o = obs[None]
        if live:
            self._schedule_plans(o)
        i = min(self.k - self.plan_k, len(self.plan) - 1)
        a_c, a_d, self.integral = self.ll.act(o, self.integral, self._guard(obs, self.plan[i])[None])
        a_c, a_d = a_c[0].astype(np.float32), a_d[0].round().astype(np.int32)
        if live:  # the gateway may ask once more after the last scored step
            self.track.step(a_c[None].astype(np.float64), a_d[None].astype(np.float64))
        self.k += 1
        return {"continuous_action": a_c, "discrete_action": a_d}


def make_mpc(name: str, meta: dict, action_spec, cache: str | None) -> MPCPolicy:
    """``name`` as parse_config reads it; the oracle takes the scenario's own replica cache."""
    cfg = parse_config(name)
    if cfg["source"] == "oracle":
        if not cache:
            raise ValueError("the oracle MPC needs the scenario's replica cache (--cache-dir)")
        from pathlib import Path

        from hvac_controller.d1.cache import load_replica

        rep = load_replica(Path(cache))
        return MPCPolicy(meta, action_spec, cfg, rep["spec"], rep["rom"], exo=rep["exo"])
    from hvac_controller.d2.build import model_from_meta

    spec, rom, ctx = model_from_meta(meta)
    return MPCPolicy(meta, action_spec, cfg, spec, rom, model_ctx=ctx)
