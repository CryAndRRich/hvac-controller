"""Direction 1: SHAC through the replica (Xu et al., ICLR 2022), re-implemented from the paper.

The training scenarios are split by building size into ``buckets``. Each bucket is one environment holding the
scenarios of several buildings, padded to the largest, with as many rows per scenario as
``row_budget >= rows x seg x VAVs`` allows; the rows are spread over the episode by a burn-in. Every iteration, for
each bucket:

1. a window of h steps from where the rows stopped. The policy acts, the replica steps, and the actor loss is
   -(1 / (B h)) sum over rows of [sum_t gamma^t r_t + gamma^h V'(s_h)], with V' the target critic. A row whose
   episode ends inside the window adds its finished return (a real end: V = 0) and restarts from its scenario's
   initial state, as the gateway's episodes do. The window runs as checkpointed segments of ``seg`` steps
   (recomputed in the backward pass), so memory grows with seg, not h. One backward pass per bucket; gradients
   accumulate, each bucket weighted by its share of the scenarios;
2. one actor step. Adam (0.7, 0.95), with the learning rate decaying linearly over the time budget, and the norm
   clipped at ``clip``. On top of that, two guards against the heavy-tailed first-order gradients of steep transitions
   (Onoda et al., ICLR 2026, §3). Per step, each row's action gradient is winsorised at ``winsor`` x the median over
   the bucket's rows. An update whose norm exceeds ``spike`` x the median of the last 50 is skipped: V1's spikes
   (5e4 against 0.005) passed a clip at 1.0 as steps 200 x normal and froze Adam through its second moment. The
   actor's exponential average (``ema``) is the policy scored and kept;
3. the critic fitted to TD(lambda) targets of the window on ``critic_rows`` fixed rows per bucket, 16 passes x 4
   shuffled minibatches, then the target critic follows it (Polyak).

The critic is a DeepSets over VAVs with privileged global inputs: the outside temperature and comfort flags of the
next 24 hours.

Scored in the replica every ``eval_minutes`` on the held-out scenarios (``--val-caches``), every scenario from its
initial state for the whole episode. The training scenarios are scored at the end. The best average, of the
held-out scenarios when there are any, is kept: ``d1_policy.pt``, the file policy.KagglePolicy loads. Other guards:
- a non-finite gradient skips the step and halves the learning rate (three in a row stop the run), and the rows
  restart;
- five skipped spikes in a row, a held-out score 15 % worse than the best on two evaluations in a row, or a moving
  average of the training reward 25 % worse than its best restore the saved best and halve the learning rate;
- the time budget ends the run cleanly; ``last.pt`` is written every 15 minutes and ``--resume`` continues from it;
- out of GPU memory before the first update restarts the process with half the ``row_budget`` (a fresh process: the
  failed attempt's memory is not reliably freed in place); after it, the run stops with the policy kept so far (a
  restart would overrun the session).
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from hvac_controller.d1.cache import load_replica
from hvac_controller.d1.evaluate import SIGMA, obs_noise_mask
from hvac_controller.d1.ops import Ops
from hvac_controller.d1.policy import Controller, Layout, _mlp, save_policy, stack_layouts
from hvac_controller.d1.torch_env import BuildingEnv

AHEAD = 24  # hours of outside temperature and comfort flags the critic sees
CRITIC_CHUNK = 1 << 19  # rows x VAVs per critic pass: its (rows, VAVs, width) activations dominate its memory


def _chunk(x_vav) -> int:
    """Rows per critic pass over (rows, VAVs, features) inputs."""
    return max(1, CRITIC_CHUNK // x_vav.shape[1])


@dataclass
class Config:
    h: int = 288
    gamma: float = 0.998
    lam: float = 0.95
    seg: int = 12  # checkpointed segment of the window
    row_budget: int = 1_435_000  # rows x seg x (padded) VAVs per bucket: ~8 GB on a T4 (2.87M ran out at 14.4 GB)
    buckets: int = 3
    minutes: float = 30.0  # training time budget
    lr_actor: float = 2e-3
    lr_critic: float = 5e-4
    clip: float = 1.0
    winsor: float = 10.0  # per-step cap of a row's action gradient, x the median over rows (0: off)
    spike: float = 10.0  # skip an update whose gradient norm exceeds this x the median of the last 50 (0: off)
    ema: float = 0.98  # decay of the actor's average, the policy scored and kept
    critic_iters: int = 16
    critic_batches: int = 4
    critic_rows: int = 256  # rows per bucket whose states the critic learns from
    target_keep: float = 0.2  # target <- keep * target + (1 - keep) * critic
    detach_obs: bool = False  # D.VA's decoupled gradient: no gradient through the observations
    memory: bool = True  # the policy's weather and zone-temperature memory
    eval_minutes: float = 20.0
    eval_seeds: int = 1
    eval_steps: int = 0  # 0: the whole episode
    width: int = 64
    seed: int = 0


def td_lambda(r, done, v_next, gamma: float, lam: float):
    """TD(lambda) returns of a window; r, done, v_next (h, B), v_next = V'(s_{t+1}) and 0 where the episode ended.
    G_{h-1} = r + gamma v; G_t = r_t + gamma ((1 - lam) v_{t+1} + lam G_{t+1}), or r_t + gamma v_{t+1} at an end."""
    h = r.shape[0]
    G = torch.empty_like(r)
    G[h - 1] = r[h - 1] + gamma * v_next[h - 1]
    for t in range(h - 2, -1, -1):
        G[t] = r[t] + gamma * torch.where(done[t], v_next[t], (1.0 - lam) * v_next[t] + lam * G[t + 1])
    return G


class Critic(nn.Module):
    """V(s) = mean over (real) VAVs of phi(VAV features, global) + psi(global), global with the next day's forecast."""

    def __init__(self, width: int, n_vav: int, n_glob: int):
        super().__init__()
        n_g = n_glob + 2 * AHEAD
        self.phi = _mlp(n_vav + n_g, width, 1)
        self.psi = _mlp(n_g, width, 1)

    def forward(self, vav, glob, v_mask):
        B, V = vav.shape[:2]
        x = torch.cat([vav, glob[:, None, :].expand(B, V, glob.shape[1])], 2)
        return (self.phi(x)[..., 0] * v_mask).sum(1) / v_mask.sum(1) + self.psi(glob)[..., 0]


class _Winsor(torch.autograd.Function):
    """Identity forward; backward caps each row's gradient norm at k x the median row norm (counts in stats)."""

    @staticmethod
    def forward(ctx, a, k: float, stats):
        ctx.k, ctx.stats = k, stats
        return a.clone()

    @staticmethod
    def backward(ctx, g):
        n = g.norm(dim=1)
        med = n.median()
        cap = torch.where(med > 0.0, ctx.k * med, torch.full_like(med, math.inf))
        scale = torch.clamp(cap / n.clamp(min=1e-30), max=1.0)
        ctx.stats.add_(torch.stack([(scale < 1.0).float().sum(), torch.ones_like(n).sum()]))
        return g * scale[:, None], None, None


class Bucket:
    """Scenarios of one or several buildings as one padded environment, their layouts and start states."""

    def __init__(self, reps: list[dict], row_budget: int, seg: int, device: str, rows_per: int | None = None):
        self.reps, self.S, self.device = reps, len(reps), device
        self.labels = [r["info"]["label"] for r in reps]
        self.layouts = [Layout(r["meta"]) for r in reps]
        for lay, r in zip(self.layouts, reps):
            lay.check(r["spec"])  # every scenario's metadata against its own simulator objects
        self.V = max(l.V for l in self.layouts)
        self.rows = self.S * (rows_per or max(4, row_budget // (seg * self.V * self.S)))
        self.env = BuildingEnv([r["spec"] for r in reps], [r["rom"] for r in reps], [r["exo"] for r in reps], self.rows,
                               Ops("torch", dtype=torch.float32, device=device), exact=False, emulate_f32=False)
        self.lay = self.layout(self.rows)
        e = self.env
        self.fc_t = e.exo[:, e.exo_cols["t_amb"]]
        self.fc_c = e.exo[:, e.exo_cols["comfort"]]
        self.ahead = torch.arange(AHEAD, device=device) * int(round(3600 / reps[0]["spec"].dt))
        n_obs = e.num_obs
        self.mask = torch.as_tensor(np.stack([np.pad(obs_noise_mask(r["spec"]), (0, n_obs - len(r["spec"].obs)))
                                              for r in reps]), dtype=torch.float32, device=device)  # (S, n_obs)
        self.critic_rows = None  # set by the trainer
        self.start = None

    def make_env(self, rows: int) -> BuildingEnv:
        """Another environment of these scenarios with its own rows (burn-in, evaluation), sharing the tensors."""
        return self.env.resized(rows)

    def layout(self, rows: int) -> dict:
        return stack_layouts(self.layouts, np.arange(rows) // (rows // self.S), self.device)

    def forecast(self, n_steps: int, row_s, k):
        idx = row_s[:, None] * n_steps + torch.clamp(k[:, None] + self.ahead, max=n_steps - 1)
        return torch.cat([(self.fc_t[idx] - 295.0) / 10.0, self.fc_c[idx]], 1)


class Trainer:
    def __init__(self, cfg: Config, buckets: list[Bucket], val: list[Bucket], device: str, out: Path, log=print):
        self.cfg, self.buckets, self.val, self.device, self.out, self.log = cfg, buckets, val, device, out, log
        torch.manual_seed(cfg.seed)
        specs = [r["spec"] for b in buckets + val for r in b.reps]
        self.ctrl = Controller(width=cfg.width, p_floor=float(max(np.max(s.ahu["min_pressure"]) for s in specs)),
                               max_recirc=float(min(np.min(s.ahu["max_recirc"]) for s in specs)),
                               min_damper=float(max(np.max(s.vav["min_damper"]) for s in specs)),
                               memory=cfg.memory).to(device)
        self.ema = copy.deepcopy(self.ctrl).requires_grad_(False)
        self.critic = Critic(cfg.width, self.ctrl.n_vav, self.ctrl.n_glob).to(device)
        self.target = Critic(cfg.width, self.ctrl.n_vav, self.ctrl.n_glob).to(device)
        self.target.load_state_dict(self.critic.state_dict())
        self.target.requires_grad_(False)
        self.opt_a = torch.optim.Adam(self.ctrl.parameters(), lr=cfg.lr_actor, betas=(0.7, 0.95))
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic, betas=(0.7, 0.95))
        self.iteration, self.lr_scale, self.best = 0, 1.0, math.inf
        self.minutes_done = 0.0  # training time of earlier sessions (resume)
        self.norms = collections.deque(maxlen=50)  # accepted gradient norms (spike guard)
        self.wstats = torch.zeros(2, device=device)  # winsorised rows, rows (since the last log)
        self.best_snap, self.bad_evals = None, 0
        for b in buckets:
            b.critic_rows = torch.arange(0, b.rows, max(1, b.rows // cfg.critic_rows), device=device)[:cfg.critic_rows]

    # ---- features ---------------------------------------------------------------------------------------------
    def critic_inputs(self, b: Bucket, env, obs, mem, rows, lay):
        """Critic features of the given rows (all when rows is None); obs, mem and lay are already those rows'."""
        vav, glob, *_ = self.ctrl.features(obs, mem, lay)
        k = env.k if rows is None else env.k[rows]
        s = env.row_s if rows is None else env.row_s[rows]
        return vav, torch.cat([glob, b.forecast(env.n_steps, s, k)], 1)

    @staticmethod
    def _rows(lay: dict, rows) -> dict:
        return {k: (v[rows] if torch.is_tensor(v) and v.shape[0] > 1 else v) for k, v in lay.items()}

    # ---- start states -----------------------------------------------------------------------------------------
    @torch.no_grad()
    def burn_in(self, b: Bucket) -> None:
        """One row per scenario for the whole episode with the current policy; row j of scenario s starts at the
        state that run reached at offset j * n / (rows per scenario)."""
        env, lay = b.make_env(b.S), b.layout(b.S)
        obs = env.reset()
        mem = Controller.init_mem(b.S, lay, self.device)
        per = b.rows // b.S
        offsets = (np.arange(per) * env.n_steps) // per
        slot = {int(o): u for u, o in enumerate(np.unique(offsets))}
        snaps = []
        for k in range(env.n_steps):
            if k in slot:
                snaps.append({**{n: v.clone() for n, v in env.state.items()}, "obs": obs.clone(),
                              **{f"mem{j}": m.clone() for j, m in enumerate(mem)}})
            a_c, a_d, mem = self.ctrl.act(obs, mem, lay)
            obs, _, _, _ = env.step(a_c, a_d)
        rows = torch.arange(b.rows, device=self.device)
        s_of = rows // per
        u_of = torch.as_tensor([slot[int(o)] for o in offsets], device=self.device)[rows % per]
        stacked = {n: torch.stack([sn[n] for sn in snaps]) for n in snaps[0]}  # (n_offsets, S, ...)
        start = {n: v[u_of, s_of] for n, v in stacked.items()}
        b.start = {"state": {n: start[n] for n in env.state}, "obs": start["obs"],
                   "mem": tuple(start[f"mem{j}"] for j in range(len(mem))),
                   "k": torch.as_tensor(offsets, device=self.device)[rows % per]}
        del env

    # ---- one window ---------------------------------------------------------------------------------------------
    def _segment(self, b: Bucket, t0: int, t1: int, x, y, tank, ret, k, obs, I, m_v, m_g, gamma_k, acc):
        """Steps t0 .. t1 - 1 of the window as a pure function of the carried tensors (checkpointed)."""
        cfg, env, ctrl, sel = self.cfg, b.env, self.ctrl, b.critic_rows
        env.state, env.k, env.obs = {"x": x, "y": y, "tank": tank, "ret": ret}, k, obs
        mem = (I, m_v, m_g)
        lay_sel = self._rows(b.lay, sel)
        loss = torch.zeros((), device=self.device)
        reward = torch.zeros((), device=self.device)
        on = torch.zeros((), device=self.device)
        buf = {"vav": [], "glob": [], "r": [], "done": []}
        d_mask = b.lay["d_mask"]
        for t in range(t0, t1):
            with torch.no_grad():
                cv, cg = self.critic_inputs(b, env, env.obs[sel], tuple(m[sel] for m in mem), sel, lay_sel)
            buf["vav"].append(cv)
            buf["glob"].append(cg)
            o = env.obs.detach() if cfg.detach_obs else env.obs
            a_c, a_d, mem = ctrl.act(o, mem, b.lay)
            if cfg.winsor > 0.0:
                a_c, a_d = _Winsor.apply(a_c, cfg.winsor, self.wstats), _Winsor.apply(a_d, cfg.winsor, self.wstats)
            _, r, done, _ = env.step(a_c, a_d)
            acc = acc + gamma_k * r
            reward = reward + r.detach().mean()
            on = on + ((a_d.detach() * d_mask).sum(1) / d_mask.sum(1)).mean()
            buf["r"].append(r[sel].detach())
            buf["done"].append(done[sel])
            if t < cfg.h - 1:
                loss = loss - torch.where(done, acc, torch.zeros_like(acc)).sum()
                gamma_k = torch.where(done, torch.ones_like(gamma_k), gamma_k * cfg.gamma)
                acc = torch.where(done, torch.zeros_like(acc), acc)
                env.reset_done(done)
                mem = Controller.reset_mem(mem, done)
        s = env.state
        return (s["x"], s["y"], s["tank"], s["ret"], env.k, env.obs, *mem, gamma_k, acc, loss,
                *(torch.stack(v) for v in buf.values()), reward, on)

    def window(self, b: Bucket, weight: float) -> dict:
        """Rollout from b.start, actor loss, backward (gradients accumulate). Returns the loss, the end state and the
        critic's data (states s_0 .. s_h, rewards and episode ends of the critic's rows)."""
        cfg, env = self.cfg, b.env
        st = b.start
        B = env.num_envs
        carry = (st["state"]["x"], st["state"]["y"], st["state"]["tank"], st["state"]["ret"], st["k"], st["obs"],
                 *st["mem"], torch.ones(B, device=self.device), torch.zeros(B, device=self.device))
        loss = torch.zeros((), device=self.device)
        reward = torch.zeros((), device=self.device)
        on = torch.zeros((), device=self.device)
        bufs = []
        for t0 in range(0, cfg.h, cfg.seg):
            args = (b, t0, min(t0 + cfg.seg, cfg.h), *carry)
            out = checkpoint(self._segment, *args, use_reentrant=False) if cfg.seg < cfg.h else self._segment(*args)
            carry = out[:11]
            loss = loss + out[11]
            bufs.append(out[12:16])
            reward, on = reward + out[16], on + out[17]
        x, y, tank, ret, k, obs, I, m_v, m_g, gamma_k, acc = carry
        env.state, env.k, env.obs = {"x": x, "y": y, "tank": tank, "ret": ret}, k, obs
        mem = (I, m_v, m_g)
        # The window's end: bootstrap with the target critic (0 where the episode just ended), then restart those rows.
        done = k >= env.n_steps
        cv, cg = self.critic_inputs(b, env, obs, mem, None, b.lay)
        v = torch.where(done, torch.zeros_like(acc), self.target(cv, cg, b.lay["v_mask"]))
        loss = loss - (acc + cfg.gamma * gamma_k * v).sum()
        sel = b.critic_rows
        env.reset_done(done)
        mem = Controller.reset_mem(mem, done)
        # Taken before the backward pass, whose recomputation of the segments rewrites env.state.
        end = {"state": {n: v.detach() for n, v in env.state.items()}, "obs": env.obs.detach(),
               "mem": tuple(m.detach() for m in mem), "k": env.k}
        data = {"vav": torch.cat([torch.cat([u[0] for u in bufs]), cv[sel].detach()[None]]),
                "glob": torch.cat([torch.cat([u[1] for u in bufs]), cg[sel].detach()[None]]),
                "r": torch.cat([u[2] for u in bufs]), "done": torch.cat([u[3] for u in bufs]),
                "mask": b.lay["v_mask"][sel] if b.lay["v_mask"].shape[0] > 1 else b.lay["v_mask"]}
        loss = loss / (B * cfg.h)
        (loss * weight).backward()
        return {**data, "loss": loss.detach(), "reward": reward / cfg.h, "on": on / cfg.h, "end": end}

    def _weight(self, b: Bucket) -> float:
        """Each scenario counts equally, as in the score (each loss is already a mean over its rows)."""
        return b.S / sum(x.S for x in self.buckets)

    # ---- updates ----------------------------------------------------------------------------------------------
    def actor_step(self, frac: float, losses: list[float]) -> tuple[float, str]:
        """Clips and steps: 'ok', or no step: 'bad' (non-finite) or 'spike' (norm far above the recent median)."""
        norm = float(torch.nn.utils.clip_grad_norm_(self.ctrl.parameters(), self.cfg.clip))
        if not math.isfinite(norm) or not all(map(math.isfinite, losses)):
            return norm, "bad"
        if self.cfg.spike > 0.0 and len(self.norms) >= 20 and norm > self.cfg.spike * float(np.median(self.norms)):
            return norm, "spike"
        self.norms.append(norm)
        for g in self.opt_a.param_groups:
            g["lr"] = self.cfg.lr_actor * self.lr_scale * max(1.0 - frac, 0.01)
        self.opt_a.step()
        with torch.no_grad():
            for pe, p in zip(self.ema.parameters(), self.ctrl.parameters()):
                pe.lerp_(p, 1.0 - self.cfg.ema)
        return norm, "ok"

    def critic_step(self, data: list[dict], frac: float) -> tuple[float, float]:
        """Fits the critic; returns its mean loss and the mean TD(lambda) target."""
        cfg = self.cfg
        for g in self.opt_c.param_groups:
            g["lr"] = cfg.lr_critic * max(1.0 - frac, 0.01)
        sets = []
        with torch.no_grad():
            for d in data:
                h1, n = d["glob"].shape[:2]  # h + 1 states of n rows
                mask = d["mask"].expand(n, -1)
                X_v, X_g, M = d["vav"][1:].flatten(0, 1), d["glob"][1:].flatten(0, 1), mask.repeat(h1 - 1, 1)
                step = _chunk(X_v)
                v = torch.cat([self.target(X_v[i:i + step], X_g[i:i + step], M[i:i + step])
                               for i in range(0, len(M), step)])
                v = torch.where(d["done"], torch.zeros_like(v.view(h1 - 1, n)), v.view(h1 - 1, n))
                G = td_lambda(d["r"], d["done"], v, cfg.gamma, cfg.lam)
                sets.append((d["vav"][:-1].flatten(0, 1), d["glob"][:-1].flatten(0, 1), M, G.flatten()))
        total = torch.zeros((), device=self.device)
        for _ in range(cfg.critic_iters):
            for X_v, X_g, M, Y in sets:
                step = _chunk(X_v)
                for idx in torch.randperm(len(Y), device=self.device).chunk(cfg.critic_batches):
                    self.opt_c.zero_grad(set_to_none=True)
                    for j in idx.split(step):  # the minibatch's mean loss, accumulated in pieces that fit
                        loss = ((self.critic(X_v[j], X_g[j], M[j]) - Y[j]) ** 2).sum() / len(idx)
                        loss.backward()
                        total += loss.detach()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.clip)
                    self.opt_c.step()
        with torch.no_grad():
            for t, c in zip(self.target.parameters(), self.critic.parameters()):
                t.mul_(cfg.target_keep).add_(c, alpha=1.0 - cfg.target_keep)
        return (float(total) / (cfg.critic_iters * cfg.critic_batches * len(sets)),
                float(torch.cat([s[3] for s in sets]).mean()))

    # ---- scoring in the replica -------------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, buckets: list[Bucket], n_steps: int = 0) -> dict:
        """Each scenario from its initial state with the averaged policy: score, reward terms, mean power (kW)."""
        seeds, out = self.cfg.eval_seeds, {}
        for b in buckets:
            rows = b.S * seeds
            env, lay = b.make_env(rows), b.layout(rows)
            obs = env.reset()
            mem = Controller.init_mem(rows, lay, self.device)
            n = min(n_steps or env.n_steps, env.n_steps)
            noisy = (torch.arange(rows, device=self.device) % seeds > 0).float()[:, None] * SIGMA * b.mask[env.row_s]
            gen = torch.Generator(device=self.device).manual_seed(12345)
            tot = torch.zeros(rows, 8, device=self.device)
            for _ in range(n):
                noise = torch.randn(obs.shape, generator=gen, device=self.device) * noisy
                a_c, a_d, mem = self.ema.act(obs + noise, mem, lay)
                obs, r, _, info = env.step(a_c, a_d)
                p = info["post"]
                tot += torch.stack([r, p["regret"], p["norm_cost"], p["norm_carbon"], p["blower"], p["cool"], p["pump"],
                                    p["gas"]], 1)
            tot = tot.cpu().numpy() / n
            del env
            for s, label in enumerate(b.labels):
                rs = tot[s * seeds:(s + 1) * seeds]
                sc = np.abs(rs[:, 0])
                out[label] = {"score": float(sc.mean()), "sd": float(sc.std(ddof=1)) if seeds > 1 else 0.0,
                              "regret": float(rs[:, 1].mean()), "cost": float(rs[:, 2].mean()),
                              "carbon": float(rs[:, 3].mean()),
                              **{f"{k}_kw": float(rs[:, j].mean() / 1e3) for j, k in enumerate(("blower", "cool", "pump", "gas"), 4)},
                              "steps": n}
        out["mean"] = float(np.mean([v["score"] for v in out.values()]))
        return out

    # ---- checkpoints ------------------------------------------------------------------------------------------
    def save(self, path: Path, minutes: float) -> None:
        torch.save({"cfg": asdict(self.cfg), "ctrl": self.ctrl.state_dict(), "ema": self.ema.state_dict(),
                    "ctrl_config": self.ctrl.config, "critic": self.critic.state_dict(),
                    "target": self.target.state_dict(), "opt_a": self.opt_a.state_dict(),
                    "opt_c": self.opt_c.state_dict(), "iteration": self.iteration, "lr_scale": self.lr_scale,
                    "best": self.best, "minutes": minutes}, path)

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device)
        self._restore(ck)
        self.iteration, self.lr_scale, self.best = ck["iteration"], ck["lr_scale"], ck["best"]
        self.minutes_done = ck["minutes"]

    def _snapshot(self) -> dict:
        return copy.deepcopy({"ctrl": self.ctrl.state_dict(), "ema": self.ema.state_dict(),
                              "critic": self.critic.state_dict(), "target": self.target.state_dict(),
                              "opt_a": self.opt_a.state_dict(), "opt_c": self.opt_c.state_dict()})

    def _restore(self, snap: dict) -> None:
        self.ctrl.load_state_dict(snap["ctrl"])
        self.ema.load_state_dict(snap["ema"])
        self.critic.load_state_dict(snap["critic"])
        self.target.load_state_dict(snap["target"])
        self.opt_a.load_state_dict(snap["opt_a"])
        self.opt_c.load_state_dict(snap["opt_c"])

    def _rollback(self, why: str, snap: dict | None) -> None:
        if snap is None:
            return
        self._restore(snap)
        self.lr_scale *= 0.5
        self.norms.clear()
        self.log(f"[train] it {self.iteration}: {why}: networks restored to the best, lr scale {self.lr_scale}")

    # ---- the loop ---------------------------------------------------------------------------------------------
    def train(self) -> dict:
        cfg, out, log = self.cfg, self.out, self.log
        report = {"setup": {}, "evals": []}
        t0 = time.time()
        for b in self.buckets:
            self.burn_in(b)
        report["setup"]["burn_in_s"] = time.time() - t0
        budget = cfg.minutes * 60.0
        t_start, t_ckpt, t_eval = time.time(), time.time(), time.time()
        bad, spikes, ema, best_ema, snap, cooldown, rollbacks = 0, 0, None, -math.inf, None, 0, 0
        n_spikes = 0
        params = list(self.ctrl.parameters())
        with open(out / "train_log.jsonl", "a") as flog:
            while True:
                elapsed = time.time() - t_start + self.minutes_done * 60.0
                if elapsed >= budget:
                    break
                frac = elapsed / budget
                t_it = time.time()
                acc_g = [torch.zeros_like(p) for p in params]
                data, b_norms = [], []
                for b in self.buckets:
                    self.opt_a.zero_grad(set_to_none=False)
                    d = self.window(b, self._weight(b))
                    b_norms.append(torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None)))
                    for a, p in zip(acc_g, params):
                        if p.grad is not None:
                            a += p.grad
                    data.append(d)
                for a, p in zip(acc_g, params):
                    p.grad = a
                losses, rewards, ons, b_norms = torch.stack(  # one host sync for the buckets' statistics
                    [torch.stack([d["loss"], d["reward"], d["on"], n]) for d, n in zip(data, b_norms)]).T.tolist()
                norm, status = self.actor_step(frac, losses)
                if status == "bad":
                    bad += 1
                    self.lr_scale *= 0.5
                    log(f"[train] iteration {self.iteration}: bad gradient (norm {norm:.3g}, losses {losses}); "
                        f"step skipped, lr scale {self.lr_scale}")
                    for b in self.buckets:  # the rows may carry non-finite states: start them afresh
                        self.burn_in(b)
                    if bad >= 3:
                        report["stopped"] = "three bad gradients in a row"
                        break
                    continue
                bad = 0
                for b, d in zip(self.buckets, data):
                    b.start = d["end"]
                c_loss, g_mean = self.critic_step(data, frac)
                self.iteration += 1
                if status == "spike":
                    spikes, n_spikes = spikes + 1, n_spikes + 1
                    log(f"[train] it {self.iteration}: gradient norm {norm:.3g} > {cfg.spike} x median "
                        f"{np.median(self.norms):.3g}: update skipped; per bucket {[f'{x:.3g}' for x in b_norms]}")
                    if spikes >= 5:
                        self._rollback("five skipped updates in a row", self.best_snap or snap)
                        spikes = 0
                else:
                    spikes = 0
                # Collapse guard on the moving average of the training reward (rewards are negative).
                r_mean = float(np.mean(rewards))
                ema = r_mean if ema is None else 0.9 * ema + 0.1 * r_mean
                if cooldown > 0:
                    cooldown -= 1
                elif ema > best_ema:
                    best_ema, snap = ema, self._snapshot()
                elif self.iteration > 20 and ema < 1.25 * best_ema and snap is not None:
                    self._rollback(f"reward average {ema:.5f} vs best {best_ema:.5f}: collapse", snap)
                    rollbacks += 1
                    ema, cooldown = None, 10
                ws = self.wstats.tolist()
                self.wstats.zero_()
                row = {"it": self.iteration, "min": elapsed / 60.0, "loss": float(np.mean(losses)),
                       "reward": r_mean, "grad_norm": norm, "bucket_norms": b_norms, "status": status,
                       "winsorised": ws[0] / max(ws[1], 1.0), "critic_loss": c_loss, "td_target_mean": g_mean,
                       "kp": float(torch.exp(self.ctrl.log_kp.detach())), "ki": float(torch.exp(self.ctrl.log_ki.detach())),
                       "on_frac": float(np.mean(ons)), "rollbacks": rollbacks, "spikes": n_spikes,
                       "s": time.time() - t_it,
                       "env_steps_per_s": sum(b.rows for b in self.buckets) * cfg.h / (time.time() - t_it),
                       "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if self.device.startswith("cuda") else 0}
                flog.write(json.dumps(row) + "\n")
                flog.flush()
                if self.iteration % 10 == 1:
                    log(f"[train] it {row['it']} {row['min']:.1f} min: reward {row['reward']:.5f} loss {row['loss']:.5f} "
                        f"grad {norm:.3g} winsorised {row['winsorised']:.3f} critic {c_loss:.3g} G {g_mean:.3f} "
                        f"on {row['on_frac']:.2f} Kp {row['kp']:.3f} Ki {row['ki']:.4f} | {row['s']:.1f} s/it "
                        f"{row['env_steps_per_s']:,.0f} env-steps/s {row['mem_gb']:.1f} GB | spikes {n_spikes}")
                if time.time() - t_eval >= cfg.eval_minutes * 60.0:
                    self._eval_and_keep(report, elapsed)
                    t_eval = time.time()
                if time.time() - t_ckpt >= 15 * 60.0:
                    self.save(out / "last.pt", elapsed / 60.0)
                    t_ckpt = time.time()
        elapsed = time.time() - t_start + self.minutes_done * 60.0
        self._eval_and_keep(report, elapsed)
        self.save(out / "last.pt", elapsed / 60.0)
        if self.best_snap is not None:  # the training scenarios, scored with the kept policy
            self._restore(self.best_snap)
        try:
            report["train_eval"] = self.evaluate(self.buckets, cfg.eval_steps)
            self.log(f"[eval] training scenarios with the kept policy: {report['train_eval']['mean']:.5f}")
        except Exception as e:  # a failed evaluation must not lose the run
            self.log(f"[eval] training scenarios failed: {type(e).__name__}: {e}")
        report.update(iterations=self.iteration, minutes=elapsed / 60.0, best=self.best, rollbacks=rollbacks,
                      spikes=n_spikes)
        return report

    def _eval_and_keep(self, report: dict, elapsed: float) -> None:
        try:
            ev = self.evaluate(self.val or self.buckets, self.cfg.eval_steps)
        except Exception as e:  # a failed evaluation must not end the training
            self.log(f"[eval] failed: {type(e).__name__}: {e}")
            return
        ev.update(it=self.iteration, min=elapsed / 60.0, held_out=bool(self.val))
        report["evals"].append(ev)
        with open(self.out / "eval_log.jsonl", "a") as f:
            f.write(json.dumps(ev) + "\n")
        key = ev["mean"]
        if key < self.best:
            self.best, self.bad_evals = key, 0
            self.best_snap = self._snapshot()
            save_policy(self.ema, self.out / "d1_policy.pt")
        elif key > 1.15 * self.best:
            self.bad_evals += 1
            if self.bad_evals >= 2:
                self._rollback(f"held-out {key:.5f} vs best {self.best:.5f} twice", self.best_snap)
                self.bad_evals = 0
        per = " ".join(f"{k}:{v['score']:.5f}" for k, v in ev.items() if isinstance(v, dict))
        what = "held-out" if self.val else "train"
        self.log(f"[eval] it {self.iteration} {elapsed / 60.0:.1f} min: {what} {key:.5f} (best {self.best:.5f}) | {per}")


# ---- start-up checks -------------------------------------------------------------------------------------------

def parity_check(ctrl: Controller, rep: dict, other: dict, device: str, n_steps: int, tmp: Path) -> dict:
    """The submission path (KagglePolicy on the organisers' simulator, CPU) and the training path (Controller.act on
    the device, in a padded batch with another building) given the same observations must act the same."""
    from hvac_controller.d1.policy import KagglePolicy
    from hvac_controller.sim import load_env
    from smart_buildings import evaluate

    save_policy(ctrl, tmp)
    env, meta = load_env(Path(rep["info"]["config"]))
    obs, _ = env.reset()
    pol = KagglePolicy(meta, env.action_spec(), weights=tmp)
    seen, acts = [], []
    for _ in range(n_steps):
        a = evaluate.validate_action(pol.get_action(obs), env.action_spec())
        seen.append(np.asarray(obs, dtype=np.float32))
        acts.append(a)
        obs, *_ = env.step(a)
    lays = [Layout(meta), Layout(other["meta"])]
    lay = stack_layouts(lays, np.arange(2), device)
    n_obs = max(len(rep["spec"].obs), len(other["spec"].obs))
    mem = Controller.init_mem(2, lay, device)
    n_c, n_d = lays[0].n_cont, lays[0].n_disc
    worst_c, worst_d = 0.0, 0
    with torch.no_grad():
        for o, a in zip(seen, acts):
            x = torch.zeros(2, n_obs, device=device)
            x[0, :len(o)] = torch.as_tensor(o, device=device)
            a_c, a_d, mem = ctrl.act(x, mem, lay)
            worst_c = max(worst_c, float(np.abs(a_c[0, :n_c].clamp(-1, 1).cpu().numpy() - a["continuous_action"]).max()))
            worst_d = max(worst_d, int(np.abs(a_d[0, :n_d].round().cpu().numpy() - a["discrete_action"]).max()))
    return {"steps": n_steps, "cont_max_abs": worst_c, "disc_mismatch": worst_d, "ok": worst_c < 1e-4 and worst_d == 0}


def group_check(reps: list[dict], device: str, n_steps: int = 288) -> dict:
    """One padded environment of several buildings against each building alone, on the same random actions."""
    ops = Ops("torch", dtype=torch.float32, device=device)
    b = 2
    g = BuildingEnv([r["spec"] for r in reps], [r["rom"] for r in reps], [r["exo"] for r in reps], b * len(reps), ops,
                    exact=False, emulate_f32=False)
    alone = [BuildingEnv(r["spec"], r["rom"], r["exo"], b, ops, exact=False, emulate_f32=False) for r in reps]
    gen = torch.Generator(device=device).manual_seed(0)
    g.reset()
    for e in alone:
        e.reset()
    worst = 0.0
    with torch.no_grad():
        for _ in range(n_steps):
            a_c = torch.rand(g.num_envs, g.num_actions[0], generator=gen, device=device) * 2.0 - 1.0
            a_d = (torch.rand(g.num_envs, g.num_actions[1], generator=gen, device=device) > 0.5).float()
            _, r_g, _, _ = g.step(a_c, a_d)
            for i, (e, r) in enumerate(zip(alone, reps)):
                rows = slice(i * b, (i + 1) * b)
                _, r_e, _, _ = e.step(a_c[rows, :len(r["spec"].cont)], a_d[rows, :len(r["spec"].disc)])
                worst = max(worst, float(((r_g[rows] - r_e).abs() / r_e.abs().clamp(min=1e-6)).max()))
    return {"steps": n_steps, "reward_rel_max": worst, "ok": worst < 1e-4}


def checkpoint_check(trainer: Trainer, b: Bucket) -> dict:
    """Gradients of one window with checkpointed segments against the plain backward pass (seg = h)."""
    params = list(trainer.ctrl.parameters())
    start = {"state": dict(b.start["state"]), "obs": b.start["obs"], "mem": b.start["mem"], "k": b.start["k"]}
    grads = []
    seg = trainer.cfg.seg
    for s in (seg, trainer.cfg.h):
        trainer.cfg.seg = s
        b.start = start
        trainer.opt_a.zero_grad(set_to_none=False)
        trainer.window(b, 1.0)
        grads.append([p.grad.clone() for p in params])
    trainer.cfg.seg = seg
    b.start = start
    trainer.opt_a.zero_grad(set_to_none=False)
    diff = torch.sqrt(sum(((x - y) ** 2).sum() for x, y in zip(*grads)))
    ref = torch.sqrt(sum((y ** 2).sum() for y in grads[1]))
    err = float(diff / ref.clamp(min=1e-30))
    finite = all(bool(torch.isfinite(g).all()) for g in grads[0])
    return {"grad_rel_err": err, "finite": finite, "ok": finite and err < 1e-4}


def load_buckets(caches: list[Path], n_buckets: int, row_budget: int, seg: int, device: str,
                 rows_per: int | None = None) -> list[Bucket]:
    """Caches (named <building>__<weather>.pkl by data.ipynb) grouped by building, buildings sorted by size and
    split into n_buckets padded environments."""
    groups: dict[str, list[dict]] = {}
    for p in sorted(caches):
        rep = load_replica(p)
        rep["info"]["label"] = p.stem
        groups.setdefault(p.stem.split("__")[0], []).append(rep)
    if not groups:
        return []
    order = sorted(groups, key=lambda k: (groups[k][0]["spec"].n_vavs, k))
    parts = [list(p) for p in np.array_split(np.array(order, dtype=object), min(n_buckets, len(order))) if len(p)]
    return [Bucket([r for k in part for r in groups[k]], row_budget, seg, device, rows_per) for part in parts]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--caches", required=True, help="comma-separated replica caches (.pkl) to train on")
    ap.add_argument("--val-caches", default="", help="comma-separated caches of held-out scenarios (scored only)")
    ap.add_argument("--resume", type=Path, help="a last.pt to continue from")
    ap.add_argument("--smoke", action="store_true", help="start-up checks before the run")
    for f, v in asdict(Config()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(v) if not isinstance(v, bool) else int, default=v)
    a = ap.parse_args()
    cfg = Config(**{f: (bool(getattr(a, f)) if isinstance(v, bool) else getattr(a, f)) for f, v in asdict(Config()).items()})
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    log = (lambda s: print(s, flush=True))
    caches = [Path(p) for p in a.caches.split(",")]
    val_caches = [Path(p) for p in a.val_caches.split(",") if p]
    report = {"cfg": asdict(cfg), "caches": [str(p) for p in caches], "val_caches": [str(p) for p in val_caches],
              "device": device,
              "device_name": torch.cuda.get_device_name() if device == "cuda" else "cpu", "torch": torch.__version__}
    (out / "config.json").write_text(json.dumps(report, indent=1))
    t0 = time.time()
    trainer = None
    try:
        buckets = load_buckets(caches, cfg.buckets, cfg.row_budget, cfg.seg, device)
        val = load_buckets(val_caches, 1, 0, cfg.seg, device, rows_per=1)
        trainer = Trainer(cfg, buckets, val, device, out, log)
        if a.resume:
            trainer.load(a.resume)
        log(f"[train] {len(buckets)} bucket(s): " + "; ".join(
            f"{b.S} scenarios of {len({l.split('__')[0] for l in b.labels})} buildings, V <= {b.V}, rows {b.rows} "
            f"({b.rows // b.S} per scenario), sweeps {b.env.rom.sweeps}" for b in buckets)
            + f" | {sum(b.S for b in val)} held-out scenarios | row budget {cfg.row_budget} "
              f"| setup {time.time() - t0:.0f}s")
        if a.smoke:
            reps = [r for b in buckets + val for r in b.reps]
            first = next(r for r in reps if Path(r["info"]["config"]).exists())  # the original simulator runs it
            other = next(r for r in reps if r["info"]["label"].split("__")[0] != first["info"]["label"].split("__")[0])
            report["parity"] = parity_check(trainer.ctrl, first, other, device, 36, out / "smoke_policy.pt")
            log(f"[smoke] parity: {report['parity']}")
            report["group"] = group_check([first, other], device)
            log(f"[smoke] padded batch vs each building alone: {report['group']}")
            small = Bucket([first, other], 0, cfg.seg, device, rows_per=4)
            small.critic_rows = torch.arange(small.rows, device=device)
            trainer.burn_in(small)
            report["checkpoint"] = checkpoint_check(trainer, small)
            log(f"[smoke] checkpointed vs plain gradient: {report['checkpoint']}")
            del small, reps, first, other
            if not (report["parity"]["ok"] and report["group"]["ok"] and report["checkpoint"]["ok"]):
                raise SystemExit("[smoke] a start-up check failed")
        report.update(trainer.train())
    except torch.cuda.OutOfMemoryError as e:
        where = [f for f in traceback.extract_tb(e.__traceback__) if "hvac_controller" in f.filename][-1]
        log(f"[train] out of memory in {where.name} ({where.filename.split('/')[-1]}:{where.lineno}), "
            f"{torch.cuda.memory_allocated() / 2 ** 30:.1f} GB allocated: {str(e).splitlines()[0]}")
        if trainer is not None and trainer.iteration > 0:  # mid-run: a restart would overrun the session
            report.update(stopped=f"out of memory at iteration {trainer.iteration}", iterations=trainer.iteration,
                          best=trainer.best)  # d1_policy.pt on disk is the best kept so far
            log(f"[train] out of memory at iteration {trainer.iteration}: stopped, the kept policy stays")
        elif cfg.row_budget // 2 >= Config.row_budget // 8:  # before the first update: half the rows, a fresh process
            args = sys.argv[1:]
            if "--row-budget" in args:
                del args[args.index("--row-budget"):args.index("--row-budget") + 2]
            log(f"[train] restarting with row budget {cfg.row_budget // 2}")
            os.execv(sys.executable, [sys.executable, "-m", "hvac_controller.d1.train", *args,
                                      "--row-budget", str(cfg.row_budget // 2)])
    report["seconds"] = time.time() - t0
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    log(f"[train] done: best replica score {report.get('best')} after {report.get('iterations')} iterations")
    if not report.get("iterations"):
        raise SystemExit("[train] no update completed")


if __name__ == "__main__":
    main()
