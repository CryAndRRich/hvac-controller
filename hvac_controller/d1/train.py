"""Direction 1, step 6: SHAC through the replica (Xu et al., ICLR 2022), re-implemented from the paper.

Per building (one ``BuildingEnv`` grouping that building's scenarios; as many batch rows as ``row_budget`` allows,
row_budget >= rows x h x VAVs, since the per-VAV activations of the window dominate the memory; rows spread over the
episode by a burn-in), every iteration, on all buildings or on ``per_update`` of them drawn at random (the others'
rows wait where they stopped):

1. a window of h steps from where the rows stopped: the policy acts, the replica steps, and the actor loss is
   -(1 / (B h)) sum over rows of [sum_t gamma^t r_t + gamma^h V'(s_h)], with V' the target critic; a row whose
   episode ends inside the window adds its finished return (the end is a real end: V = 0, so V' is only evaluated at
   the window's end) and restarts from the scenario's initial state, as the gateway's episodes do; one backward pass
   (gradients accumulate over the iteration's buildings, each weighted by its share of their scenarios);
2. one actor step: gradient norm clipped, Adam (0.7, 0.95), learning rate decaying linearly over the time budget;
3. the critic fitted to TD(lambda) targets of the window on ``critic_rows`` fixed rows per building (spread over the
   episode by the burn-in; all h x B states would cost far more than the window), 16 passes x 4 shuffled
   minibatches, then the target critic follows it (Polyak).

The clip is ``clip``, or ``clip_rel`` x the median pre-clip norm of the last ``clip_window`` steps when that is
lower. First-order gradients through the run-command and clamp kinks are heavy-tailed (V1: typically 0.005, spikes
up to 5.7e4), so the absolute clip let a spike through as a step about 200 x normal, whose second moment then damped
Adam for the rest of the run; Adam needs a clip at the scale of the typical gradient under heavy-tailed noise
(Chezhegov et al., ICML 2025), and a threshold relative to recent norms keeps it there as that scale drifts (as
SPAM, ICLR 2025, and ZClip, 2025, do for language models). Spikes are clipped, not skipped: the step still counts.
The clip also holds per window: each building's part of the gradient (the gradient after its backward pass minus
before) is clipped to the limit x its weight, so no building takes more than its share of a step, and dropped only
when it or the window's loss is non-finite. Over 288 steps the backward pass through the closed loop grows
exponentially once the policy is past about 30 updates (V2.2/V2.3: 1e4 to inf in most windows of every building).
Dropping those windows (V2.3) left the policy unchanged and so exploding for the rest of the run; clipped, they still
step. The cause itself is the policy's feedback (Schnell & Thuerey, ICLR 2024): ``feedback`` weights the gradient
through the observations (``window``) and ``anneal`` halves it whenever a window's gradient bursts, so the run
starts as full BPTT, which learnt fastest in V2.2, and moves towards the stopped-feedback field only as far as the
policy's own gain requires.

The critic is a DeepSets over VAVs with privileged global inputs: the outside temperature and comfort flags of the
next 24 hours.

Scored in the replica every ``eval_minutes``: every scenario from its initial state for the whole episode, seed 0
plain and seeds >= 1 with observation noise of evaluate.SIGMA, on the held-out scenarios whose label starts with
``select`` (``--val-caches``; all held-out ones when empty, the training ones when there are none). The best policy
by their mean is kept (``d1_policy.pt``, the file policy.KagglePolicy loads); at the end it is scored on every
held-out scenario. Guards: a window's non-finite gradient is dropped (above); when every window of an iteration is
dropped no step is taken, every 5 such iterations in a row restore the networks saved at the best training reward,
and 20 stop the run; rows move on, and those with non-finite states restart; a collapse (the mean over buildings of
the moving average of each one's training reward 50 % worse than its best) restores the networks and optimisers
saved at that best and halves the learning rate; ``patience`` scorings without a better kept score end the run; the
time budget ends the run cleanly,
``last.pt`` is written every 15 minutes and ``--resume`` continues from it; out of GPU memory, the process restarts
itself with half the row budget, from ``last.pt`` when there is one.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hvac_controller.d1.cache import load_replica
from hvac_controller.d1.evaluate import SIGMA, obs_noise_mask
from hvac_controller.d1.ops import Ops
from hvac_controller.d1.policy import N_GLOBAL, N_VAV, Controller, Layout, _mlp, save_policy
from hvac_controller.d1.torch_env import BuildingEnv

AHEAD = 24  # hours of outside temperature and comfort flags the critic sees
SPEC_FIELDS = ("dt", "zone_names", "vav_ids", "ahu_ids", "zone_cells", "vav_zone_share", "vav_ahu", "ahu_weight", "ahu",
               "vav", "hws", "sim", "reward", "phys", "cont", "disc", "obs", "exo_fields", "occupancy_norm", "reset_raw",
               "tank0", "ret0", "field_names")


@dataclass
class Config:
    h: int = 72
    gamma: float = 0.995
    lam: float = 0.95
    row_budget: int = 2_870_000  # rows x h x VAVs per window: ~12 GB on a T4 (pilot, 2026-10-20)
    minutes: float = 30.0  # training time budget
    lr_actor: float = 2e-3
    lr_critic: float = 5e-4
    clip: float = 1.0
    clip_rel: float = 0.0  # > 0: clip at min(clip, clip_rel x median of the last clip_window pre-clip norms)
    clip_window: int = 50
    per_update: int = 0  # buildings per iteration, drawn at random; 0: all
    select: str = ""  # held-out labels starting with this choose the kept policy
    critic_iters: int = 16
    critic_batches: int = 4
    critic_rows: int = 256  # rows whose states the critic learns from
    target_keep: float = 0.2  # target <- keep * target + (1 - keep) * critic
    feedback: float = 1.0  # weight of the gradient through the policy's observations: 1 full BPTT, 0 stopped
    anneal: bool = False  # halve the feedback weight whenever a window's gradient bursts
    burst: float = 10.0  # a window's gradient bursts above burst x its cap (or when non-finite)
    patience: int = 0  # stop after this many scorings without a better kept score; 0: never
    eval_minutes: float = 10.0
    eval_seeds: int = 4
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
    """V(s) = mean over VAVs of phi(VAV features, global) + psi(global), global with the next day's forecast."""

    def __init__(self, width: int):
        super().__init__()
        n_g = N_GLOBAL + 2 * AHEAD
        self.phi = _mlp(N_VAV + n_g, width, 1)
        self.psi = _mlp(n_g, width, 1)

    def forward(self, vav, glob):
        B, V = vav.shape[:2]
        x = torch.cat([vav, glob[:, None, :].expand(B, V, glob.shape[1])], 2)
        return self.phi(x)[..., 0].mean(1) + self.psi(glob)[..., 0]


def _plant_key(spec) -> bytes:
    return pickle.dumps([getattr(spec, f) for f in SPEC_FIELDS])


class Building:
    """One building's scenarios as one grouped environment, its layout and its start states."""

    def __init__(self, reps: list[dict], row_budget: int, h: int, device: str):
        spec = reps[0]["spec"]
        for r in reps:
            if _plant_key(r["spec"]) != _plant_key(spec):
                raise ValueError(f"{r['info']['label']} is not the same building as {reps[0]['info']['label']}")
            Layout(r["meta"]).check(r["spec"])  # every scenario's metadata against its own simulator objects
        self.reps, self.S, self.device = reps, len(reps), device
        self.labels = [r["info"]["label"] for r in reps]
        self.lay = Layout(reps[0]["meta"]).tensors(device)
        self.ops = Ops("torch", dtype=torch.float32, device=device)
        rows = row_budget // (h * self.lay["V"])
        self.rows = max(rows - rows % self.S, self.S)
        self.env = self.make_env(self.rows)
        e = self.env
        self.fc_t = e.exo[:, e.exo_cols["t_amb"]]
        self.fc_c = e.exo[:, e.exo_cols["comfort"]]
        self.ahead = torch.arange(AHEAD, device=device) * int(round(3600 / spec.dt))
        self.mask = torch.as_tensor(obs_noise_mask(spec), dtype=torch.float32, device=device)
        self.critic_rows = None  # set by the trainer
        self.start = None

    def make_env(self, rows: int) -> BuildingEnv:
        return BuildingEnv(self.reps[0]["spec"], [r["rom"] for r in self.reps], [r["exo"] for r in self.reps], rows,
                           self.ops, exact=False, emulate_f32=False)

    def forecast(self, n_steps: int, row_s, k):
        idx = row_s[:, None] * n_steps + torch.clamp(k[:, None] + self.ahead, max=n_steps - 1)
        return torch.cat([(self.fc_t[idx] - 295.0) / 10.0, self.fc_c[idx]], 1)


class Trainer:
    def __init__(self, cfg: Config, buildings: list[Building], val: list[Building], device: str, out: Path, log=print):
        self.cfg, self.buildings, self.val, self.device, self.out, self.log = cfg, buildings, val, device, out, log
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        floors = {(float(np.max(s.ahu["min_pressure"])), float(np.min(s.ahu["max_recirc"])),
                   float(np.max(s.vav["min_damper"]))) for b in buildings + val for s in (r["spec"] for r in b.reps)}
        if len(floors) != 1:
            raise ValueError(f"the controller's equipment floors differ between buildings: {floors}")
        p_floor, max_recirc, min_damper = floors.pop()
        self.ctrl = Controller(width=cfg.width, p_floor=p_floor, max_recirc=max_recirc,
                               min_damper=min_damper).to(device)
        self.critic = Critic(cfg.width).to(device)
        self.target = Critic(cfg.width).to(device)
        self.target.load_state_dict(self.critic.state_dict())
        self.target.requires_grad_(False)
        self.params = list(self.ctrl.parameters())
        for p in self.params:  # each window's part of the gradient is the difference of the accumulated ones
            p.grad = torch.zeros_like(p)
        self.opt_a = torch.optim.Adam(self.params, lr=cfg.lr_actor, betas=(0.7, 0.95))
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic, betas=(0.7, 0.95))
        self.iteration, self.lr_scale, self.best = 0, 1.0, math.inf
        self.best_ctrl = None  # the kept policy's weights, for the final scoring
        self.kappa = cfg.feedback  # the current feedback weight
        self.stale = 0  # scorings since the last better kept score
        self.norms = collections.deque(maxlen=cfg.clip_window)  # pre-clip norms of the last steps taken
        self.minutes_done = 0.0  # training time of earlier sessions (resume)
        pool = val or buildings
        self.scored = [b for b in pool if any(s.startswith(cfg.select) for s in b.labels)]  # scored while training
        if not self.scored:
            raise ValueError(f"no {'held-out' if val else 'training'} scenario starts with select {cfg.select!r}")
        for b in buildings:  # spread over every scenario (V1's first 256 rows missed the last); a slice gives views
            b.critic_rows = slice(0, b.rows, -(-b.rows // cfg.critic_rows))

    # ---- features ---------------------------------------------------------------------------------------------
    def critic_inputs(self, b: Building, obs, I, k, s):
        """Critic features of states given by their observations, integrators, step indices and scenarios."""
        vav, glob, *_ = self.ctrl.features(obs, I, b.lay)
        return vav, torch.cat([glob, b.forecast(b.env.n_steps, s, k)], 1)

    # ---- start states -----------------------------------------------------------------------------------------
    @torch.no_grad()
    def burn_in(self, b: Building) -> None:
        """One row per scenario for the whole episode with the current policy; row j of scenario s starts at the
        state that run reached at offset j * n / (rows per scenario)."""
        env = b.make_env(b.S)
        obs = env.reset()
        I = torch.zeros(b.S, b.lay["V"], device=self.device)
        per = b.rows // b.S
        offsets = (np.arange(per) * env.n_steps) // per
        slot = {int(o): u for u, o in enumerate(np.unique(offsets))}
        snaps = []
        for k in range(env.n_steps):
            if k in slot:
                snaps.append({**{n: v.clone() for n, v in env.state.items()}, "obs": obs.clone(), "I": I.clone()})
            a_c, a_d, I = self.ctrl.act(obs, I, b.lay)
            obs, _, _, _ = env.step(a_c, a_d)
        rows = torch.arange(b.rows, device=self.device)
        s_of = rows // per
        u_of = torch.as_tensor([slot[int(o)] for o in offsets], device=self.device)[rows % per]
        stacked = {n: torch.stack([sn[n] for sn in snaps]) for n in snaps[0]}  # (n_offsets, S, ...)
        start = {n: v[u_of, s_of] for n, v in stacked.items()}
        b.start = {"state": {n: start[n] for n in env.state}, "obs": start["obs"], "I": start["I"],
                   "k": torch.as_tensor(offsets, device=self.device)[rows % per]}

    # ---- one window ---------------------------------------------------------------------------------------------
    def window(self, b: Building, weight: float) -> dict:
        """Rollout from b.start, actor loss, backward (gradients accumulate). Returns the loss, the end state and the
        critic's data (states s_0 .. s_h, rewards and episode ends of the critic's rows), all on the device.

        The policy sees its observations with their gradient scaled by ``kappa`` (the forward values are unchanged):
        the adjoint then runs through the plant in full and through the policy's feedback (observation -> action)
        at kappa. The plant's own state Jacobian is a damped thermal system, so with kappa 0 the backward pass cannot
        grow over the window (the stopped-feedback field of Schnell & Thuerey, ICLR 2024; D.VA, NeurIPS 2025).
        Per step only the policy, the replica and the return run; the critic's features are computed once from the
        stored observations, and the episode-end bookkeeping runs only at the steps where some row's episode ends,
        known from the start indices (h < episode, so at most once per row)."""
        cfg, env, ctrl, kappa = self.cfg, b.env, self.ctrl, self.kappa
        env.state = dict(b.start["state"])
        env.k, env.obs, I = b.start["k"], b.start["obs"], b.start["I"]
        B, n, h, sel = env.num_envs, env.n_steps, cfg.h, b.critic_rows
        ends = {int(t) for t in n - 1 - b.start["k"].cpu().numpy() if t < h - 1}
        gamma_k = torch.ones(B, device=self.device)
        acc = torch.zeros(B, device=self.device)
        loss = torch.zeros((), device=self.device)
        seen = {"obs": [], "I": [], "k": [], "r": [], "done": [], "on": []}  # per step, detached
        for t in range(h):
            obs = env.obs
            seen["obs"].append(obs.detach()[sel])
            seen["I"].append(I.detach()[sel])
            seen["k"].append(env.k[sel])
            if kappa < 1.0:
                obs = obs.detach() + kappa * (obs - obs.detach()) if kappa > 0.0 else obs.detach()
            a_c, a_d, I = ctrl.act(obs, I, b.lay)
            _, r, done, _ = env.step(a_c, a_d)
            acc = acc + gamma_k * r
            seen["r"].append(r.detach())
            seen["done"].append(done)
            seen["on"].append(a_d.detach())
            if t in ends:
                loss = loss - torch.where(done, acc, torch.zeros_like(acc)).sum()
                gamma_k = torch.where(done, torch.ones_like(gamma_k), gamma_k * cfg.gamma)
                acc = torch.where(done, torch.zeros_like(acc), acc)
                env.reset_done(done)
                I = torch.where(done[:, None], torch.zeros_like(I), I)
            elif t < h - 1:
                gamma_k = gamma_k * cfg.gamma
        # The window's end: bootstrap with the target critic (0 where the episode just ended), then restart those rows.
        end_obs, end_I, end_k = env.obs.detach()[sel], I.detach()[sel], env.k[sel]
        cv, cg = self.critic_inputs(b, env.obs, I, env.k, env.row_s)
        v = torch.where(done, torch.zeros_like(acc), self.target(cv, cg))
        loss = loss - (acc + cfg.gamma * gamma_k * v).sum()
        env.reset_done(done)
        I = torch.where(done[:, None], torch.zeros_like(I), I)
        loss = loss / (B * h)
        (loss * weight).backward()
        with torch.no_grad():
            obs_s = torch.stack(seen["obs"] + [end_obs])  # (h + 1, n, n_obs)
            m = obs_s.shape[1]
            cv, cg = self.critic_inputs(b, obs_s.flatten(0, 1), torch.stack(seen["I"] + [end_I]).flatten(0, 1),
                                        torch.stack(seen["k"] + [end_k]).flatten(), env.row_s[sel].repeat(h + 1))
            r_all = torch.stack(seen["r"])
            out = {"vav": cv.view(h + 1, m, *cv.shape[1:]), "glob": cg.view(h + 1, m, -1), "r": r_all[:, sel],
                   "done": torch.stack(seen["done"])[:, sel]}
            out.update(loss=loss.detach(), reward=r_all.mean(), on=torch.stack(seen["on"]).mean(),
                       end={"state": {n_: v_.detach() for n_, v_ in env.state.items()}, "obs": env.obs.detach(),
                            "I": I.detach(), "k": env.k})
        return out

    # ---- updates ----------------------------------------------------------------------------------------------
    def clip_limit(self) -> float:
        """``clip``, or clip_rel x the median of the recent pre-clip norms when lower (after 5 steps)."""
        cfg = self.cfg
        if cfg.clip_rel > 0.0 and len(self.norms) >= 5:
            return min(cfg.clip, cfg.clip_rel * float(np.median(self.norms)))
        return cfg.clip

    def bound(self, before: list, cap: float, finite_loss: bool) -> tuple[float, bool]:
        """Holds the gradient one window added (the gradient now minus ``before``) to norm ``cap``, or takes it back
        when it or the window's loss is non-finite. Returns its norm and whether it was kept."""
        delta = [p.grad - g for p, g in zip(self.params, before)]
        norm = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(x) for x in delta])))
        if finite_loss and norm <= cap:  # False for a NaN norm
            return norm, True
        keep = finite_loss and math.isfinite(norm)
        with torch.no_grad():
            for p, g, x in zip(self.params, before, delta):
                p.grad.copy_(g + (cap / norm) * x if keep else g)
        return norm, keep

    def actor_step(self, frac: float, limit: float) -> float:
        """Clips the summed gradient to ``limit`` and steps; returns its pre-clip norm."""
        cfg = self.cfg
        norm = float(torch.nn.utils.clip_grad_norm_(self.params, limit))
        self.norms.append(norm)
        for g in self.opt_a.param_groups:
            g["lr"] = cfg.lr_actor * self.lr_scale * max(1.0 - frac, 0.01)
        self.opt_a.step()
        return norm

    def critic_step(self, data: list[dict], frac: float) -> tuple[float, float]:
        """Fits the critic; returns its mean loss and the mean TD(lambda) target."""
        cfg = self.cfg
        for g in self.opt_c.param_groups:
            g["lr"] = cfg.lr_critic * max(1.0 - frac, 0.01)
        sets = []
        with torch.no_grad():
            for d in data:
                h1, n = d["glob"].shape[:2]  # h + 1 states of n rows
                v = self.target(d["vav"][1:].flatten(0, 1), d["glob"][1:].flatten(0, 1)).view(h1 - 1, n)
                v = torch.where(d["done"], torch.zeros_like(v), v)
                G = td_lambda(d["r"], d["done"], v, cfg.gamma, cfg.lam)
                sets.append((d["vav"][:-1].flatten(0, 1), d["glob"][:-1].flatten(0, 1), G.flatten()))
        total = torch.zeros((), device=self.device)
        for _ in range(cfg.critic_iters):
            for X_v, X_g, Y in sets:
                for idx in torch.randperm(len(Y), device=self.device).chunk(cfg.critic_batches):
                    loss = ((self.critic(X_v[idx], X_g[idx]) - Y[idx]) ** 2).mean()
                    self.opt_c.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.clip)
                    self.opt_c.step()
                    total += loss.detach()
        with torch.no_grad():
            for t, c in zip(self.target.parameters(), self.critic.parameters()):
                t.mul_(cfg.target_keep).add_(c, alpha=1.0 - cfg.target_keep)
        return (float(total) / (cfg.critic_iters * cfg.critic_batches * len(sets)),
                float(torch.cat([Y for _, _, Y in sets]).mean()))

    # ---- scoring in the replica -------------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, buildings: list[Building], n_steps: int = 0) -> dict:
        """Per scenario of the given buildings: score (and its sd over seeds), regret, cost and carbon; "mean" over
        them all and "key", the mean over the labels starting with ``select``, which chooses the kept policy."""
        seeds, out = self.cfg.eval_seeds, {}
        for b in buildings:
            env = b.make_env(b.S * seeds)
            obs = env.reset()
            n = min(n_steps or env.n_steps, env.n_steps)
            I = torch.zeros(env.num_envs, b.lay["V"], device=self.device)
            noisy = (torch.arange(env.num_envs, device=self.device) % seeds > 0).float()[:, None] * SIGMA * b.mask
            gen = torch.Generator(device=self.device).manual_seed(12345)
            tot = torch.zeros(env.num_envs, 4, device=self.device)
            for _ in range(n):
                noise = torch.randn(obs.shape, generator=gen, device=self.device) * noisy
                a_c, a_d, I = self.ctrl.act(obs + noise, I, b.lay)
                obs, r, _, info = env.step(a_c, a_d)
                p = info["post"]
                tot += torch.stack([r, p["regret"], p["norm_cost"], p["norm_carbon"]], 1)
            tot = tot.cpu().numpy() / n
            for s, label in enumerate(b.labels):
                rows = tot[s * seeds:(s + 1) * seeds]
                sc = np.abs(rows[:, 0])
                out[label] = {"score": float(sc.mean()), "sd": float(sc.std(ddof=1)) if seeds > 1 else 0.0,
                              "regret": float(rows[:, 1].mean()), "cost": float(rows[:, 2].mean()),
                              "carbon": float(rows[:, 3].mean()), "steps": n}
        scores = {k: v["score"] for k, v in out.items()}
        out["mean"] = float(np.mean(list(scores.values())))
        out["key"] = float(np.mean([v for k, v in scores.items() if k.startswith(self.cfg.select)]))
        return out

    # ---- checkpoints ------------------------------------------------------------------------------------------
    def save(self, path: Path, minutes: float) -> None:
        torch.save({"cfg": asdict(self.cfg), "ctrl": self.ctrl.state_dict(), "ctrl_config": self.ctrl.config,
                    "critic": self.critic.state_dict(), "target": self.target.state_dict(),
                    "opt_a": self.opt_a.state_dict(), "opt_c": self.opt_c.state_dict(), "iteration": self.iteration,
                    "lr_scale": self.lr_scale, "best": self.best, "best_ctrl": self.best_ctrl,
                    "norms": list(self.norms), "kappa": self.kappa, "stale": self.stale, "minutes": minutes}, path)

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device)
        self.ctrl.load_state_dict(ck["ctrl"])
        self.critic.load_state_dict(ck["critic"])
        self.target.load_state_dict(ck["target"])
        self.opt_a.load_state_dict(ck["opt_a"])
        self.opt_c.load_state_dict(ck["opt_c"])
        self.iteration, self.lr_scale, self.best = ck["iteration"], ck["lr_scale"], ck["best"]
        self.best_ctrl = ck["best_ctrl"]
        self.norms.extend(ck["norms"])
        self.kappa, self.stale = ck["kappa"], ck["stale"]
        self.minutes_done = ck["minutes"]

    # ---- the loop ---------------------------------------------------------------------------------------------
    def train(self) -> dict:
        cfg, out, log = self.cfg, self.out, self.log
        report = {"setup": {}, "evals": []}
        t0 = time.time()
        for b in self.buildings:
            self.burn_in(b)
        report["setup"]["burn_in_s"] = time.time() - t0
        log(f"[train] burn-in of {len(self.buildings)} building(s): {time.time() - t0:.0f}s")
        budget = cfg.minutes * 60.0
        t_start, t_ckpt, t_eval = time.time(), time.time(), time.time()
        best_ema, snap, cooldown, rollbacks, dropped, skipped, streak = -math.inf, None, 0, 0, 0, 0, 0
        ema: dict[int, float] = {}  # per building: moving average of its training reward (rewards are negative)
        n_pick = cfg.per_update if 0 < cfg.per_update < len(self.buildings) else len(self.buildings)
        with open(out / "train_log.jsonl", "a") as flog:
            while True:
                elapsed = time.time() - t_start + self.minutes_done * 60.0
                if elapsed >= budget:
                    break
                if cfg.patience and self.stale >= cfg.patience:
                    report["stopped"] = f"no better kept score in {cfg.patience} scorings"
                    break
                frac = elapsed / budget
                t_it = time.time()
                self.opt_a.zero_grad(set_to_none=False)
                picked = sorted(self.rng.choice(len(self.buildings), n_pick, replace=False).tolist())
                chosen = [self.buildings[i] for i in picked]
                n_scen = sum(b.S for b in chosen)
                limit = self.clip_limit()
                kept, data, seen, w_norms = [], [], [], []  # seen: (building, loss, reward, on) of finite windows
                for i, b in zip(picked, chosen):
                    weight = b.S / n_scen  # each scenario counts equally, as in the score
                    before = [p.grad.clone() for p in self.params]
                    d = self.window(b, weight)
                    loss = float(d["loss"])
                    w_norm, ok = self.bound(before, limit * weight, math.isfinite(loss))
                    w_norms.append(w_norm)
                    if math.isfinite(loss):
                        seen.append((i, loss, float(d["reward"]), float(d["on"])))
                    if ok:
                        kept.append(i)
                        data.append(d)
                    else:
                        dropped += 1
                        log(f"[train] it {self.iteration + 1}: {b.labels[0].split('__')[0]} window gradient "
                            f"{w_norm:.3g} (loss {loss:.4g}) dropped")
                    # Rows with finite states move on (a fresh burn-in with the same policy would replay the same
                    # window); rows with non-finite states start afresh.
                    end = [*d["end"]["state"].values(), d["end"]["obs"], d["end"]["I"]]
                    if bool(torch.stack([torch.isfinite(v).all() for v in end]).all()):
                        b.start = d["end"]
                    else:
                        self.burn_in(b)
                bursting = any(not math.isfinite(x) or x > cfg.burst * limit * c.S / n_scen
                               for x, c in zip(w_norms, chosen))
                if cfg.anneal and bursting and self.kappa > 0.0:
                    self.kappa = self.kappa / 2 if self.kappa > 1 / 64 else 0.0
                    log(f"[train] it {self.iteration + 1}: gradient burst, feedback weight -> {self.kappa:.4g}")
                if not kept:
                    skipped += 1
                    streak += 1
                    log(f"[train] it {self.iteration + 1}: every window dropped, no step")
                    if streak >= 20:
                        report["stopped"] = "20 iterations in a row without a step"
                        break
                    if streak % 5 == 0 and snap is not None:  # the same policy would explode again
                        self._restore(snap)
                        log(f"[train] it {self.iteration + 1}: networks restored to the best training reward")
                    continue
                streak = 0
                norm = self.actor_step(frac, limit)
                c_loss, g_mean = self.critic_step(data, frac)
                self.iteration += 1
                # Collapse guard on the mean over buildings of their reward averages (with every building in every
                # iteration: the moving average of the mean training reward). It waits until every building has an
                # average, so a mean over a few buildings never becomes the best the full mean is held to.
                for i, _, r, _ in seen:
                    ema[i] = r if i not in ema else 0.9 * ema[i] + 0.1 * r
                avg = float(np.mean(list(ema.values())))
                if cooldown > 0:
                    cooldown -= 1
                elif len(ema) < len(self.buildings):
                    pass
                elif avg > best_ema:
                    best_ema, snap = avg, self._snapshot()
                elif self.iteration > 20 and avg < 1.5 * best_ema and snap is not None:
                    self._restore(snap)
                    self.lr_scale *= 0.5
                    rollbacks += 1
                    log(f"[train] it {self.iteration}: reward average {avg:.5f} vs best {best_ema:.5f}: collapse, "
                        f"networks restored to that best, lr scale {self.lr_scale}")
                    ema, cooldown = {}, 10
                row = {"it": self.iteration, "min": elapsed / 60.0, "loss": float(np.mean([x[1] for x in seen])),
                       "reward": float(np.mean([x[2] for x in seen])), "grad_norm": norm, "clip": limit,
                       "critic_loss": c_loss, "td_target_mean": g_mean,
                       "kp": float(torch.exp(self.ctrl.log_kp.detach())), "ki": float(torch.exp(self.ctrl.log_ki.detach())),
                       "on_frac": float(np.mean([x[3] for x in seen])), "rollbacks": rollbacks, "buildings": picked,
                       "window_norms": w_norms, "dropped": dropped, "feedback": self.kappa,
                       "s": time.time() - t_it,
                       "env_steps_per_s": sum(b.rows for b in chosen) * cfg.h / (time.time() - t_it),
                       "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if self.device.startswith("cuda") else 0}
                flog.write(json.dumps(row) + "\n")
                flog.flush()
                if self.iteration % 20 == 1:
                    log(f"[train] it {row['it']} {row['min']:.1f} min: reward {row['reward']:.5f} loss {row['loss']:.5f} "
                        f"grad {norm:.3g} (clip {limit:.3g}) critic {c_loss:.3g} G {g_mean:.3f} on {row['on_frac']:.2f} "
                        f"Kp {row['kp']:.3f} Ki {row['ki']:.4f} | {row['s']:.2f} s/it {row['env_steps_per_s']:,.0f} env-steps/s "
                        f"{row['mem_gb']:.1f} GB | {dropped} windows dropped, feedback {self.kappa:.4g}")
                if time.time() - t_eval >= cfg.eval_minutes * 60.0:
                    self._eval_and_keep(report, elapsed)
                    t_eval = time.time()
                if time.time() - t_ckpt >= 15 * 60.0:
                    self.save(out / "last.pt", elapsed / 60.0)
                    t_ckpt = time.time()
        elapsed = time.time() - t_start + self.minutes_done * 60.0
        self._eval_and_keep(report, elapsed)
        self.save(out / "last.pt", elapsed / 60.0)
        report.update(iterations=self.iteration, minutes=elapsed / 60.0, best=self.best, rollbacks=rollbacks,
                      dropped=dropped, skipped=skipped, feedback=self.kappa)
        if self.best_ctrl is not None and self.val:  # the kept policy on every held-out scenario
            self.ctrl.load_state_dict(self.best_ctrl)
            report["kept"] = self.evaluate(self.val, cfg.eval_steps)
            with open(out / "eval_log.jsonl", "a") as f:
                f.write(json.dumps({**report["kept"], "kept": True}) + "\n")
            log(f"[eval] kept policy: all held-out {report['kept']['mean']:.5f}, {cfg.select or 'key'} "
                f"{report['kept']['key']:.5f} | " + " ".join(
                    f"{k}:{v['score']:.5f}" for k, v in report["kept"].items() if isinstance(v, dict)))
        return report

    def _snapshot(self) -> dict:
        return copy.deepcopy({"ctrl": self.ctrl.state_dict(), "critic": self.critic.state_dict(),
                              "target": self.target.state_dict(), "opt_a": self.opt_a.state_dict(),
                              "opt_c": self.opt_c.state_dict()})

    def _restore(self, snap: dict) -> None:
        self.ctrl.load_state_dict(snap["ctrl"])
        self.critic.load_state_dict(snap["critic"])
        self.target.load_state_dict(snap["target"])
        self.opt_a.load_state_dict(snap["opt_a"])
        self.opt_c.load_state_dict(snap["opt_c"])

    def _eval_and_keep(self, report: dict, elapsed: float) -> None:
        try:
            ev = self.evaluate(self.scored, self.cfg.eval_steps)
        except Exception as e:  # a failed evaluation must not end the training
            self.log(f"[eval] failed: {type(e).__name__}: {e}")
            return
        ev.update(it=self.iteration, min=elapsed / 60.0)
        report["evals"].append(ev)
        with open(self.out / "eval_log.jsonl", "a") as f:
            f.write(json.dumps(ev) + "\n")
        self.stale += 1
        if ev["key"] < self.best:
            self.best, self.stale = ev["key"], 0
            self.best_ctrl = copy.deepcopy(self.ctrl.state_dict())
            save_policy(self.ctrl, self.out / "d1_policy.pt")
        per = " ".join(f"{k}:{v['score']:.5f}" for k, v in ev.items() if isinstance(v, dict))
        self.log(f"[eval] it {self.iteration} {elapsed / 60.0:.1f} min: {'held-out' if self.val else 'train'} "
                 f"{self.cfg.select or 'mean'} {ev['key']:.5f} (best {self.best:.5f}) | {per}")


# ---- start-up checks -------------------------------------------------------------------------------------------

def parity_check(ctrl: Controller, rep: dict, device: str, n_steps: int, tmp: Path) -> dict:
    """The submission path (KagglePolicy on the organisers' simulator, CPU) and the training path (Controller.act on
    the device) given the same observations must act the same."""
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
    lay = Layout(meta).tensors(device)
    I = torch.zeros(1, lay["V"], device=device)
    worst_c, worst_d = 0.0, 0
    with torch.no_grad():
        for o, a in zip(seen, acts):
            a_c, a_d, I = ctrl.act(torch.as_tensor(o, device=device)[None], I, lay)
            worst_c = max(worst_c, float(np.abs(a_c[0].clamp(-1, 1).cpu().numpy() - a["continuous_action"]).max()))
            worst_d = max(worst_d, int(np.abs(a_d[0].round().cpu().numpy() - a["discrete_action"]).max()))
    return {"steps": n_steps, "cont_max_abs": worst_c, "disc_mismatch": worst_d, "ok": worst_c < 1e-4 and worst_d == 0}


def load_buildings(caches: list[Path], row_budget: int, h: int, device: str) -> list[Building]:
    """Caches (named <building>__<weather>.pkl by notebook 20) grouped by building, one Building each."""
    groups: dict[str, list[dict]] = {}
    for p in sorted(caches):
        rep = load_replica(p)
        rep["info"]["label"] = p.stem
        groups.setdefault(p.stem.split("__")[0], []).append(rep)
    return [Building(reps, row_budget, h, device) for _, reps in sorted(groups.items())]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--caches", required=True, help="comma-separated replica caches (.pkl) to train on")
    ap.add_argument("--val-caches", default="", help="comma-separated caches of held-out buildings (scored only)")
    ap.add_argument("--resume", type=Path, help="a last.pt to continue from")
    ap.add_argument("--smoke", action="store_true", help="start-up checks and a short run")
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
    try:
        buildings = load_buildings(caches, cfg.row_budget, cfg.h, device)
        val = load_buildings(val_caches, 0, cfg.h, device)
        trainer = Trainer(cfg, buildings, val, device, out, log)
        if a.resume:
            trainer.load(a.resume)
        held = torch.cuda.memory_allocated() / 2 ** 30 if device == "cuda" else 0.0
        log(f"[train] {len(buildings)} building(s): " + "; ".join(
            f"{'/'.join(b.labels)} V={b.lay['V']} A={b.lay['A']} rows={b.rows} sweeps={b.env.rom.sweeps}"
            for b in buildings) + f" | setup {time.time() - t0:.0f}s, {held:.2f} GB held")
        if a.smoke:
            # The organisers' simulator needs the scenario's config.gin; generated scenarios come as caches only.
            rep = next(r for b in buildings for r in b.reps if Path(r["info"]["config"]).exists())
            report["parity"] = parity_check(trainer.ctrl, rep, device, 36, out / "smoke_policy.pt")
            log(f"[smoke] parity on {rep['info']['label']}: {report['parity']}")
            report["eval_288"] = trainer.evaluate(trainer.scored, 288)
            log(f"[smoke] replica eval over 288 steps: {report['eval_288']['key']:.5f}")
        report.update(trainer.train())
    except torch.cuda.OutOfMemoryError:
        # A retry in the same process keeps the failed attempt's memory (V2, 2026-10-22): start afresh with half the
        # rows, from the last checkpoint when there is one.
        if cfg.row_budget < 200_000:
            raise
        log(f"[train] out of memory: restarting with row budget {cfg.row_budget // 2}")
        argv = [*sys.argv[1:], "--row-budget", str(cfg.row_budget // 2)]
        if (out / "last.pt").exists():
            argv += ["--resume", str(out / "last.pt")]
        os.execv(sys.executable, [sys.executable, "-m", "hvac_controller.d1.train", *argv])
    report["seconds"] = time.time() - t0
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    log(f"[train] done: best replica score {report.get('best')} after {report.get('iterations')} iterations")


if __name__ == "__main__":
    main()
