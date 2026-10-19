"""Direction 1, step 6: SHAC through the replica (Xu et al., ICLR 2022), re-implemented from the paper.

Per building (one ``BuildingEnv`` grouping that building's scenarios, ``rows`` batch rows each spread over the
episode by a burn-in), every iteration:

1. a window of h steps from where the rows stopped: the policy acts, the replica steps, and the actor loss is
   -(1 / (B h)) sum over rows of [sum_t gamma^t r_t + gamma^h V'(s_h)], with V' the target critic; a row whose
   episode ends inside the window adds its finished return (the end is a real end: V = 0, so V' is only evaluated at
   the window's end) and restarts from the scenario's initial state, as the gateway's episodes do; one backward pass
   (gradients accumulate over buildings);
2. one actor step: gradient norm clipped, Adam (0.7, 0.95), learning rate decaying linearly over the time budget;
3. the critic fitted to TD(lambda) targets of the window on ``critic_rows`` fixed rows (spread over the episode by
   the burn-in; all h x B states would cost far more than the window), 16 passes x 4 shuffled minibatches, then the
   target critic follows it (Polyak).

The critic is a DeepSets over VAVs with privileged global inputs: the outside temperature and comfort flags of the
next 24 hours. With ``graph`` the window (forward and backward) of each building is captured once as a CUDA graph and
replayed: the step is launch-bound eagerly (about 14 ms whatever the batch); at start-up one window is run both ways
and the graph is used only if its loss and gradients match eager.

Scored in the replica every ``eval_minutes``: every scenario from its initial state for the whole episode, seed 0
plain and seeds >= 1 with observation noise of evaluate.SIGMA, on the training buildings and on held-out ones
(``--val-caches``). The best policy, by the held-out mean when there are held-out buildings, is kept
(``d1_policy.pt``, the file policy.KagglePolicy loads). Guards: a non-finite or huge gradient skips the step and halves the learning rate (three
in a row stop the run), non-finite states restart the rows, the time budget ends the run cleanly, ``last.pt`` is
written every 15 minutes and ``--resume`` continues from it.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
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
    rows: int = 2048  # batch rows per building
    minutes: float = 30.0  # training time budget
    lr_actor: float = 2e-3
    lr_critic: float = 5e-4
    clip: float = 1.0
    critic_iters: int = 16
    critic_batches: int = 4
    critic_rows: int = 256  # rows whose states the critic learns from
    target_keep: float = 0.2  # target <- keep * target + (1 - keep) * critic
    detach_obs: bool = False  # D.VA's decoupled gradient: no gradient through the observations
    eval_minutes: float = 10.0
    eval_seeds: int = 4
    eval_steps: int = 0  # 0: the whole episode
    width: int = 64
    seed: int = 0
    graph: bool = True


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

    def __init__(self, reps: list[dict], rows: int, device: str):
        spec = reps[0]["spec"]
        for r in reps:
            if _plant_key(r["spec"]) != _plant_key(spec):
                raise ValueError(f"{r['info']['label']} is not the same building as {reps[0]['info']['label']}")
            Layout(r["meta"]).check(r["spec"])  # every scenario's metadata against its own simulator objects
        self.reps, self.S, self.device = reps, len(reps), device
        self.labels = [r["info"]["label"] for r in reps]
        self.lay = Layout(reps[0]["meta"]).tensors(device)
        self.ops = Ops("torch", dtype=torch.float32, device=device)
        self.rows = max(rows - rows % self.S, self.S)
        self.env = self.make_env(self.rows)
        e = self.env
        self.fc_t = e.exo[:, e.exo_cols["t_amb"]]
        self.fc_c = e.exo[:, e.exo_cols["comfort"]]
        self.ahead = torch.arange(AHEAD, device=device) * int(round(3600 / spec.dt))
        self.mask = torch.as_tensor(obs_noise_mask(spec), dtype=torch.float32, device=device)
        self.critic_rows = None  # set by the trainer
        self.start = None
        self.graph = None

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
        b0 = buildings[0].reps[0]["spec"]
        self.ctrl = Controller(width=cfg.width, p_floor=float(np.max(b0.ahu["min_pressure"])),
                               max_recirc=float(np.min(b0.ahu["max_recirc"])),
                               min_damper=float(np.max(b0.vav["min_damper"]))).to(device)
        self.critic = Critic(cfg.width).to(device)
        self.target = Critic(cfg.width).to(device)
        self.target.load_state_dict(self.critic.state_dict())
        self.target.requires_grad_(False)
        self.opt_a = torch.optim.Adam(self.ctrl.parameters(), lr=cfg.lr_actor, betas=(0.7, 0.95))
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic, betas=(0.7, 0.95))
        self.iteration, self.lr_scale, self.best = 0, 1.0, math.inf
        self.minutes_done = 0.0  # training time of earlier sessions (resume)
        for b in buildings:
            b.critic_rows = torch.arange(0, b.rows, max(1, b.rows // cfg.critic_rows), device=device)[:cfg.critic_rows]

    # ---- features ---------------------------------------------------------------------------------------------
    def critic_inputs(self, b: Building, env, obs, I, rows):
        """Critic features of the given rows (all when rows is None); obs and I are already those rows'."""
        vav, glob, *_ = self.ctrl.features(obs, I, b.lay)
        k = env.k if rows is None else env.k[rows]
        s = env.row_s if rows is None else env.row_s[rows]
        return vav, torch.cat([glob, b.forecast(env.n_steps, s, k)], 1)

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
        critic's data (states s_0 .. s_h, rewards and episode ends of the critic's rows). Everything stays on the
        device: this is what a CUDA graph captures."""
        cfg, env, ctrl = self.cfg, b.env, self.ctrl
        env.state = dict(b.start["state"])
        env.k, env.obs, I = b.start["k"], b.start["obs"], b.start["I"]
        B = env.num_envs
        sel = b.critic_rows
        gamma_k = torch.ones(B, device=self.device)
        acc = torch.zeros(B, device=self.device)
        loss = torch.zeros((), device=self.device)
        buf = {"vav": [], "glob": [], "r": [], "done": []}
        reward = torch.zeros((), device=self.device)
        for t in range(cfg.h):
            cv, cg = self.critic_inputs(b, env, env.obs[sel], I[sel], sel)
            buf["vav"].append(cv.detach())
            buf["glob"].append(cg.detach())
            obs = env.obs.detach() if cfg.detach_obs else env.obs
            a_c, a_d, I = ctrl.act(obs, I, b.lay)
            _, r, done, _ = env.step(a_c, a_d)
            acc = acc + gamma_k * r
            reward = reward + r.detach().mean()
            buf["r"].append(r[sel].detach())
            buf["done"].append(done[sel])
            if t < cfg.h - 1:
                loss = loss - torch.where(done, acc, torch.zeros_like(acc)).sum()
                gamma_k = torch.where(done, torch.ones_like(gamma_k), gamma_k * cfg.gamma)
                acc = torch.where(done, torch.zeros_like(acc), acc)
                env.reset_done(done)
                I = torch.where(done[:, None], torch.zeros_like(I), I)
        # The window's end: bootstrap with the target critic (0 where the episode just ended), then restart those rows.
        cv, cg = self.critic_inputs(b, env, env.obs, I, None)
        v = torch.where(done, torch.zeros_like(acc), self.target(cv, cg))
        loss = loss - (acc + cfg.gamma * gamma_k * v).sum()
        buf["vav"].append(cv[sel].detach())
        buf["glob"].append(cg[sel].detach())
        env.reset_done(done)
        I = torch.where(done[:, None], torch.zeros_like(I), I)
        loss = loss / (B * cfg.h)
        (loss * weight).backward()
        out = {name: torch.stack(v) for name, v in buf.items()}
        out.update(loss=loss.detach(), reward=reward / cfg.h,
                   end={"state": {n: v.detach() for n, v in env.state.items()}, "obs": env.obs.detach(),
                        "I": I.detach(), "k": env.k})
        return out

    def _advance_start(self, b: Building, end: dict) -> None:
        if b.graph is None:
            b.start = end
            return
        for n, v in b.start["state"].items():
            v.copy_(end["state"][n])
        for n in ("obs", "I", "k"):
            b.start[n].copy_(end[n])

    def _weight(self, b: Building) -> float:
        return b.rows / sum(x.rows for x in self.buildings)

    def capture(self, b: Building) -> dict:
        """Captures b's window as a CUDA graph if it reproduces eager; returns the comparison."""
        params = list(self.ctrl.parameters())
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        start = {"state": {n: v.clone() for n, v in b.start["state"].items()},
                 **{n: b.start[n].clone() for n in ("obs", "I", "k")}}
        b.start = start
        self.opt_a.zero_grad(set_to_none=False)
        ref = self.window(b, 1.0)
        ref_grad = [p.grad.clone() for p in params]
        ref_loss = float(ref["loss"])
        del ref
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self.window(b, 1.0)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            self.opt_a.zero_grad(set_to_none=False)
            with torch.cuda.graph(graph):
                out = self.window(b, self._weight(b))
            self.opt_a.zero_grad(set_to_none=False)
            graph.replay()
            torch.cuda.synchronize()
            w = self._weight(b)
            diff = torch.sqrt(sum(((p.grad - w * g) ** 2).sum() for p, g in zip(params, ref_grad)))
            g_err = float(diff / torch.sqrt(sum(((w * g) ** 2).sum() for g in ref_grad)).clamp(min=1e-30))
            l_err = abs(float(out["loss"]) - ref_loss) / max(abs(ref_loss), 1e-30)
            ok = g_err < 1e-4 and l_err < 1e-5
        except Exception as e:  # capture not possible here: eager it is
            self.opt_a.zero_grad(set_to_none=False)
            return {"graph": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        self.opt_a.zero_grad(set_to_none=False)
        if ok:
            b.graph, b.graph_out = graph, out
        return {"graph": ok, "grad_rel_err": g_err, "loss_rel_err": l_err}

    def run_window(self, b: Building) -> dict:
        if b.graph is not None:
            b.graph.replay()
            out = b.graph_out
        else:
            out = self.window(b, self._weight(b))
        return out

    # ---- updates ----------------------------------------------------------------------------------------------
    def actor_step(self, frac: float, losses: list[float]) -> tuple[float, bool]:
        """Clips and steps; no step (False) when a loss or the gradient norm is non-finite or huge."""
        norm = float(torch.nn.utils.clip_grad_norm_(self.ctrl.parameters(), self.cfg.clip))
        if not math.isfinite(norm) or norm > 1e6 or not all(map(math.isfinite, losses)):
            return norm, False
        for g in self.opt_a.param_groups:
            g["lr"] = self.cfg.lr_actor * self.lr_scale * max(1.0 - frac, 0.01)
        self.opt_a.step()
        return norm, True

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
        total = 0.0
        for _ in range(cfg.critic_iters):
            for X_v, X_g, Y in sets:
                for idx in torch.randperm(len(Y), device=self.device).chunk(cfg.critic_batches):
                    loss = ((self.critic(X_v[idx], X_g[idx]) - Y[idx]) ** 2).mean()
                    self.opt_c.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.clip)
                    self.opt_c.step()
                    total += float(loss)
        with torch.no_grad():
            for t, c in zip(self.target.parameters(), self.critic.parameters()):
                t.mul_(cfg.target_keep).add_(c, alpha=1.0 - cfg.target_keep)
        return (total / (cfg.critic_iters * cfg.critic_batches * len(sets)),
                float(torch.cat([Y for _, _, Y in sets]).mean()))

    # ---- scoring in the replica -------------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, n_steps: int = 0) -> dict:
        seeds, out = self.cfg.eval_seeds, {}
        for b in self.buildings + self.val:
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
                              "carbon": float(rows[:, 3].mean()), "steps": n, "held_out": b in self.val}
        train = [v["score"] for v in out.values() if not v["held_out"]]
        held = [v["score"] for v in out.values() if v["held_out"]]
        out["mean"] = float(np.mean(train))
        out["val_mean"] = float(np.mean(held)) if held else None
        return out

    # ---- checkpoints ------------------------------------------------------------------------------------------
    def save(self, path: Path, minutes: float) -> None:
        torch.save({"cfg": asdict(self.cfg), "ctrl": self.ctrl.state_dict(), "ctrl_config": self.ctrl.config,
                    "critic": self.critic.state_dict(), "target": self.target.state_dict(),
                    "opt_a": self.opt_a.state_dict(), "opt_c": self.opt_c.state_dict(), "iteration": self.iteration,
                    "lr_scale": self.lr_scale, "best": self.best, "minutes": minutes}, path)

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device)
        self.ctrl.load_state_dict(ck["ctrl"])
        self.critic.load_state_dict(ck["critic"])
        self.target.load_state_dict(ck["target"])
        self.opt_a.load_state_dict(ck["opt_a"])
        self.opt_c.load_state_dict(ck["opt_c"])
        self.iteration, self.lr_scale, self.best = ck["iteration"], ck["lr_scale"], ck["best"]
        self.minutes_done = ck["minutes"]

    # ---- the loop ---------------------------------------------------------------------------------------------
    def train(self) -> dict:
        cfg, out, log = self.cfg, self.out, self.log
        report = {"setup": {}, "evals": []}
        t0 = time.time()
        for b in self.buildings:
            self.burn_in(b)
        report["setup"]["burn_in_s"] = time.time() - t0
        if cfg.graph and self.device.startswith("cuda"):
            report["setup"]["graph"] = {"/".join(b.labels): self.capture(b) for b in self.buildings}
            log(f"[train] CUDA graph: {report['setup']['graph']}")
        budget = cfg.minutes * 60.0
        t_start, t_ckpt, t_eval = time.time(), time.time(), time.time()
        bad = 0
        with open(out / "train_log.jsonl", "a") as flog:
            while True:
                elapsed = time.time() - t_start + self.minutes_done * 60.0
                if elapsed >= budget:
                    break
                frac = elapsed / budget
                t_it = time.time()
                self.opt_a.zero_grad(set_to_none=False)
                data, losses, rewards = [], [], []
                for b in self.buildings:
                    d = self.run_window(b)
                    data.append(d)
                    losses.append(float(d["loss"]))
                    rewards.append(float(d["reward"]))
                norm, stepped = self.actor_step(frac, losses)
                if not stepped:
                    bad += 1
                    self.lr_scale *= 0.5
                    log(f"[train] iteration {self.iteration}: bad gradient (norm {norm:.3g}, losses {losses}); "
                        f"step skipped, lr scale {self.lr_scale}")
                    for b in self.buildings:  # the rows may carry non-finite states: start them afresh
                        self._restart_rows(b)
                    if bad >= 3:
                        report["stopped"] = "three bad gradients in a row"
                        break
                    continue
                bad = 0
                for b, d in zip(self.buildings, data):
                    self._advance_start(b, d["end"])
                c_loss, g_mean = self.critic_step(data, frac)
                self.iteration += 1
                row = {"it": self.iteration, "min": elapsed / 60.0, "loss": float(np.mean(losses)),
                       "reward": float(np.mean(rewards)), "grad_norm": norm, "critic_loss": c_loss,
                       "td_target_mean": g_mean,
                       "kp": float(torch.exp(self.ctrl.log_kp)), "ki": float(torch.exp(self.ctrl.log_ki)),
                       "s": time.time() - t_it,
                       "env_steps_per_s": sum(b.rows for b in self.buildings) * cfg.h / (time.time() - t_it),
                       "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if self.device.startswith("cuda") else 0}
                flog.write(json.dumps(row) + "\n")
                flog.flush()
                if self.iteration % 20 == 1:
                    log(f"[train] it {row['it']} {row['min']:.1f} min: reward {row['reward']:.5f} loss {row['loss']:.5f} "
                        f"grad {norm:.3g} critic {c_loss:.3g} G {g_mean:.3f} Kp {row['kp']:.3f} "
                        f"Ki {row['ki']:.4f} | {row['s']:.2f} s/it {row['env_steps_per_s']:,.0f} env-steps/s "
                        f"{row['mem_gb']:.1f} GB")
                if time.time() - t_eval >= cfg.eval_minutes * 60.0:
                    self._eval_and_keep(report, elapsed)
                    t_eval = time.time()
                if time.time() - t_ckpt >= 15 * 60.0:
                    self.save(out / "last.pt", elapsed / 60.0)
                    t_ckpt = time.time()
        elapsed = time.time() - t_start + self.minutes_done * 60.0
        self._eval_and_keep(report, elapsed)
        self.save(out / "last.pt", elapsed / 60.0)
        report.update(iterations=self.iteration, minutes=elapsed / 60.0, best=self.best)
        return report

    def _restart_rows(self, b: Building) -> None:
        b.graph = None  # the graph's start buffers may be poisoned: back to eager from fresh start states
        self.burn_in(b)

    def _eval_and_keep(self, report: dict, elapsed: float) -> None:
        try:
            ev = self.evaluate(self.cfg.eval_steps)
        except Exception as e:  # a failed evaluation must not end the training
            self.log(f"[eval] failed: {type(e).__name__}: {e}")
            return
        ev.update(it=self.iteration, min=elapsed / 60.0)
        report["evals"].append(ev)
        with open(self.out / "eval_log.jsonl", "a") as f:
            f.write(json.dumps(ev) + "\n")
        key = ev["mean"] if ev["val_mean"] is None else ev["val_mean"]
        if key < self.best:
            self.best = key
            save_policy(self.ctrl, self.out / "d1_policy.pt")
        per = " ".join(f"{k}:{v['score']:.5f}" for k, v in ev.items() if isinstance(v, dict))
        val = "" if ev["val_mean"] is None else f" held-out {ev['val_mean']:.5f}"
        self.log(f"[eval] it {self.iteration} {elapsed / 60.0:.1f} min: train {ev['mean']:.5f}{val} (best {self.best:.5f}) | {per}")


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


def load_buildings(caches: list[Path], rows: int, device: str) -> list[Building]:
    """Caches (named <building>__<weather>.pkl by notebook 20) grouped by building, one Building each."""
    groups: dict[str, list[dict]] = {}
    for p in sorted(caches):
        rep = load_replica(p)
        rep["info"]["label"] = p.stem
        groups.setdefault(p.stem.split("__")[0], []).append(rep)
    return [Building(reps, rows, device) for _, reps in sorted(groups.items())]


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
    for attempt in range(4):  # out of memory at start-up: halve the rows
        try:
            buildings = load_buildings(caches, cfg.rows, device)
            val = load_buildings(val_caches, 1, device)
            trainer = Trainer(cfg, buildings, val, device, out, log)
            if a.resume:
                trainer.load(a.resume)
            log(f"[train] {len(buildings)} building(s): " + "; ".join(
                f"{'/'.join(b.labels)} V={b.lay['V']} A={b.lay['A']} rows={b.rows} sweeps={b.env.rom.sweeps}"
                for b in buildings) + f" | setup {time.time() - t0:.0f}s")
            if a.smoke:
                report["parity"] = parity_check(trainer.ctrl, buildings[0].reps[0], device, 36, out / "smoke_policy.pt")
                log(f"[smoke] parity: {report['parity']}")
                report["eval_288"] = trainer.evaluate(288)
                log(f"[smoke] replica eval over 288 steps: {report['eval_288']['mean']:.5f}")
            report.update(trainer.train())
            break
        except torch.cuda.OutOfMemoryError:
            cfg.rows //= 2
            report.setdefault("oom_halvings", []).append(cfg.rows)
            log(f"[train] out of memory: rows -> {cfg.rows}")
            buildings = val = trainer = None
            torch.cuda.empty_cache()
    report["seconds"] = time.time() - t0
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    log(f"[train] done: best replica score {report.get('best')} after {report.get('iterations')} iterations")


if __name__ == "__main__":
    main()
