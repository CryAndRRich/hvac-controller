"""Direction 2: a per-air-handler policy, cloned from the MPC teacher and then improved by PPO on the replica.

The MPC teacher (teacher.py) and the student (student.py) drive every air handler alike with six building-wide
decisions. This policy decides per air handler: each one sees the building's features (features.FeatureBuilder) and
its own zones' (``local``), and one network shared by all air handlers maps them to its five decisions (run, zone
target, supply temperature, recirculation, static pressure). The hot-water setpoint is held at 330 K: a policy that
moved it could learn the boiler's billing mismatch (forum threads 749531 and 752142), which the final re-run fixes.
The low-level controller (lowlevel.py, sequenced heating, per-AHU decisions) runs every step as in the MPC.

- ``bc``: behaviour cloning on the teacher's labels: every air handler gets the teacher's building-wide decision.
- ``train``: PPO. Each iteration rolls ``rows`` noisy copies of each of ``per_iter`` scenarios (d1 replica caches)
  for ``steps`` steps in worker processes, one decision an hour; then clipped-surrogate updates with GAE over the
  hourly rewards. Stochastic actions are Gaussian in the decisions' logit space.
- ``RLPolicy``: the deterministic policy (the mean action) for the evaluation harness (``d2r:<weights.pt>``) and the
  gateway.

    python -m hvac_controller.d2.rl bc BC.pt --data LABEL_DIR
    python -m hvac_controller.d2.rl train OUT_DIR --caches CACHE_DIR --init BC.pt --minutes 600
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hvac_controller.d2.features import LOCAL_NAMES, N_FEATURES, N_LOCAL, NAMES, FeatureBuilder

N_ACT = 5  # per air handler: run, target, sat, recirc, pressure (lowlevel.DEC[:5])
WATER = 0.5  # decision for 330 K on the 310-350 K range, held
HOUR = 12


class ActorCritic(nn.Module):
    def __init__(self, width: int = 128):
        super().__init__()
        self.width = width
        self.register_buffer("g_mean", torch.zeros(N_FEATURES))
        self.register_buffer("g_std", torch.ones(N_FEATURES))
        self.register_buffer("l_mean", torch.zeros(N_LOCAL))
        self.register_buffer("l_std", torch.ones(N_LOCAL))
        self.ahu = nn.Sequential(nn.Linear(N_FEATURES + N_LOCAL, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.mu = nn.Linear(width, N_ACT)
        # Exploration in logit space; wider on the run decision, whose logit is far from 0 once cloned.
        self.log_std = nn.Parameter(torch.tensor([0.0, -1.5, -1.5, -1.5, -1.5]))
        # The critic has its own trunk: value errors are large early on and would otherwise reshape the actor's.
        self.critic_ahu = nn.Sequential(nn.Linear(N_FEATURES + N_LOCAL, width), nn.SiLU())
        self.critic = nn.Sequential(nn.Linear(N_FEATURES + width, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                    nn.Linear(width, 1))
        self.register_buffer("ret_scale", torch.ones(()))  # values are predicted in units of this return scale

    def forward(self, g: torch.Tensor, l: torch.Tensor):
        """g (B, F), l (B, A, F_local) -> action means (B, A, 5) in logit space, values (B,)."""
        gn = (g - self.g_mean) / self.g_std
        ln = (l - self.l_mean) / self.l_std
        A = l.shape[1]
        x = torch.cat([gn[:, None, :].expand(-1, A, -1), ln], 2)
        v = self.critic(torch.cat([gn, self.critic_ahu(x).mean(1)], 1))[:, 0]
        return self.mu(self.ahu(x)), v

    def log_prob(self, mu: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Log-density of logits z (B, A, 5) under N(mu, std), summed over air handlers and decisions."""
        std = self.log_std.exp()
        return (-0.5 * ((z - mu) / std) ** 2 - self.log_std - 0.5 * np.log(2 * np.pi)).sum((1, 2))


def decisions(z: np.ndarray, A: int) -> np.ndarray:
    """Logits (A, 5) -> the low-level controller's per-AHU decision vector [run | target | sat | recirc | pressure |
    water] (5A + 1)."""
    d = 1.0 / (1.0 + np.exp(-z))
    return np.concatenate([d.T.reshape(-1), [WATER]]).astype(np.float32)


def save_model(net: ActorCritic, path: Path, **extra) -> None:
    torch.save({"state_dict": net.state_dict(), "width": net.width, "features": list(NAMES),
                "local": list(LOCAL_NAMES), **extra}, path)


def load_model(path) -> ActorCritic:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if tuple(ck["features"]) != NAMES or tuple(ck["local"]) != LOCAL_NAMES:
        raise ValueError(f"{path}: trained on other features than features.py makes")
    net = ActorCritic(ck["width"])
    net.load_state_dict(ck["state_dict"])
    return net.eval()


class RLPolicy:
    """The gateway's policy: the mean action every hour, the low-level controller every step."""

    def __init__(self, meta: dict, action_spec, weights):
        from hvac_controller.d2.lowlevel import LowLevel

        torch.set_num_threads(1)
        self.net = load_model(weights)
        self.ll = LowLevel(meta, per_ahu=True, heat_seq=True)
        self.fb = FeatureBuilder(meta)
        self.action_spec = action_spec
        self.integral = np.zeros((1, self.ll.n_int), dtype=np.float32)
        self.k, self.dec = 0, None

    def get_action(self, obs) -> dict:
        o = np.asarray(obs, dtype=np.float32)
        f = self.fb.update(o)
        if self.k % HOUR == 0 or self.dec is None:
            with torch.no_grad():
                mu, _ = self.net(torch.as_tensor(f)[None], torch.as_tensor(self.fb.local())[None])
            self.dec = decisions(mu[0].numpy(), self.ll.A)
        a_c, a_d, self.integral = self.ll.act(o[None], self.integral, self.dec[None])
        self.k += 1
        s = self.action_spec
        return {"continuous_action": np.clip(a_c[0], s["continuous_action"].low, s["continuous_action"].high).astype(np.float32),
                "discrete_action": np.clip(a_d[0].round(), s["discrete_action"].low, s["discrete_action"].high).astype(np.int32)}


# ---- behaviour cloning -------------------------------------------------------------------------------------------
def bc(out: Path, dirs: list[Path], epochs: int = 60, width: int = 128, lr: float = 1e-3, seed: int = 0) -> dict:
    """Every air handler imitates the teacher's building-wide decision (its first five)."""
    torch.manual_seed(seed)
    data = []
    for p in sorted(q for d in dirs for q in Path(d).rglob("*.npz")):
        z = np.load(p)
        if "XL" in z.files:
            data.append((z["X"], z["XL"], np.clip(z["Y"][:, :N_ACT], 0.0, 1.0)))
    if not data:
        raise FileNotFoundError(f"no labels with per-air-handler features under {dirs}")
    rng = np.random.default_rng(seed)
    val = set(rng.choice(len(data), len(data) // 10, replace=False).tolist())  # whole scenarios held out
    net = ActorCritic(width)
    G = np.concatenate([d[0] for i, d in enumerate(data) if i not in val])
    L = np.concatenate([d[1].reshape(-1, N_LOCAL) for i, d in enumerate(data) if i not in val])
    net.g_mean.copy_(torch.as_tensor(G.mean(0))), net.g_std.copy_(torch.as_tensor(G.std(0) + 1e-3))
    net.l_mean.copy_(torch.as_tensor(L.mean(0))), net.l_std.copy_(torch.as_tensor(L.std(0) + 1e-3))
    opt = torch.optim.Adam([p for n, p in net.named_parameters() if not n.startswith("critic")], lr=lr)
    bce = nn.BCEWithLogitsLoss()

    def loss_of(g, l, y):
        mu, _ = net(torch.as_tensor(g), torch.as_tensor(l))
        y = torch.as_tensor(y)[:, None, :].expand_as(mu)
        return bce(mu[..., 0], (y[..., 0] > 0.5).float()) + ((torch.sigmoid(mu[..., 1:]) - y[..., 1:]) ** 2).mean()

    best, best_state, t0 = np.inf, None, time.time()
    for ep in range(epochs):
        net.train()
        for i in rng.permutation(len(data)):
            if i in val:
                continue
            g, l, y = data[i]
            loss = loss_of(g, l, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():  # too few scenarios to hold any out: select on the training loss
            v = float(np.mean([float(loss_of(*data[i])) for i in (val or range(len(data)))]))
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[bc] epoch {ep:3d} val {v:.4f} ({time.time() - t0:.0f}s)", flush=True)
    net.load_state_dict(best_state)
    save_model(net, out, bc_val=best)
    return {"out": str(out), "scenarios": len(data), "val": best}


# ---- PPO rollouts --------------------------------------------------------------------------------------------------
def rollout(cache: str, state: dict, width: int, rows: int, steps: int, seed: int) -> dict:
    """``rows`` copies of one scenario, each with its own action noise; one decision an hour per air handler."""
    from hvac_controller.d1.cache import load_replica
    from hvac_controller.d1.ops import Ops
    from hvac_controller.d1.torch_env import BuildingEnv
    from hvac_controller.d2.lowlevel import LowLevel

    torch.set_num_threads(1)
    net = ActorCritic(width)
    net.load_state_dict(state)
    rep = load_replica(Path(cache))
    meta = rep["meta"]
    env = BuildingEnv(rep["spec"], rep["rom"], rep["exo"], rows, Ops("numpy", dtype=np.float32), emulate_f32=False)
    ll = LowLevel(meta, per_ahu=True, heat_seq=True)
    A = ll.A
    fbs = [FeatureBuilder(meta) for _ in range(rows)]
    rng = np.random.default_rng(seed)
    std = net.log_std.exp().detach().numpy()
    obs = env.reset()
    integral = np.zeros((rows, ll.n_int), dtype=np.float32)
    n = min(steps, env.n_steps)
    H = n // HOUR
    G = np.zeros((H, rows, N_FEATURES), np.float32)
    L = np.zeros((H, rows, A, N_LOCAL), np.float32)
    Z = np.zeros((H, rows, A, N_ACT), np.float32)
    R = np.zeros((H, rows), np.float32)
    total = np.zeros(rows)
    dec = None
    for k in range(H * HOUR):
        o = np.asarray(obs, dtype=np.float32)
        feats = [fb.update(o[i]) for i, fb in enumerate(fbs)]
        if k % HOUR == 0:
            h = k // HOUR
            G[h] = np.stack(feats)
            L[h] = np.stack([fb.local() for fb in fbs])
            with torch.no_grad():
                mu, _ = net(torch.as_tensor(G[h]), torch.as_tensor(L[h]))
            Z[h] = mu.numpy() + std * rng.standard_normal(mu.shape)
            dec = np.stack([decisions(Z[h, i], A) for i in range(rows)])
        a_c, a_d, integral = ll.act(o, integral, dec)
        obs, reward, _, _ = env.step(np.clip(a_c, -1.0, 1.0), a_d)
        R[k // HOUR] += reward
        total += reward
    return {"cache": cache, "G": G, "L": L, "Z": Z, "R": R, "score": float(np.mean(np.abs(total) / (H * HOUR)))}


def _rollout_job(kw: dict) -> dict:
    try:
        return rollout(**kw)
    except Exception as e:
        import traceback

        return {"cache": kw["cache"], "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}


def ppo_update(net: ActorCritic, opt, trajs: list[dict], epochs: int = 4, clip: float = 0.2, gamma: float = 0.99,
               lam: float = 0.95, ent: float = 1e-3, vf: float = 0.5, max_kl: float = 0.03) -> dict:
    """Clipped-surrogate PPO with GAE on hourly rewards; one gradient step per scenario chunk. Values are learnt in
    units of a running return scale (net.ret_scale); the epochs stop early once the policy has moved ``max_kl``."""
    s_old = float(net.ret_scale)
    batches = []
    with torch.no_grad():
        for tr in trajs:
            H, rows = tr["R"].shape
            g = torch.as_tensor(tr["G"].reshape(H * rows, -1))
            l = torch.as_tensor(tr["L"].reshape(H * rows, *tr["L"].shape[2:]))
            z = torch.as_tensor(tr["Z"].reshape(H * rows, *tr["Z"].shape[2:]))
            mu, v = net(g, l)
            logp = net.log_prob(mu, z)
            v = v.reshape(H, rows).numpy() * s_old
            r = tr["R"]
            adv = np.zeros((H, rows), np.float32)
            last = np.zeros(rows, np.float32)
            for t in range(H - 1, -1, -1):  # the episode ends at H: no bootstrap
                nv = v[t + 1] if t + 1 < H else 0.0
                delta = r[t] + gamma * nv - v[t]
                last = delta + gamma * lam * last
                adv[t] = last
            ret = adv + v
            batches.append((g, l, z, logp, torch.as_tensor(adv.reshape(-1)), torch.as_tensor(ret.reshape(-1))))
    all_adv = torch.cat([b[4] for b in batches])
    a_mean, a_std = all_adv.mean(), all_adv.std() + 1e-8
    s_new = 0.9 * s_old + 0.1 * float(torch.cat([b[5] for b in batches]).std() + 1e-6) if s_old != 1.0 else \
        float(torch.cat([b[5] for b in batches]).std() + 1e-6)
    net.ret_scale.fill_(s_new)
    stats = {"pi": 0.0, "v": 0.0, "kl": 0.0, "n": 0, "epochs": 0}
    for _ in range(epochs):
        kl_epoch = []
        for i in np.random.permutation(len(batches)):
            g, l, z, logp_old, adv, ret = batches[i]
            adv = (adv - a_mean) / a_std
            mu, v = net(g, l)
            logp = net.log_prob(mu, z)
            ratio = torch.exp(logp - logp_old)
            pi_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
            v_loss = ((v - ret / s_new) ** 2).mean()
            loss = pi_loss + vf * v_loss - ent * net.log_std.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            stats["pi"] += float(pi_loss)
            stats["v"] += float(v_loss)
            kl = float((logp_old - logp).mean())
            kl_epoch.append(kl)
            stats["kl"] += kl
            stats["n"] += 1
        stats["epochs"] += 1
        if np.mean(kl_epoch) > max_kl:
            break
    n = max(stats.pop("n"), 1)
    ep = stats.pop("epochs")
    return {**{k: v / n for k, v in stats.items()}, "epochs": ep, "ret_scale": s_new}


def train(out_dir: Path, caches: list[Path], init: Path | None, minutes: float, rows: int = 8, steps: int = 2880,
          per_iter: int = 16, procs: int = 4, lr: float = 1e-4, seed: int = 0, width: int = 128) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    net = load_model(init) if init else ActorCritic(width)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    log, t0, it = [], time.time(), 0
    ctx = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(procs, mp_context=ctx) as pool:
        while (time.time() - t0) / 60.0 < minutes:
            chosen = rng.choice(len(caches), min(per_iter, len(caches)), replace=False)
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            jobs = [{"cache": str(caches[i]), "state": state, "width": net.width, "rows": rows, "steps": steps,
                     "seed": int(rng.integers(1 << 30))} for i in chosen]
            trajs = []
            for f in cf.as_completed([pool.submit(_rollout_job, j) for j in jobs]):
                r = f.result()
                if "error" in r:
                    print(f"[ppo] {Path(r['cache']).stem}: {r['error']}\n{r['traceback']}", flush=True)
                else:
                    trajs.append(r)
            if not trajs:
                break
            stats = ppo_update(net, opt, trajs)
            score = float(np.mean([t["score"] for t in trajs]))
            it += 1
            row = {"iter": it, "minutes": (time.time() - t0) / 60.0, "score": score,
                   "std": net.log_std.exp().mean().item(), **stats}
            log.append(row)
            print(f"[ppo] iter {it:4d} {row['minutes']:6.1f} min train score {score:.5f} std {row['std']:.3f} "
                  f"pi {stats['pi']:+.4f} v {stats['v']:.4f} kl {stats['kl']:+.4f}", flush=True)
            save_model(net, out_dir / "last.pt", iters=it)
            (out_dir / "log.json").write_text(json.dumps(log))
    return {"iters": it, "log": str(out_dir / "log.json")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bc")
    b.add_argument("out", type=Path)
    b.add_argument("--data", type=Path, nargs="+", required=True)
    b.add_argument("--epochs", type=int, default=60)
    t = sub.add_parser("train")
    t.add_argument("out_dir", type=Path)
    t.add_argument("--caches", type=Path, required=True)
    t.add_argument("--init", type=Path)
    t.add_argument("--minutes", type=float, default=60.0)
    t.add_argument("--rows", type=int, default=8)
    t.add_argument("--steps", type=int, default=2880)
    t.add_argument("--per-iter", type=int, default=16)
    t.add_argument("--procs", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    if a.cmd == "bc":
        print(bc(a.out, a.data, a.epochs))
    else:
        caches = sorted(q for q in a.caches.glob("*.pkl") if not q.stem.startswith("building_train_"))
        print(f"[ppo] {len(caches)} training scenarios", flush=True)
        print(train(a.out_dir, caches, a.init, a.minutes, a.rows, a.steps, a.per_iter, a.procs, a.lr))


if __name__ == "__main__":
    main()
