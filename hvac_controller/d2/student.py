"""Direction 2: the student, a small network that makes the teacher's supervisory decisions from the features alone.

No model, no forecast, no planning at run time: every ``rp`` steps the network maps the building's features
(features.py) to the six decisions of the low-level controller (lowlevel.py, with sequenced heating as the teacher
used), which then runs every step as in the MPC. So it costs about a millisecond a step and needs no ROM build.

Training (``train``) is behaviour cloning on the teacher's labels (teacher.py): binary cross-entropy for the run
decision (the teacher's run command is on or off), squared error for the five continuous ones, features
standardised with the training set's statistics, scenarios held out whole for validation. ``StudentPolicy`` is the
policy the evaluation harness and the gateway run (``d2s:<weights.pt>`` in d1/evaluate.py).

    python -m hvac_controller.d2.student train OUT.pt --data DIR [DIR ...] --epochs 60
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hvac_controller.d2.features import N_FEATURES, NAMES, FeatureBuilder
from hvac_controller.d2.lowlevel import N_DEC, LowLevel


class StudentNet(nn.Module):
    def __init__(self, n_in: int = N_FEATURES, width: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                  nn.Linear(width, N_DEC))
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits of the six decisions."""
        return self.body((x - self.mean) / self.std)


def load_student(path) -> tuple[StudentNet, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if tuple(ck["features"]) != NAMES:
        raise ValueError(f"{path}: trained on other features than features.py makes")
    net = StudentNet(len(NAMES), ck["width"])
    net.load_state_dict(ck["state_dict"])
    return net.eval(), ck


class StudentPolicy:
    def __init__(self, meta: dict, action_spec, weights):
        torch.set_num_threads(1)
        self.net, ck = load_student(weights)
        self.rp = int(ck.get("rp", 12))
        self.ll = LowLevel(meta, heat_seq=bool(ck.get("hs", 1)))
        self.fb = FeatureBuilder(meta)
        self.action_spec = action_spec
        self.integral = np.zeros((1, self.ll.n_int), dtype=np.float32)
        self.k, self.dec = 0, None

    def decide(self, f: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(self.net(torch.as_tensor(f, dtype=torch.float32)[None]))[0].numpy()

    def get_action(self, obs) -> dict:
        o = np.asarray(obs, dtype=np.float32)
        f = self.fb.update(o)
        if self.k % self.rp == 0 or self.dec is None:
            self.dec = self.decide(f)
        a_c, a_d, self.integral = self.ll.act(o[None], self.integral, self.dec[None].astype(np.float32))
        self.k += 1
        space = self.action_spec
        return {"continuous_action": np.clip(a_c[0], space["continuous_action"].low,
                                             space["continuous_action"].high).astype(np.float32),
                "discrete_action": np.clip(a_d[0].round(), space["discrete_action"].low,
                                           space["discrete_action"].high).astype(np.int32)}


# ---- training ------------------------------------------------------------------------------------------------
def _load(dirs: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, Y, G = [], [], []
    files = sorted(p for d in dirs for p in Path(d).rglob("*.npz"))
    for g, p in enumerate(files):
        z = np.load(p)
        X.append(z["X"]), Y.append(z["Y"]), G.append(np.full(len(z["X"]), g))
    if not X:
        raise FileNotFoundError(f"no .npz under {dirs}")
    return np.concatenate(X), np.concatenate(Y), np.concatenate(G)


def train(out: Path, dirs: list[Path], epochs: int = 60, width: int = 128, lr: float = 1e-3, batch: int = 512,
          val_frac: float = 0.1, seed: int = 0, run_weight: float = 1.0) -> dict:
    torch.manual_seed(seed)
    X, Y, G = _load(dirs)
    rng = np.random.default_rng(seed)
    groups = np.unique(G)
    val_g = rng.choice(groups, max(1, int(len(groups) * val_frac)), replace=False)
    val = np.isin(G, val_g)
    net = StudentNet(X.shape[1], width)
    net.mean.copy_(torch.as_tensor(X[~val].mean(0)))
    net.std.copy_(torch.as_tensor(X[~val].std(0) + 1e-3))
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(np.clip(Y, 0.0, 1.0))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()
    tr_idx = np.flatnonzero(~val)

    def loss_of(xb, yb):
        z = net(xb)
        run = (yb[:, 0] > 0.5).float()
        return run_weight * bce(z[:, 0], run) + mse(torch.sigmoid(z[:, 1:]), yb[:, 1:])

    t0, best, best_state = time.time(), float("inf"), None
    for ep in range(epochs):
        net.train()
        rng.shuffle(tr_idx)
        for i in range(0, len(tr_idx), batch):
            b = torch.as_tensor(tr_idx[i:i + batch])
            loss = loss_of(Xt[b], Yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            v = float(loss_of(Xt[val], Yt[val])) if val.any() else float("nan")
            z = net(Xt[val])
            run_acc = float(((z[:, 0] > 0) == (Yt[val][:, 0] > 0.5)).float().mean()) if val.any() else float("nan")
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[student] epoch {ep:3d} val loss {v:.4f} run acc {run_acc:.3f} ({time.time() - t0:.0f}s)",
                  flush=True)
    torch.save({"state_dict": best_state, "width": width, "features": list(NAMES), "rp": 12, "hs": 1,
                "n_train": int((~val).sum()), "n_val": int(val.sum()), "val_loss": best}, out)
    return {"out": str(out), "samples": len(X), "scenarios": len(groups), "val_loss": best}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("out", type=Path)
    t.add_argument("--data", type=Path, nargs="+", required=True)
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--width", type=int, default=128)
    a = ap.parse_args()
    if a.cmd == "train":
        print(train(a.out, a.data, a.epochs, a.width))


if __name__ == "__main__":
    main()
