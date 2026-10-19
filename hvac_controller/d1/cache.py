"""One scenario's replica inputs, built once on a CPU and read back by every training and evaluation session.

``save_replica`` loads the scenario, builds the spec, the ROM and the exogenous series (torch_env.replica_from_env;
the ROM of an 85-zone building takes about 4 minutes) and pickles them together with the policy metadata and the
action spec the organisers' gateway hands a policy, and writes the build information next to it as JSON.
``load_replica`` checks the format version.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

VERSION = 1


def save_replica(config: Path, path: Path, n_steps: int = 8640) -> dict:
    from hvac_controller.d1.ops import Ops
    from hvac_controller.d1.plant import Plant
    from hvac_controller.d1.torch_env import ThermalRom, replica_from_env
    from hvac_controller.sim import load_env

    t0 = time.time()
    env, meta = load_env(Path(config))
    spec, rom, exo, _, timings = replica_from_env(env, n_steps)
    thermal = ThermalRom([rom], Ops("numpy", dtype="float32"), Plant(spec, Ops("numpy")).max_air_conductance())
    info = {"config": str(config), "label": spec.label, "n_zones": spec.n_zones, "n_vavs": spec.n_vavs,
            "n_ahus": spec.n_ahus, "n_obs": len(spec.obs), "order": rom.order, "h": rom.h, "jacobi_q": thermal.q,
            "sweeps_float32": thermal.sweeps, "n_steps": exo.n_steps, **timings, "seconds": time.time() - t0}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"version": VERSION, "spec": spec, "rom": rom, "exo": exo, "meta": meta,
                     "action_spec": env.action_spec(), "info": info}, f)
    path.with_suffix(".json").write_text(json.dumps(info, indent=1, default=str))
    return info


def load_replica(path: Path) -> dict:
    with open(path, "rb") as f:
        rep = pickle.load(f)
    if rep.get("version") != VERSION:
        raise ValueError(f"{path}: cache version {rep.get('version')}, this code reads {VERSION}; rebuild it")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", type=Path, help="output .pkl")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=8640)
    a = ap.parse_args()
    info = save_replica(a.config, a.path, a.steps)
    print(f"[cache] {info['label']}: Z={info['n_zones']} V={info['n_vavs']} A={info['n_ahus']} r={info['order']} "
          f"q={info['jacobi_q']:.3f} sweeps={info['sweeps_float32']} | ROM {info['rom_s']:.0f}s exo {info['exo_s']:.0f}s "
          f"total {info['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
