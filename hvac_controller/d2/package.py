"""Direction 2: the code a submission needs, as one zip to upload as a Kaggle dataset (Internet is off when scoring).

The policy imports only ``hvac_controller.d1`` (replica: spec, core, ROM, plant, ops, the policy Layout),
``hvac_controller.d2``, ``hvac_controller.sim`` and the K10 fallback; ``smart_buildings`` (the simulator the model is
rebuilt with) comes from the competition data itself. The zip holds the package at its top level, so the dataset folder goes on
sys.path as it is.

    python -m hvac_controller.d2.package /tmp/hvac-d2-code.zip
"""

from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ["hvac_controller/__init__.py", "hvac_controller/sim.py", "hvac_controller/baseline_k10.py",
         *[f"hvac_controller/d1/{m}.py" for m in ("__init__", "core", "rom", "plant", "spec", "torch_env", "ops",
                                                  "policy", "cache")],
         *[f"hvac_controller/d2/{m}.py" for m in ("__init__", "lowlevel", "mpc", "build", "submission", "lp",
                                                  "features", "student")]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--weights", type=Path, nargs="*", default=[], help="student weights to ship alongside")
    a = ap.parse_args()
    commit = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h %s"], capture_output=True,
                            text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "hvac_controller"],
                           capture_output=True, text=True).stdout.strip()
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(ROOT / f, f)
        for w in a.weights:
            z.write(w, f"weights/{w.name}")
        z.writestr("README.txt", f"hvac_controller (direction 2 MPC) at {commit}{' + uncommitted changes' if dirty else ''}\n")
    print(f"wrote {a.out} ({len(FILES)} files) from {commit}{' (dirty)' if dirty else ''}")


if __name__ == "__main__":
    main()
