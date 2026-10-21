"""Loading the organisers' simulator."""

from __future__ import annotations

from pathlib import Path

from smart_buildings import evaluate


def load_env(config_gin: Path):
    """A fresh environment and its policy metadata.

    Build a fresh one per episode: ``env.reset()`` does not restore every internal state (probe 2026-10-17:
    two rollouts after reset diverge from step 3, while two fresh environments are bit-identical).
    """
    env = evaluate.load_hybrid_environment(Path(config_gin))
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0  # as evaluate.py sets it
    meta = evaluate.create_building_meta_data(env, Path(config_gin).parent)
    return env, meta
