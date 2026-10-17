"""Helpers around the organisers' simulator: load an environment, snapshot it mid-episode."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

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


def _find_unpicklable(root, type_names=("SuperLU",)) -> list:
    """Objects of the given types reachable from root through attributes and containers."""
    found, seen, stack = [], set(), [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen or isinstance(obj, (np.ndarray, str, bytes, int, float, bool, type(None))):
            continue
        seen.add(id(obj))
        if type(obj).__name__ in type_names:
            found.append(obj)
            continue
        if isinstance(obj, dict):
            stack.extend(obj.keys())
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        else:
            d = getattr(obj, "__dict__", None)
            if d is not None:
                stack.extend(d.values())
            for slot in getattr(type(obj), "__slots__", ()):
                if hasattr(obj, slot):
                    stack.append(getattr(obj, slot))
    return found


def snapshot(*objs):
    """Deep copy of (env, policy, ...) that shares the sparse LU factorization with the original.

    The finite-difference solver caches ``scipy.sparse.linalg.splu`` of its matrix, which cannot be pickled or copied.
    The factorization is only read (``solve``) and is rebuilt if the convection coefficient changes, so sharing it
    between the original and the copy is safe.
    """
    memo = {id(lu): lu for lu in _find_unpicklable(objs)}
    return copy.deepcopy(objs, memo)
