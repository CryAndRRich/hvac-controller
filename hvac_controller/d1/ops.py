"""A thin array namespace, so the plant equations run unchanged on NumPy (validation) and PyTorch (training).

Only what plant.py and the ROM step need. Scalars may be mixed with arrays everywhere; they are converted to the
backend's dtype and device first, because torch.minimum and friends do not take Python numbers.
"""

from __future__ import annotations

import numpy as np


class Ops:
    def __init__(self, backend: str = "numpy", dtype=None, device=None):
        self.backend = backend
        if backend == "numpy":
            self.dtype = np.dtype(dtype or np.float64)
            self.device = None
        elif backend == "torch":
            import torch

            self.torch = torch
            self.dtype = dtype or torch.float64
            self.device = device or "cpu"
        else:
            raise ValueError(backend)

    @property
    def is_torch(self) -> bool:
        return self.backend == "torch"

    # ---- construction -------------------------------------------------------------------------------------------
    def asarray(self, x):
        if self.is_torch:
            t = self.torch
            if isinstance(x, t.Tensor):
                return x.to(dtype=self.dtype, device=self.device)
            return t.as_tensor(np.asarray(x, dtype=np.float64), dtype=self.dtype, device=self.device)
        return np.asarray(x, dtype=self.dtype)

    def index(self, x):
        """An integer index array."""
        if self.is_torch:
            return self.torch.as_tensor(np.asarray(x, dtype=np.int64), device=self.device)
        return np.asarray(x, dtype=np.int64)

    def zeros(self, shape):
        if self.is_torch:
            return self.torch.zeros(shape, dtype=self.dtype, device=self.device)
        return np.zeros(shape, dtype=self.dtype)

    def full(self, shape, value):
        if self.is_torch:
            return self.torch.full(shape, float(value), dtype=self.dtype, device=self.device)
        return np.full(shape, value, dtype=self.dtype)

    def eye(self, n):
        if self.is_torch:
            return self.torch.eye(n, dtype=self.dtype, device=self.device)
        return np.eye(n, dtype=self.dtype)

    def to_numpy(self, x) -> np.ndarray:
        if self.is_torch and isinstance(x, self.torch.Tensor):
            return x.detach().cpu().to(self.torch.float64).numpy()
        return np.asarray(x, dtype=np.float64)

    def _t(self, x):
        return x if (not self.is_torch or isinstance(x, self.torch.Tensor)) else self.asarray(x)

    # ---- elementwise ----------------------------------------------------------------------------------------------
    def where(self, cond, a, b):
        if self.is_torch:
            if not isinstance(cond, self.torch.Tensor):
                cond = self.torch.as_tensor(cond, device=self.device)
            return self.torch.where(cond.bool(), self._t(a), self._t(b))
        return np.where(cond, a, b)

    def amin(self, x, axis):
        return x.amin(dim=axis) if self.is_torch else x.min(axis=axis)

    def amax(self, x, axis):
        return x.amax(dim=axis) if self.is_torch else x.max(axis=axis)

    def minimum(self, a, b):
        return self.torch.minimum(self._t(a), self._t(b)) if self.is_torch else np.minimum(a, b)

    def maximum(self, a, b):
        return self.torch.maximum(self._t(a), self._t(b)) if self.is_torch else np.maximum(a, b)

    def clip(self, x, lo, hi):
        return self.minimum(self.maximum(x, lo), hi)

    def relu(self, x):
        return self.maximum(x, 0.0)

    def sqrt(self, x):
        return self.torch.sqrt(self._t(x)) if self.is_torch else np.sqrt(x)

    def abs(self, x):
        return self.torch.abs(self._t(x)) if self.is_torch else np.abs(x)

    def sigmoid(self, x):
        return self.torch.sigmoid(self._t(x)) if self.is_torch else 1.0 / (1.0 + np.exp(-x))

    def log(self, x):
        return self.torch.log(self._t(x)) if self.is_torch else np.log(x)

    def round_f32(self, x):
        """Rounds to float32 and back: what storing a value in a protobuf `float` field does."""
        if self.is_torch:
            return self._t(x).to(self.torch.float32).to(self.dtype)
        return np.asarray(x, dtype=np.float32).astype(self.dtype)

    # ---- reductions and linear algebra ----------------------------------------------------------------------------
    def sum(self, x, axis):
        return x.sum(dim=axis) if self.is_torch else x.sum(axis=axis)

    def stack(self, xs, axis):
        return self.torch.stack(xs, dim=axis) if self.is_torch else np.stack(xs, axis=axis)

    def concatenate(self, xs, axis):
        return self.torch.cat(xs, dim=axis) if self.is_torch else np.concatenate(xs, axis=axis)

    def solve(self, a, b):
        """Batched a x = b with a (B, n, n) and b (B, n)."""
        if self.is_torch:
            return self.torch.linalg.solve(a, b.unsqueeze(-1)).squeeze(-1)
        return np.linalg.solve(a, b[..., None])[..., 0]
