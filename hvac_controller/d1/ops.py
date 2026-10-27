"""A thin array namespace, so the plant equations run unchanged on NumPy (validation) and PyTorch (training).

Only what plant.py and the ROM step need. Python numbers may be mixed with arrays everywhere. On torch they go to
clamp / where as numbers rather than as new tensors, because building a tensor from a number copies it to the
device on every call.
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

    def _is_t(self, x) -> bool:
        return isinstance(x, self.torch.Tensor)

    # ---- elementwise ----------------------------------------------------------------------------------------------
    def where(self, cond, a, b):
        if self.is_torch:
            if not self._is_t(a) and not self._is_t(b):  # two numbers would give torch's default dtype
                a = self.torch.full_like(cond, a, dtype=self.dtype)
            return self.torch.where(cond, a, b)
        return np.where(cond, a, b)

    def amin(self, x, axis):
        return x.amin(dim=axis) if self.is_torch else x.min(axis=axis)

    def amax(self, x, axis):
        return x.amax(dim=axis) if self.is_torch else x.max(axis=axis)

    def minimum(self, a, b):
        if not self.is_torch:
            return np.minimum(a, b)
        if not self._is_t(b):
            return self.torch.clamp(a, max=b)
        if not self._is_t(a):
            return self.torch.clamp(b, max=a)
        return self.torch.minimum(a, b)

    def maximum(self, a, b):
        if not self.is_torch:
            return np.maximum(a, b)
        if not self._is_t(b):
            return self.torch.clamp(a, min=b)
        if not self._is_t(a):
            return self.torch.clamp(b, min=a)
        return self.torch.maximum(a, b)

    def clip(self, x, lo, hi):
        return self.minimum(self.maximum(x, lo), hi)

    def relu(self, x):
        return self.maximum(x, 0.0)

    def sqrt(self, x):
        return self.torch.sqrt(x) if self.is_torch else np.sqrt(x)

    def abs(self, x):
        return self.torch.abs(x) if self.is_torch else np.abs(x)

    def round_f32(self, x):
        """Rounds to float32 and back: what storing a value in a protobuf `float` field does."""
        if self.is_torch:
            return x.to(self.torch.float32).to(self.dtype)
        return np.asarray(x, dtype=np.float32).astype(self.dtype)

    # ---- reductions and linear algebra ----------------------------------------------------------------------------
    def sum(self, x, axis):
        return x.sum(dim=axis) if self.is_torch else x.sum(axis=axis)

    def concatenate(self, xs, axis):
        return self.torch.cat(xs, dim=axis) if self.is_torch else np.concatenate(xs, axis=axis)

    def solve(self, a, b):
        """Batched a x = b with a (B, n, n) and b (B, n)."""
        if self.is_torch:
            return self.torch.linalg.solve(a, b.unsqueeze(-1)).squeeze(-1)
        return np.linalg.solve(a, b[..., None])[..., 0]
