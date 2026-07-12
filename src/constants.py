
from __future__ import annotations

import numpy as np

def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable numpy softmax (was reimplemented in 7+ scripts)."""
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)
