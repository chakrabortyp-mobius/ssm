"""
utils/activations.py
====================
All activation functions and their gradients used by the SSM.
Pure NumPy. Single source of truth — import from here everywhere.

Functions
---------
sigmoid(x)          → σ(x)              forward
sigmoid_grad(x)     → σ(x)(1 - σ(x))   gradient
softplus(x)         → log(1 + e^x)      forward
softplus_grad(x)    → σ(x)              gradient (= sigmoid)
tanh_grad(x)        → 1 - tanh²(x)      gradient of tanh
"""

import numpy as np

# ── numerical clip to prevent overflow in exp/log ────────────────────────────
_CLIP = 30.0


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid: σ(x) = 1 / (1 + e^{-x})
    Uses two-branch formula to avoid overflow for both large positive
    and large negative inputs.
    """
    x = np.clip(x, -_CLIP, _CLIP)
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    """d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))"""
    s = sigmoid(x)
    return s * (1.0 - s)


def softplus(x: np.ndarray) -> np.ndarray:
    """
    softplus(x) = log(1 + e^x)
    Always positive → used as gate Δ to ensure Δ > 0.
    """
    return np.log1p(np.exp(np.clip(x, -_CLIP, _CLIP)))


def softplus_grad(x: np.ndarray) -> np.ndarray:
    """d/dx softplus(x) = sigmoid(x)"""
    return sigmoid(x)


def tanh_safe(x: np.ndarray) -> np.ndarray:
    """Clipped tanh for numerical safety."""
    return np.tanh(np.clip(x, -_CLIP, _CLIP))


def tanh_grad(x: np.ndarray) -> np.ndarray:
    """d/dx tanh(x) = 1 - tanh²(x)"""
    return 1.0 - tanh_safe(x) ** 2
