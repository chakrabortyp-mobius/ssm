"""
utils/adam.py
=============
Adam optimizer for a single NumPy parameter array.

Each trainable weight in the SSM (W_in, A_log, etc.) gets its own
AdamParam instance. This keeps optimizer state (m, v, t) co-located
with the parameter it manages, making it easy to add/remove parameters
without touching a central optimizer loop.

Usage
-----
    param = AdamParam(np.random.randn(16, 32), lr=1e-3)
    grad  = compute_gradient(...)       # same shape as param.data
    param.step(grad)                    # in-place update of param.data
    value = param.data                  # read updated weights
"""

import numpy as np


class AdamParam:
    """
    Single Adam-managed parameter.

    Parameters
    ----------
    data  : np.ndarray    initial weight values (copied internally)
    lr    : float         learning rate α  (default 1e-3)
    b1    : float         first moment decay  (default 0.9)
    b2    : float         second moment decay (default 0.999)
    eps   : float         denominator epsilon (default 1e-8)
    clip  : float | None  gradient clip norm; None = no clipping

    Attributes
    ----------
    data  : np.ndarray    current weight values (updated in-place by step())
    t     : int           number of update steps taken
    """

    def __init__(
        self,
        data: np.ndarray,
        lr:   float = 1e-3,
        b1:   float = 0.9,
        b2:   float = 0.999,
        eps:  float = 1e-8,
        clip: float = 1.0,
    ):
        self.data = data.copy().astype(np.float64)
        self.lr   = lr
        self.b1   = b1
        self.b2   = b2
        self.eps  = eps
        self.clip = clip

        # Optimizer state
        self.m = np.zeros_like(self.data)   # first moment  (mean)
        self.v = np.zeros_like(self.data)   # second moment (uncentered variance)
        self.t = 0                          # step counter

    def step(self, grad: np.ndarray) -> None:
        """
        Apply one Adam update step.

        grad : np.ndarray — gradient of the loss w.r.t. this parameter,
                            same shape as self.data.
        """
        if grad.shape != self.data.shape:
            raise ValueError(
                f"Gradient shape {grad.shape} != parameter shape {self.data.shape}"
            )

        # Optional gradient clipping (element-wise hard clip)
        if self.clip is not None:
            grad = np.clip(grad, -self.clip, self.clip)

        self.t += 1

        # Biased moment estimates
        self.m = self.b1 * self.m + (1.0 - self.b1) * grad
        self.v = self.b2 * self.v + (1.0 - self.b2) * grad ** 2

        # Bias-corrected estimates
        m_hat = self.m / (1.0 - self.b1 ** self.t)
        v_hat = self.v / (1.0 - self.b2 ** self.t)

        # Parameter update
        self.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self) -> None:
        """Reset optimizer state (useful when re-training)."""
        self.m = np.zeros_like(self.data)
        self.v = np.zeros_like(self.data)
        self.t = 0

    def __repr__(self) -> str:
        return (
            f"AdamParam(shape={self.data.shape}, lr={self.lr}, "
            f"steps={self.t})"
        )
