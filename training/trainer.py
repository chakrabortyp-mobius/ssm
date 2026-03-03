"""
training/trainer.py
===================
Trainer — manages the SSM training loop.

Separated from the model so you can:
  - Swap training strategies (mini-batch, full-sequence, windowed)
    without touching model code
  - Add early stopping / LR scheduling without re-writing the SSM
  - Inspect loss curves and gradient norms per-epoch for debugging

Current strategy: full-sequence gradient descent (one pass over the
entire sorted time series per epoch). Suitable for sequences up to
~5000 timesteps. For longer sequences, use WindowedTrainer (below).
"""

import numpy as np
from typing import List, Optional

from model.ssm import SelectiveSSM
from utils.logger import get_logger

log = get_logger("Trainer")


class Trainer:
    """
    Full-sequence trainer for SelectiveSSM.

    Parameters
    ----------
    ssm          : the SelectiveSSM instance to train
    n_iter       : number of training epochs
    patience     : early stopping patience (epochs without improvement)
                   None = no early stopping
    min_delta    : minimum loss improvement to count as "better"
    log_every    : print loss every N epochs
    """

    def __init__(
        self,
        ssm:       SelectiveSSM,
        n_iter:    int = 50,
        patience:  Optional[int] = None,
        min_delta: float = 1e-6,
        log_every: int = 10,
    ):
        self.ssm       = ssm
        self.n_iter    = n_iter
        self.patience  = patience
        self.min_delta = min_delta
        self.log_every = log_every

        # History — inspect after training
        self.loss_history:  List[float] = []
        self.param_norms:   List[dict]  = []

    def fit(self, X: np.ndarray) -> "Trainer":
        """
        Train the SSM on encoded feature matrix X.

        Parameters
        ----------
        X : (T, d_input)  encoded, sorted time series

        Returns
        -------
        self  (for chaining)
        """
        T, d = X.shape
        log.info(
            f"Training: T={T} timesteps | d={d} features | "
            f"epochs={self.n_iter} | patience={self.patience}"
        )

        best_loss  = np.inf
        no_improve = 0
        self.loss_history = []
        self.param_norms  = []

        for epoch in range(1, self.n_iter + 1):
            loss = self.ssm.train_step(X)
            self.loss_history.append(loss)
            self.param_norms.append(self.ssm.param_norms())

            if epoch % self.log_every == 0 or epoch == self.n_iter:
                log.info(f"  epoch {epoch:4d}/{self.n_iter} | loss = {loss:.6f}")

            # ── Early stopping ────────────────────────────────────────────
            if self.patience is not None:
                if loss < best_loss - self.min_delta:
                    best_loss  = loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        log.info(
                            f"Early stopping at epoch {epoch} "
                            f"(no improvement for {self.patience} epochs)"
                        )
                        break

        log.info(
            f"Training complete. "
            f"Initial loss: {self.loss_history[0]:.6f} → "
            f"Final loss: {self.loss_history[-1]:.6f}"
        )
        return self

    def loss_summary(self) -> dict:
        """Return basic statistics about the loss curve."""
        h = np.array(self.loss_history)
        return {
            "initial":     float(h[0]),
            "final":       float(h[-1]),
            "min":         float(h.min()),
            "reduction_%": float((h[0] - h[-1]) / (h[0] + 1e-10) * 100),
            "n_epochs":    len(h),
        }


class WindowedTrainer:
    """
    Windowed trainer — splits long sequences into overlapping windows
    and trains on each window sequentially.

    Use when T > 5000 (pure BPTT over full sequence is slow).

    Parameters
    ----------
    ssm         : SelectiveSSM instance
    window_size : number of timesteps per training window
    stride      : step between window starts (default = window_size // 2)
    n_iter      : epochs over all windows
    log_every   : print loss frequency
    """

    def __init__(
        self,
        ssm:         SelectiveSSM,
        window_size: int = 512,
        stride:      Optional[int] = None,
        n_iter:      int = 10,
        log_every:   int = 5,
    ):
        self.ssm         = ssm
        self.window_size = window_size
        self.stride      = stride or window_size // 2
        self.n_iter      = n_iter
        self.log_every   = log_every
        self.loss_history: List[float] = []

    def fit(self, X: np.ndarray) -> "WindowedTrainer":
        T = len(X)
        starts  = list(range(0, T - self.window_size + 1, self.stride))
        n_wins  = len(starts)
        log.info(
            f"WindowedTrainer: T={T} | window={self.window_size} | "
            f"stride={self.stride} | windows={n_wins} | epochs={self.n_iter}"
        )

        for epoch in range(1, self.n_iter + 1):
            epoch_losses = []
            for start in starts:
                window = X[start: start + self.window_size]
                loss   = self.ssm.train_step(window)
                epoch_losses.append(loss)
            mean_loss = float(np.mean(epoch_losses))
            self.loss_history.append(mean_loss)
            if epoch % self.log_every == 0 or epoch == self.n_iter:
                log.info(f"  epoch {epoch:4d}/{self.n_iter} | mean_loss = {mean_loss:.6f}")

        return self
