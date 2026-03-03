"""
model/scorer.py
===============
ChangePointScorer — combines three signals into a single shock_score ∈ [0, 1].

Three signals
-------------
1. Delta spike (weight 0.4)
   Δ_t is the SSM selective gate.
   Large Δ means the model is resetting its state because the current input
   is very different from what it was tracking.
   → Direct Mamba signal for regime change.

2. State velocity (weight 0.4)
   ||z_t - z_{t-1}||₂  — how far the latent state jumped in one step.
   Large jump = the hidden representation of the world changed suddenly.
   → Catches structural breaks that cause rapid z-space movement.

3. Reconstruction error (weight 0.2)
   ||x_t - x̂_t||²  — how surprised the model was by the actual input.
   High error = model did not expect this input given past context.
   → Catches unusual events even when z moves smoothly.

All three signals are z-score normalised before combining, then
passed through sigmoid to produce a final score in [0, 1].
"""

import numpy as np
from utils.activations import sigmoid
from utils.logger import get_logger

log = get_logger("ChangePointScorer")


class ChangePointScorer:
    """
    Combine SSM outputs into a per-timestep shock score.

    Parameters
    ----------
    w_delta    : weight for delta spike signal    (default 0.4)
    w_velocity : weight for state velocity signal (default 0.4)
    w_recon    : weight for reconstruction error  (default 0.2)
    """

    def __init__(
        self,
        w_delta:    float = 0.4,
        w_velocity: float = 0.4,
        w_recon:    float = 0.2,
    ):
        assert abs(w_delta + w_velocity + w_recon - 1.0) < 1e-6, \
            "Weights must sum to 1.0"
        self.w_delta    = w_delta
        self.w_velocity = w_velocity
        self.w_recon    = w_recon

    def score(
        self,
        Z:            np.ndarray,
        deltas:       np.ndarray,
        recon_errors: np.ndarray,
    ) -> np.ndarray:
        """
        Compute shock_score for each timestep.

        Parameters
        ----------
        Z            : (T, d_state)  latent state trajectory
        deltas       : (T,)          SSM gate values
        recon_errors : (T,)          per-timestep reconstruction MSE

        Returns
        -------
        shock_scores : (T,)  values in [0, 1]
        """
        T = len(Z)

        # ── Signal 1: Delta spike ─────────────────────────────────────────
        d_norm = self._normalise(deltas)

        # ── Signal 2: State velocity ──────────────────────────────────────
        velocity       = np.zeros(T)
        velocity[1:]   = np.linalg.norm(np.diff(Z, axis=0), axis=1)
        v_norm         = self._normalise(velocity)

        # ── Signal 3: Reconstruction error ───────────────────────────────
        e_norm = self._normalise(recon_errors)

        # ── Weighted combination → sigmoid ────────────────────────────────
        raw    = self.w_delta * d_norm + self.w_velocity * v_norm + self.w_recon * e_norm
        scores = sigmoid(raw)

        log.debug(
            f"shock_score stats: "
            f"mean={scores.mean():.3f} | max={scores.max():.3f} | "
            f"n_high={int((scores > 0.8).sum())} (>0.8)"
        )
        return scores

    def signal_breakdown(
        self,
        Z:            np.ndarray,
        deltas:       np.ndarray,
        recon_errors: np.ndarray,
    ) -> dict:
        """
        Return all three individual signals (normalised) alongside the
        combined score. Useful for debugging which signal is driving detections.

        Returns
        -------
        dict with keys: delta_norm, velocity_norm, recon_norm, shock_score
        Each value is a (T,) array.
        """
        T = len(Z)
        velocity     = np.zeros(T)
        velocity[1:] = np.linalg.norm(np.diff(Z, axis=0), axis=1)

        d_norm = self._normalise(deltas)
        v_norm = self._normalise(velocity)
        e_norm = self._normalise(recon_errors)
        raw    = self.w_delta * d_norm + self.w_velocity * v_norm + self.w_recon * e_norm

        return {
            "delta_norm":    d_norm,
            "velocity_norm": v_norm,
            "recon_norm":    e_norm,
            "shock_score":   sigmoid(raw),
        }

    @staticmethod
    def _normalise(x: np.ndarray) -> np.ndarray:
        """Z-score normalisation with safe denominator."""
        return (x - x.mean()) / (x.std() + 1e-8)
