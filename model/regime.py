"""
model/regime.py
===============
RegimeDiscovery — unsupervised k-means clustering on latent states z_t.

Why k-means on z_t (not x_t)?
  x_t is noisy, high-dimensional, and column-heterogeneous.
  z_t is a low-dimensional, temporally coherent summary learned by the SSM.
  Regimes in z-space reflect how the system EVOLVES, not just what it looks
  like at a single moment — which is what matters for geopolitical analysis.

The model is fully unsupervised:
  - You supply K (number of regimes) as a hyperparameter
  - Centroid initialisation is deterministic (seeded)
  - Post-hoc labelling is done by the analyst, not the model
    (e.g. "Regime 2 has high delta and high recon_error → call it Shock")
"""

import numpy as np
from scipy.special import softmax
from typing import Optional

from utils.logger import get_logger

log = get_logger("RegimeDiscovery")


class RegimeDiscovery:
    """
    Unsupervised k-means on latent state trajectories.

    Parameters
    ----------
    n_regimes   : number of clusters K
    max_iter    : k-means iteration limit (default 200)
    temperature : softmax sharpness for predict_proba (higher = sharper)
    seed        : random seed for centroid initialisation
    """

    def __init__(
        self,
        n_regimes:   int   = 3,
        max_iter:    int   = 200,
        temperature: float = 3.0,
        seed:        int   = 42,
    ):
        self.n_regimes   = n_regimes
        self.max_iter    = max_iter
        self.temperature = temperature
        self.seed        = seed
        self.centroids: Optional[np.ndarray] = None   # (K, d_state)
        self.fitted = False

    # ────────────────────────────────────────────────────────────────────────
    # Fit
    # ────────────────────────────────────────────────────────────────────────

    def fit(self, Z: np.ndarray) -> "RegimeDiscovery":
        """
        Run k-means on latent states Z.

        Parameters
        ----------
        Z : (T, d_state)  latent state matrix from SelectiveSSM.predict()
        """
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(Z), self.n_regimes, replace=False)
        self.centroids = Z[idx].copy()

        for iteration in range(self.max_iter):
            # Assignment step
            dists  = np.linalg.norm(
                Z[:, None, :] - self.centroids[None, :, :], axis=2
            )  # (T, K)
            labels = np.argmin(dists, axis=1)

            # Update step
            new_centroids = np.array([
                Z[labels == k].mean(axis=0)
                if (labels == k).any()
                else self.centroids[k]
                for k in range(self.n_regimes)
            ])

            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids

            if shift < 1e-7:
                log.debug(f"K-means converged at iteration {iteration + 1}")
                break

        # Log regime sizes
        sizes = [int((labels == k).sum()) for k in range(self.n_regimes)]
        log.info(f"Regime sizes: { {k: s for k, s in enumerate(sizes)} }")
        self.fitted = True
        return self

    # ────────────────────────────────────────────────────────────────────────
    # Prediction
    # ────────────────────────────────────────────────────────────────────────

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """
        Hard regime assignment.

        Returns
        -------
        labels : (T,)  integer regime index for each timestep
        """
        self._require_fitted()
        dists = np.linalg.norm(
            Z[:, None, :] - self.centroids[None, :, :], axis=2
        )
        return np.argmin(dists, axis=1)

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        """
        Soft regime assignment via distance-based softmax.

        Returns
        -------
        probs : (T, K)  probability of each regime at each timestep
        """
        self._require_fitted()
        dists = np.linalg.norm(
            Z[:, None, :] - self.centroids[None, :, :], axis=2
        )
        return softmax(-dists * self.temperature, axis=1)

    # ────────────────────────────────────────────────────────────────────────
    # Inspection
    # ────────────────────────────────────────────────────────────────────────

    def centroid_distances(self) -> np.ndarray:
        """
        Return K×K matrix of pairwise centroid distances.
        Well-separated regimes → large off-diagonal values.
        """
        self._require_fitted()
        K = self.n_regimes
        D = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                D[i, j] = np.linalg.norm(self.centroids[i] - self.centroids[j])
        return D

    def _require_fitted(self):
        if not self.fitted or self.centroids is None:
            raise RuntimeError("Call fit() before predict().")
