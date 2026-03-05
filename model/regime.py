"""
model/regime.py
===============
RegimeDiscovery — unsupervised k-means on latent states z_t.

K is chosen automatically via silhouette score (K=2..10).
Regime labels (0,1,2...) are raw integers. The analyst interprets
them by inspecting avg_shock and avg_delta per regime.
"""

import numpy as np
from scipy.special import softmax
from typing import Optional, Dict
from utils.logger import get_logger

log = get_logger("RegimeDiscovery")


def best_k_by_silhouette(
    Z:           np.ndarray,
    k_max:       int = 10,
    sample_size: int = 10_000,
    seed:        int = 42,
) -> Dict:
    """
    Test K = 2..k_max and return the K with the highest silhouette score.

    Silhouette score: how well each point fits its own cluster vs
    the nearest other cluster. Range [-1, +1]. Higher = better.

    Parameters
    ----------
    Z           : (T, d_state) latent state matrix from SSM
    k_max       : max K to test (tests 2, 3, ..., k_max)
    sample_size : rows to subsample (silhouette is O(n^2) — keep ≤10k)
    seed        : random seed

    Returns
    -------
    dict:
        best_k      : int   — K with highest silhouette score
        scores      : dict  — {k: score} for all tested K
        best_score  : float — silhouette score at best_k
    """
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    if len(Z) > sample_size:
        idx      = rng.choice(len(Z), sample_size, replace=False)
        Z_sample = Z[idx]
    else:
        Z_sample = Z

    scores = {}
    log.info(f"Silhouette K selection: K=2..{k_max} on {len(Z_sample):,} samples")

    for k in range(2, k_max + 1):
        rd     = RegimeDiscovery(n_regimes=k, seed=seed)
        rd.fit(Z_sample)
        labels = rd.predict(Z_sample)

        if len(np.unique(labels)) < 2:
            scores[k] = -1.0
            log.warning(f"  K={k} produced only 1 cluster — score=-1")
            continue

        score     = float(silhouette_score(Z_sample, labels, random_state=seed))
        scores[k] = round(score, 6)
        log.info(f"  K={k:2d} | silhouette = {score:.6f}")

    best_k     = max(scores, key=scores.get)
    best_score = scores[best_k]
    log.info(f"Best K = {best_k}  (silhouette = {best_score:.6f})")

    return {"best_k": best_k, "scores": scores, "best_score": best_score}


class RegimeDiscovery:
    def __init__(self, n_regimes=3, max_iter=200, temperature=3.0, seed=42):
        self.n_regimes   = n_regimes
        self.max_iter    = max_iter
        self.temperature = temperature
        self.seed        = seed
        self.centroids: Optional[np.ndarray] = None
        self.fitted = False

    def fit(self, Z: np.ndarray) -> "RegimeDiscovery":
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(Z), self.n_regimes, replace=False)
        self.centroids = Z[idx].copy()

        for it in range(self.max_iter):
            dists  = np.linalg.norm(Z[:,None,:] - self.centroids[None,:,:], axis=2)
            labels = np.argmin(dists, axis=1)
            new_c  = np.array([
                Z[labels==k].mean(axis=0) if (labels==k).any() else self.centroids[k]
                for k in range(self.n_regimes)
            ])
            if np.linalg.norm(new_c - self.centroids) < 1e-7:
                break
            self.centroids = new_c

        sizes = {k: int((labels==k).sum()) for k in range(self.n_regimes)}
        log.info(f"Regime sizes: {sizes}")
        self.fitted = True
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        self._check(); dists = np.linalg.norm(Z[:,None,:] - self.centroids[None,:,:], axis=2)
        return np.argmin(dists, axis=1)

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        self._check(); dists = np.linalg.norm(Z[:,None,:] - self.centroids[None,:,:], axis=2)
        return softmax(-dists * self.temperature, axis=1)

    def centroid_distances(self) -> np.ndarray:
        self._check()
        K = self.n_regimes; D = np.zeros((K,K))
        for i in range(K):
            for j in range(K):
                D[i,j] = np.linalg.norm(self.centroids[i] - self.centroids[j])
        return D

    def _check(self):
        if not self.fitted: raise RuntimeError("Call fit() first.")