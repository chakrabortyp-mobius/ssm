"""
training/pipeline.py
====================
ShockDetectionPipeline — top-level orchestrator.

K is now auto-selected via silhouette score (K=2..max_k).
n_regimes parameter removed — replaced by max_k (default 10).
silhouette_results and best_k stored after fit() for reporting.
"""

import numpy as np
import pandas as pd
from typing import Optional

from data.encoder import AutoColumnEncoder
from model.ssm import SelectiveSSM
from model.regime import RegimeDiscovery, best_k_by_silhouette
from model.scorer import ChangePointScorer
from training.trainer import Trainer, WindowedTrainer
from utils.logger import get_logger

log = get_logger("ShockDetectionPipeline")


class ShockDetectionPipeline:
    """
    Parameters
    ----------
    latent_dim        : shock embedding size (z_t dimensions)
    max_k             : max K to test for silhouette (tests K=2..max_k)
    n_iter            : training epochs
    lr                : Adam learning rate
    time_col          : column name for temporal ordering
    group_col         : extra column re-attached to output (e.g. country)
    ohe_threshold     : string OHE threshold (default 0.20)
    patience          : early stopping patience (None = disabled)
    windowed          : use WindowedTrainer for large data
    window_size       : window size if windowed=True
    silhouette_sample : rows to subsample for silhouette (keep ≤10k)
    """

    def __init__(
        self,
        latent_dim:        int   = 16,
        max_k:             int   = 10,
        n_iter:            int   = 50,
        lr:                float = 1e-3,
        time_col:          Optional[str] = None,
        group_col:         Optional[str] = None,
        ohe_threshold:     float = 0.20,
        patience:          Optional[int] = None,
        windowed:          bool  = False,
        window_size:       int   = 512,
        silhouette_sample: int   = 10_000,
    ):
        self.latent_dim        = latent_dim
        self.max_k             = max_k
        self.n_iter            = n_iter
        self.lr                = lr
        self.time_col          = time_col
        self.group_col         = group_col
        self.ohe_threshold     = ohe_threshold
        self.patience          = patience
        self.windowed          = windowed
        self.window_size       = window_size
        self.silhouette_sample = silhouette_sample

        self.encoder:            Optional[AutoColumnEncoder] = None
        self.ssm:                Optional[SelectiveSSM]      = None
        self.trainer:            Optional[Trainer]           = None
        self.regime:             Optional[RegimeDiscovery]   = None
        self.scorer              = ChangePointScorer()
        self.fitted              = False
        self.best_k:             int  = 3
        self.silhouette_results: dict = {}

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "ShockDetectionPipeline":
        df = self._sort(df)
        log.info(f"Fitting pipeline on {len(df):,} rows")

        # Step 1: Encode
        self.encoder = AutoColumnEncoder(
            time_col=self.time_col,
            ohe_threshold=self.ohe_threshold,
        )
        X = self.encoder.fit_transform(df)
        log.info(f"Encoded shape: {X.shape}")

        # Step 2: Build + train SSM
        self.ssm = SelectiveSSM(d_input=X.shape[1], d_state=self.latent_dim, lr=self.lr)
        if self.windowed and len(X) > self.window_size:
            self.trainer = WindowedTrainer(self.ssm, window_size=self.window_size, n_iter=self.n_iter)
        else:
            self.trainer = Trainer(self.ssm, self.n_iter, self.patience)
        self.trainer.fit(X)

        # Step 3: Latent states
        Z, _, _ = self.ssm.predict(X)

        # Step 4: Silhouette K selection
        log.info(f"Selecting best K via silhouette (K=2..{self.max_k}) ...")
        sil                  = best_k_by_silhouette(Z, k_max=self.max_k,
                                                    sample_size=self.silhouette_sample)
        self.silhouette_results = sil
        self.best_k             = sil["best_k"]
        log.info(f"Auto-selected K = {self.best_k}")

        # Step 5: Fit regimes with best K
        self.regime = RegimeDiscovery(n_regimes=self.best_k)
        self.regime.fit(Z)

        self.fitted = True
        log.info("Pipeline fit complete.")
        return self

    # ── Transform ────────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check()
        df = self._sort(df)
        X  = self.encoder.transform(df)

        Z, deltas, recon = self.ssm.predict(X)
        recon_errors  = np.mean((X - recon) ** 2, axis=1)
        shock_scores  = self.scorer.score(Z, deltas, recon_errors)
        regime_probs  = self.regime.predict_proba(Z)
        regime_labels = np.argmax(regime_probs, axis=1)

        out = pd.DataFrame(Z, columns=[f"z_{i}" for i in range(self.latent_dim)])
        out["shock_score"] = shock_scores
        out["regime"]      = regime_labels
        out["delta"]       = deltas
        out["recon_error"] = recon_errors
        for k in range(self.best_k):
            out[f"regime_{k}_prob"] = regime_probs[:, k]
        for col in [self.time_col, self.group_col]:
            if col and col in df.columns:
                out[col] = df[col].values
        return out

    # ── Helpers ───────────────────────────────────────────────────────────────

    def top_shocks(self, results, top_n=10):
        cols = ["shock_score","regime","delta","recon_error"]
        for c in [self.time_col, self.group_col]:
            if c and c in results.columns: cols = [c] + cols
        return results.nlargest(top_n, "shock_score")[cols]

    def regime_summary(self, results):
        return results.groupby("regime").agg(
            count         =("shock_score","count"),
            avg_shock     =("shock_score","mean"),
            avg_delta     =("delta","mean"),
            avg_recon_err =("recon_error","mean"),
        ).round(4)

    def training_summary(self):
        self._check(); return self.trainer.loss_summary()

    def encoder_summary(self):
        self._check(); return self.encoder.summary()

    def save(self, path="model.pkl"):
        import pickle, os
        self._check()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path,"wb") as f: pickle.dump(self,f)
        mb = os.path.getsize(path)/1024/1024
        log.info(f"Model saved → {path}  ({mb:.2f} MB)")

    @classmethod
    def load(cls, path):
        import pickle
        with open(path,"rb") as f: obj = pickle.load(f)
        log.info(f"Model loaded ← {path}"); return obj

    def _sort(self, df):
        if self.time_col and self.time_col in df.columns:
            return df.sort_values(self.time_col).reset_index(drop=True)
        return df.reset_index(drop=True)

    def _check(self):
        if not self.fitted: raise RuntimeError("Call fit() before transform().")

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def signal_breakdown(self, df_original: pd.DataFrame) -> pd.DataFrame:
        self._check()
        df_sorted = self._sort(df_original)
        X = self.encoder.transform(df_sorted)
        Z, deltas, recon = self.ssm.predict(X)
        recon_errors = np.mean((X - recon) ** 2, axis=1)
        breakdown = self.scorer.signal_breakdown(Z, deltas, recon_errors)
        out = pd.DataFrame(breakdown)
        for col in [self.time_col, self.group_col]:
            if col and col in df_sorted.columns:
                out[col] = df_sorted[col].values
        return out