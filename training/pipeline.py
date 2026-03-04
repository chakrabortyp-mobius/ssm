"""
training/pipeline.py
====================
ShockDetectionPipeline — top-level orchestrator.

Wires together:
    GDELTDataLoader   → load / filter / sort raw data
    AutoColumnEncoder → encode to numeric matrix
    Trainer           → train SelectiveSSM
    RegimeDiscovery   → unsupervised regime clustering
    ChangePointScorer → produce shock_score per timestep

This is the only file you need to import in your notebooks or scripts.
All other modules can be imported individually for debugging.

Typical usage
-------------
    from training.pipeline import ShockDetectionPipeline

    pipe = ShockDetectionPipeline(
        latent_dim = 16,
        n_regimes  = 3,
        n_iter     = 100,
        time_col   = 'Day',
    )
    pipe.fit(df_train)
    results = pipe.transform(df_new)
"""

import numpy as np
import pandas as pd
from typing import Optional

from data.encoder import AutoColumnEncoder
from model.ssm import SelectiveSSM
from model.regime import RegimeDiscovery
from model.scorer import ChangePointScorer
from training.trainer import Trainer, WindowedTrainer
from utils.logger import get_logger

log = get_logger("ShockDetectionPipeline")


class ShockDetectionPipeline:
    """
    End-to-end shock detection pipeline.

    Parameters
    ----------
    latent_dim      : dimension of shock embedding z_t
    n_regimes       : number of unsupervised regimes
    n_iter          : training epochs
    lr              : Adam learning rate
    time_col        : column name for temporal ordering
    group_col       : informational column re-attached to output (e.g. country)
    ohe_threshold   : string OHE threshold (default 0.20)
    patience        : early stopping patience (None = disabled)
    windowed        : use WindowedTrainer for long sequences (T > 5000)
    window_size     : window size if windowed=True
    """

    def __init__(
        self,
        latent_dim:    int   = 16,
        n_regimes:     int   = 3,
        n_iter:        int   = 50,
        lr:            float = 1e-3,
        time_col:      Optional[str] = None,
        group_col:     Optional[str] = None,
        ohe_threshold: float = 0.20,
        patience:      Optional[int] = None,
        windowed:      bool  = False,
        window_size:   int   = 512,
    ):
        self.latent_dim    = latent_dim
        self.n_regimes     = n_regimes
        self.n_iter        = n_iter
        self.lr            = lr
        self.time_col      = time_col
        self.group_col     = group_col
        self.ohe_threshold = ohe_threshold
        self.patience      = patience
        self.windowed      = windowed
        self.window_size   = window_size

        # Components — initialised in fit()
        self.encoder:  Optional[AutoColumnEncoder] = None
        self.ssm:      Optional[SelectiveSSM]      = None
        self.trainer:  Optional[Trainer]           = None
        self.regime:   Optional[RegimeDiscovery]   = None
        self.scorer    = ChangePointScorer()
        self.fitted    = False

    # ────────────────────────────────────────────────────────────────────────
    # Fit
    # ────────────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "ShockDetectionPipeline":
        """
        Fit the full pipeline on training data.

        Steps:
            1. Sort by time_col
            2. Fit AutoColumnEncoder + encode to matrix X
            3. Build SelectiveSSM with correct d_input
            4. Train via Trainer (or WindowedTrainer)
            5. Run forward pass → get z_t for all rows
            6. Fit RegimeDiscovery on z_t
        """
        df = self._sort(df)
        log.info(f"Fitting pipeline on {len(df):,} rows")

        # Step 1: Encode
        self.encoder = AutoColumnEncoder(
            time_col=self.time_col,
            ohe_threshold=self.ohe_threshold,
        )
        X = self.encoder.fit_transform(df)
        d_input = X.shape[1]
        log.info(f"Encoded shape: {X.shape}")

        # Step 2: Build SSM
        self.ssm = SelectiveSSM(
            d_input=d_input,
            d_state=self.latent_dim,
            lr=self.lr,
        )

        # Step 3: Train
        if self.windowed and len(X) > self.window_size:
            self.trainer = WindowedTrainer(
                ssm=self.ssm,
                window_size=self.window_size,
                n_iter=self.n_iter,
            )
        else:
            self.trainer = Trainer(
                ssm=self.ssm,
                n_iter=self.n_iter,
                patience=self.patience,
            )
        self.trainer.fit(X)

        # Step 4: Get latent states → fit regimes
        Z, _, _ = self.ssm.predict(X)
        self.regime = RegimeDiscovery(n_regimes=self.n_regimes)
        self.regime.fit(Z)

        self.fitted = True
        log.info("Pipeline fit complete.")
        return self

    # ────────────────────────────────────────────────────────────────────────
    # Transform (inference)
    # ────────────────────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on new data.

        The SAME column encoding decisions from fit() are applied.
        New/unseen columns are silently ignored.

        Returns
        -------
        DataFrame with one row per input row:
            z_0 ... z_{d-1}   shock embedding dimensions
            shock_score       [0,1] shock probability
            regime            discrete regime label
            regime_k_prob     soft regime probabilities
            delta             SSM gate (large = shock)
            recon_error       reconstruction MSE per step
            [time_col]        re-attached if present
            [group_col]       re-attached if present
        """
        self._require_fitted()
        df = self._sort(df)

        X = self.encoder.transform(df)
        Z, deltas, recon = self.ssm.predict(X)
        recon_errors  = np.mean((X - recon) ** 2, axis=1)
        shock_scores  = self.scorer.score(Z, deltas, recon_errors)
        regime_probs  = self.regime.predict_proba(Z)
        regime_labels = np.argmax(regime_probs, axis=1)

        # Assemble output DataFrame
        out = pd.DataFrame(
            Z, columns=[f"z_{i}" for i in range(self.latent_dim)]
        )
        out["shock_score"] = shock_scores
        out["regime"]      = regime_labels
        out["delta"]       = deltas
        out["recon_error"] = recon_errors
        for k in range(self.n_regimes):
            out[f"regime_{k}_prob"] = regime_probs[:, k]

        # Re-attach identifier columns
        for col in [self.time_col, self.group_col]:
            if col and col in df.columns:
                out[col] = df[col].values

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    # ────────────────────────────────────────────────────────────────────────
    # Analysis helpers
    # ────────────────────────────────────────────────────────────────────────

    def top_shocks(self, results: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Return top-N timesteps by shock_score."""
        cols = ["shock_score", "regime", "delta", "recon_error"]
        for c in [self.time_col, self.group_col]:
            if c and c in results.columns:
                cols = [c] + cols
        return results.nlargest(top_n, "shock_score")[cols]

    def regime_summary(self, results: pd.DataFrame) -> pd.DataFrame:
        """Aggregate statistics per discovered regime."""
        return results.groupby("regime").agg(
            count         = ("shock_score", "count"),
            avg_shock     = ("shock_score", "mean"),
            avg_delta     = ("delta",       "mean"),
            avg_recon_err = ("recon_error", "mean"),
        ).round(4)

    def training_summary(self) -> dict:
        """Return loss curve summary from trainer."""
        self._require_fitted()
        return self.trainer.loss_summary()

    def encoder_summary(self) -> pd.DataFrame:
        """Show what was done to each column."""
        self._require_fitted()
        return self.encoder.summary()

    def regime_distances(self) -> np.ndarray:
        """Pairwise centroid distances — are regimes well separated?"""
        self._require_fitted()
        return self.regime.centroid_distances()

    def signal_breakdown(self, results_df: pd.DataFrame, df_original: pd.DataFrame) -> pd.DataFrame:
        """
        Return the three raw signals that compose shock_score.
        Useful for debugging which signal is driving a detection.
        """
        self._require_fitted()
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

    # ────────────────────────────────────────────────────────────────────────
    # Private
    # ────────────────────────────────────────────────────────────────────────

    def _sort(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.time_col and self.time_col in df.columns:
            return df.sort_values(self.time_col).reset_index(drop=True)
        return df.reset_index(drop=True)

    def _require_fitted(self):
        if not self.fitted:
            raise RuntimeError("Call fit() before transform() or analysis methods.")

    # ────────────────────────────────────────────────────────────────────────
    # Save / Load
    # ────────────────────────────────────────────────────────────────────────

    def save(self, path: str = "model.pkl") -> None:
        """
        Save the fitted pipeline to disk using pickle.

        Saves everything needed for inference:
            encoder decisions, SSM weights, regime centroids, scorer config.

        Parameters
        ----------
        path : file path, e.g. "model.pkl" or "/mnt/data/ssm/model.pkl"
        """
        import pickle, os
        self._require_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        size_mb = os.path.getsize(path) / 1024 / 1024
        log.info(f"Model saved → {path}  ({size_mb:.2f} MB)")

    @classmethod
    def load(cls, path: str) -> "ShockDetectionPipeline":
        """
        Load a previously saved pipeline from disk.

        Parameters
        ----------
        path : path to the .pkl file written by save()

        Returns
        -------
        Fitted ShockDetectionPipeline ready for .transform()
        """
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        log.info(f"Model loaded ← {path}")
        return obj