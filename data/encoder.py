"""
data/encoder.py
===============
AutoColumnEncoder — reads any DataFrame, auto-detects column types,
applies the correct encoding, and remembers all decisions so that
inference data receives the EXACT same transformation.

Column type decisions
---------------------
  datetime  → 5 cyclical features: sin/cos day-of-year, sin/cos month, year_norm
  numeric   → StandardScaler (z-score)
  bool      → cast to 0.0 / 1.0
  string    → apply string strategy (see below)

String strategy (when n_rows > min_rows_for_strategy, default 2000)
-------------------------------------------------------------------
  For each string column:
    Compute value frequency distribution.
    If max single-value frequency > ohe_threshold (default 20%):
        → One-Hot Encode
          Keep categories with freq > 1% (cap at 50 categories)
    Else:
        → DROP the column (too sparse, no predictive signal)

  When n_rows <= min_rows_for_strategy:
        → OHE all string columns (not enough data to be selective)

All decisions are stored in self.col_decisions so inference data
can be transformed identically without re-fitting.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, List

from utils.logger import get_logger

log = get_logger("AutoColumnEncoder")


class AutoColumnEncoder:
    """
    Fit-once, transform-many column encoder for arbitrary tabular data.

    Parameters
    ----------
    time_col               : name of the primary time column (forced to datetime)
    ohe_threshold          : max-frequency threshold for OHE vs DROP (default 0.20)
    min_rows_for_strategy  : below this, OHE all strings (default 2000)
    """

    def __init__(
        self,
        time_col: str = None,
        ohe_threshold: float = 0.20,
        min_rows_for_strategy: int = 2000,
    ):
        self.time_col = time_col
        self.ohe_threshold = ohe_threshold
        self.min_rows_for_strategy = min_rows_for_strategy

        # ── Fitted state (serialisable) ──────────────────────────────────────
        self.col_decisions:    Dict[str, str]       = {}   # col → decision tag
        self.scalers:          Dict[str, StandardScaler] = {}
        self.ohe_categories:   Dict[str, List[str]] = {}
        self.output_feature_names: List[str]        = []
        self.fitted: bool = False

    # ────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ────────────────────────────────────────────────────────────────────────

    def _is_datetime(self, series: pd.Series) -> bool:
        """Heuristic: is this column a date/time series?"""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        sample = series.dropna().astype(str).head(30)
        # Try YYYYMMDD integer format (common in GDELT)
        try:
            parsed = pd.to_datetime(sample, format="%Y%m%d", errors="coerce")
            if parsed.notna().mean() > 0.8:
                return True
        except Exception:
            pass
        # Try generic datetime parsing
        try:
            parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
            if parsed.notna().mean() > 0.8:
                return True
        except Exception:
            pass
        return False

    def _parse_dt(self, series: pd.Series) -> pd.Series:
        """Parse a column to datetime, trying YYYYMMDD first."""
        try:
            result = pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")
            if result.notna().mean() > 0.8:
                return result
        except Exception:
            pass
        try:
            return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
        except Exception:
            return pd.Series([pd.NaT] * len(series), index=series.index)

    def _encode_datetime(self, series: pd.Series) -> np.ndarray:
        """
        Convert datetime series to 5 cyclical + normalised features:
            [sin_doy, cos_doy, sin_month, cos_month, year_norm]
        """
        dt = self._parse_dt(series).ffill().fillna(pd.Timestamp("2000-01-01"))
        doy   = dt.dt.dayofyear.fillna(1).values.astype(float)
        month = dt.dt.month.fillna(1).values.astype(float)
        year  = dt.dt.year.fillna(2000).values.astype(float)
        yr_norm = (year - year.min()) / (year.max() - year.min() + 1e-6)
        return np.column_stack([
            np.sin(2 * np.pi * doy   / 365),
            np.cos(2 * np.pi * doy   / 365),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            yr_norm,
        ])

    # ────────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "AutoColumnEncoder":
        """
        Inspect df and record transformation decisions for every column.
        Does NOT transform — call transform() separately.

        Parameters
        ----------
        df : training DataFrame (any number of columns)
        """
        n = len(df)
        use_string_strategy = n > self.min_rows_for_strategy

        self.col_decisions = {}
        self.scalers = {}
        self.ohe_categories = {}
        self.output_feature_names = []

        log.info(f"{n} rows | {len(df.columns)} columns")
        log.info(
            f"String strategy: {'active' if use_string_strategy else 'inactive'} "
            f"(OHE threshold = {self.ohe_threshold * 100:.0f}%)"
        )

        for col in df.columns:
            series = df[col]

            # ── Datetime ────────────────────────────────────────────────────
            if col == self.time_col or self._is_datetime(series):
                self.col_decisions[col] = "datetime"
                self.output_feature_names += [
                    f"{col}_sin_doy",
                    f"{col}_cos_doy",
                    f"{col}_sin_month",
                    f"{col}_cos_month",
                    f"{col}_year_norm",
                ]
                log.debug(f"  {col:45s} → datetime  (5 features)")
                continue

            # ── Bool ────────────────────────────────────────────────────────
            if pd.api.types.is_bool_dtype(series):
                self.col_decisions[col] = "bool"
                self.output_feature_names.append(col)
                log.debug(f"  {col:45s} → bool")
                continue

            # ── Numeric ─────────────────────────────────────────────────────
            if pd.api.types.is_numeric_dtype(series):
                self.col_decisions[col] = "numeric"
                sc = StandardScaler()
                fill_val = series.median() if series.notna().any() else 0.0
                sc.fit(series.fillna(fill_val).values.reshape(-1, 1))
                self.scalers[col] = sc
                self.output_feature_names.append(col)
                log.debug(f"  {col:45s} → numeric   (z-score)")
                continue

            # ── String / categorical ─────────────────────────────────────────
            str_series = series.fillna("MISSING").astype(str)
            freq       = str_series.value_counts(normalize=True)
            max_freq   = float(freq.iloc[0]) if len(freq) else 0.0
            top_val    = str(freq.index[0])  if len(freq) else ""

            do_ohe = (not use_string_strategy) or (max_freq > self.ohe_threshold)

            if do_ohe:
                # Keep categories with freq > 1%, hard-cap at 50
                cats = freq[freq > 0.01].index.tolist()[:50]
                self.col_decisions[col]    = "ohe"
                self.ohe_categories[col]   = cats
                for cat in cats:
                    self.output_feature_names.append(f"{col}__{cat}")
                log.info(
                    f"  {col:45s} → OHE  "
                    f"({len(cats)} cats | top='{top_val}' {max_freq*100:.1f}%)"
                )
            else:
                self.col_decisions[col] = "drop"
                log.warning(
                    f"  {col:45s} → DROP "
                    f"(max_freq={max_freq*100:.1f}% < {self.ohe_threshold*100:.0f}%)"
                )

        self.fitted = True
        log.info(f"→ {len(self.output_feature_names)} total encoded features\n")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply fitted transformations to df.
        Columns not seen during fit are silently skipped.
        Missing categories in OHE columns are encoded as all-zeros.

        Returns
        -------
        X : np.ndarray of shape (n_rows, n_features), dtype float32
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before transform().")

        parts = []

        for col in df.columns:
            if col not in self.col_decisions:
                log.debug(f"Skipping unseen column: {col}")
                continue

            decision = self.col_decisions[col]
            series   = df[col]

            if decision == "drop":
                continue

            elif decision == "numeric":
                fill_val = series.median() if series.notna().any() else 0.0
                vals = series.fillna(fill_val).values.reshape(-1, 1)
                parts.append(self.scalers[col].transform(vals))

            elif decision == "datetime":
                parts.append(self._encode_datetime(series))

            elif decision == "bool":
                parts.append(series.fillna(False).astype(float).values.reshape(-1, 1))

            elif decision == "ohe":
                str_s = series.fillna("MISSING").astype(str)
                cats  = self.ohe_categories[col]
                mat   = np.zeros((len(df), len(cats)), dtype=np.float32)
                for i, cat in enumerate(cats):
                    mat[:, i] = (str_s == cat).astype(float).values
                parts.append(mat)

        if not parts:
            raise ValueError("No encodable columns after transformation.")

        X = np.hstack(parts).astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # ────────────────────────────────────────────────────────────────────────
    # Inspection helpers
    # ────────────────────────────────────────────────────────────────────────

    @property
    def n_features(self) -> int:
        return len(self.output_feature_names)

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame describing each column's decision."""
        rows = []
        for col, dec in self.col_decisions.items():
            if dec == "ohe":
                n_out = len(self.ohe_categories.get(col, []))
            elif dec == "datetime":
                n_out = 5
            elif dec == "drop":
                n_out = 0
            else:
                n_out = 1
            rows.append({"column": col, "decision": dec, "output_features": n_out})
        return pd.DataFrame(rows)
