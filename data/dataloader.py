"""
data/dataloader.py
==================
GDELTDataLoader — handles reading, sorting, and splitting
raw data files (parquet or CSV) before encoding.

Responsibilities
----------------
  - Load parquet or CSV (auto-detected by extension)
  - Sort by time column
  - Optional group-level filtering (e.g. single country)
  - Train / inference split by date or row index
  - Return raw DataFrames — encoding happens in AutoColumnEncoder

Why separate from encoder?
  Loading and splitting logic changes frequently during debugging.
  Keeping it isolated means you can swap file formats or split strategies
  without touching the encoder or model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from utils.logger import get_logger

log = get_logger("GDELTDataLoader")

# Standard GDELT 2.0 column names (58 cols, tab-separated, no header)
GDELT_COLUMNS = [
    "GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL",
]


class GDELTDataLoader:
    """
    Load and prepare GDELT data for the SSM pipeline.

    Parameters
    ----------
    time_col    : column to sort by (default 'Day')
    group_col   : optional column to filter by (e.g. 'Actor1CountryCode')
    group_val   : if group_col set, keep only rows where col == group_val
    """

    def __init__(
        self,
        time_col:  str = "Day",
        group_col: Optional[str] = None,
        group_val: Optional[str] = None,
    ):
        self.time_col  = time_col
        self.group_col = group_col
        self.group_val = group_val
        self._df: Optional[pd.DataFrame] = None

    # ────────────────────────────────────────────────────────────────────────
    # Loading
    # ────────────────────────────────────────────────────────────────────────

    def load(self, path: str) -> "GDELTDataLoader":
        """
        Load a parquet or CSV file.

        For raw GDELT CSV (tab-separated, no header):
            pass  use_gdelt_header=True  or supply column names yourself.

        Parameters
        ----------
        path : path to .parquet, .csv, or .tsv file
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = p.suffix.lower()
        log.info(f"Loading {suffix} file: {path}")

        if suffix == ".parquet":
            self._df = pd.read_parquet(path)

        elif suffix in (".csv", ".tsv"):
            sep = "\t" if suffix == ".tsv" else ","
            # Try with header first; if it looks like raw GDELT use named cols
            df_try = pd.read_csv(path, sep=sep, nrows=2, header=0)
            if df_try.columns[0].isdigit() or len(df_try.columns) >= 55:
                # Looks like raw GDELT (numeric first col = no header)
                log.info("Detected raw GDELT format — applying standard column names")
                self._df = pd.read_csv(
                    path, sep=sep, header=None,
                    names=GDELT_COLUMNS[:],
                    low_memory=False,
                )
            else:
                self._df = pd.read_csv(path, sep=sep, low_memory=False)

        else:
            raise ValueError(f"Unsupported file type: {suffix}. Use .parquet, .csv, or .tsv")

        log.info(f"Loaded {len(self._df):,} rows × {len(self._df.columns)} cols")
        return self

    def load_dataframe(self, df: pd.DataFrame) -> "GDELTDataLoader":
        """Accept an already-loaded DataFrame directly."""
        self._df = df.copy()
        log.info(f"Received DataFrame: {len(df):,} rows × {len(df.columns)} cols")
        return self

    # ────────────────────────────────────────────────────────────────────────
    # Filtering & sorting
    # ────────────────────────────────────────────────────────────────────────

    def filter_group(self, group_col: str, group_val: str) -> "GDELTDataLoader":
        """Keep only rows where group_col == group_val."""
        self._require_loaded()
        before = len(self._df)
        self._df = self._df[self._df[group_col] == group_val].reset_index(drop=True)
        log.info(f"Filtered {group_col}=={group_val}: {before:,} → {len(self._df):,} rows")
        return self

    def sort_by_time(self) -> "GDELTDataLoader":
        """Sort ascending by time_col."""
        self._require_loaded()
        if self.time_col and self.time_col in self._df.columns:
            self._df = self._df.sort_values(self.time_col).reset_index(drop=True)
            log.info(f"Sorted by '{self.time_col}'")
        else:
            log.warning(f"time_col='{self.time_col}' not found — skipping sort")
        return self

    # ────────────────────────────────────────────────────────────────────────
    # Splitting
    # ────────────────────────────────────────────────────────────────────────

    def split_by_date(
        self, cutoff_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split into train (before cutoff) and inference (from cutoff onwards).

        Parameters
        ----------
        cutoff_date : str in 'YYYYMMDD' format, e.g. '20200101'

        Returns
        -------
        (df_train, df_infer)
        """
        self._require_loaded()
        col = self.time_col
        if col not in self._df.columns:
            raise ValueError(f"time_col '{col}' not in DataFrame")

        cutoff = int(cutoff_date)
        df_train = self._df[self._df[col] <  cutoff].reset_index(drop=True)
        df_infer = self._df[self._df[col] >= cutoff].reset_index(drop=True)
        log.info(
            f"Split at {cutoff}: train={len(df_train):,} rows | "
            f"infer={len(df_infer):,} rows"
        )
        return df_train, df_infer

    def split_by_fraction(
        self, train_frac: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split first train_frac rows → train, remainder → infer.
        Preserves temporal ordering.
        """
        self._require_loaded()
        n     = len(self._df)
        split = int(n * train_frac)
        df_train = self._df.iloc[:split].reset_index(drop=True)
        df_infer = self._df.iloc[split:].reset_index(drop=True)
        log.info(
            f"Split at {train_frac:.0%}: train={len(df_train):,} | "
            f"infer={len(df_infer):,}"
        )
        return df_train, df_infer

    # ────────────────────────────────────────────────────────────────────────
    # Accessors
    # ────────────────────────────────────────────────────────────────────────

    @property
    def df(self) -> pd.DataFrame:
        """Return the current (possibly filtered/sorted) DataFrame."""
        self._require_loaded()
        return self._df

    @property
    def shape(self) -> Tuple[int, int]:
        self._require_loaded()
        return self._df.shape

    def describe(self) -> pd.DataFrame:
        """Quick numeric summary of the loaded data."""
        self._require_loaded()
        return self._df.describe()

    def column_types(self) -> pd.Series:
        """Return dtype of each column."""
        self._require_loaded()
        return self._df.dtypes

    # ────────────────────────────────────────────────────────────────────────
    # Private
    # ────────────────────────────────────────────────────────────────────────

    def _require_loaded(self):
        if self._df is None:
            raise RuntimeError("No data loaded. Call load() or load_dataframe() first.")
