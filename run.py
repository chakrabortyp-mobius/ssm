"""
run.py
======
Entry point for GDELT Shock SSM.
Run from inside the gdelt_ssm/ directory:

    python run.py

What this script does:
    1.  Load parquet
    2.  Assign GDELT column names if integer-indexed
    3.  Sort by Day (time)
    4.  Split: 95% train / 5% validation (time-ordered, no shuffle)
    5.  Fit pipeline on train data
        - AutoColumnEncoder (OHE/numeric/datetime)
        - SelectiveSSM training (windowed BPTT)
        - Silhouette K selection (K=2..10) → auto pick best K
        - RegimeDiscovery with best K
    6.  Save model → model.pkl
    7.  Transform train split  → reports/train/
    8.  Transform val split    → reports/val/
    9.  Each report folder contains:
            report.csv, top_shocks.csv, regime_summary.csv,
            cluster_plot.png, shock_score_timeline.png
        Train folder also contains:
            silhouette_scores.csv, silhouette_plot.png
"""

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from data.dataloader import GDELTDataLoader, GDELT_COLUMNS
from training.pipeline import ShockDetectionPipeline
from reports.generate_report import generate_report

# ── 1. Load ───────────────────────────────────────────────────────────────────
PARQUET_PATH = "/mnt/data/TEST_CSV/2000.parquet"

loader = GDELTDataLoader(time_col="Day")
loader.load(PARQUET_PATH)
df = loader.df

# ── 2. Assign column names if integer-indexed ─────────────────────────────────
if isinstance(df.columns[0], int) or str(df.columns[0]).isdigit():
    print("[run.py] Integer columns detected — assigning GDELT column names")
    df.columns = GDELT_COLUMNS[: len(df.columns)]

# ── 3. Sort by time ───────────────────────────────────────────────────────────
df = df.sort_values("Day").reset_index(drop=True)
print(f"[run.py] Loaded & sorted: {df.shape}")

# ── 4. 95 / 5 split (time-ordered, no shuffle) ───────────────────────────────
split_idx  = int(len(df) * 0.95)
df_train   = df.iloc[:split_idx].reset_index(drop=True)
df_val     = df.iloc[split_idx:].reset_index(drop=True)
print(f"[run.py] Train: {len(df_train):,} rows  |  Val: {len(df_val):,} rows")

# ── 5. Build pipeline ─────────────────────────────────────────────────────────
pipe = ShockDetectionPipeline(
    latent_dim        = 16,
    max_k             = 10,      # silhouette tests K=2..10, picks best
    n_iter            = 1,       # ← increase to 50-100 for production
    lr                = 1e-3,
    time_col          = "Day",
    group_col         = "Actor1CountryCode",
    ohe_threshold     = 0.20,
    windowed          = True,
    window_size       = 512,
    silhouette_sample = 10_000,
)

# ── 6. Fit on train ───────────────────────────────────────────────────────────
pipe.fit(df_train)
print(f"[run.py] Best K selected = {pipe.best_k}")

# ── 7. Save model ─────────────────────────────────────────────────────────────
pipe.save("model.pkl")

# ── 8. Transform both splits ──────────────────────────────────────────────────
print("[run.py] Running inference on train split ...")
results_train = pipe.transform(df_train)

print("[run.py] Running inference on val split ...")
results_val   = pipe.transform(df_val)

# ── 9. Generate reports ───────────────────────────────────────────────────────
generate_report(
    results        = results_train,
    split_name     = "train",
    pipe           = pipe,
    out_root       = "reports",
    include_silhouette = True,    # silhouette CSV + plot only in train
)

generate_report(
    results        = results_val,
    split_name     = "val",
    pipe           = pipe,
    out_root       = "reports",
    include_silhouette = False,
)

# ── 10. Print quick summary ───────────────────────────────────────────────────
print("\n" + "="*60)
print(f"BEST K (silhouette) : {pipe.best_k}")
print("\nSILHOUETTE SCORES:")
for k, v in sorted(pipe.silhouette_results['scores'].items()):
    marker = " ← best" if k == pipe.best_k else ""
    print(f"  K={k:2d}  score={v:.6f}{marker}")

print("\nREGIME SUMMARY (train):")
print(pipe.regime_summary(results_train).to_string())

print("\nTOP 10 SHOCKS (train):")
print(pipe.top_shocks(results_train, top_n=10).to_string(index=False))

print("\nREPORTS SAVED:")
print("  reports/train/  — report.csv, top_shocks.csv, regime_summary.csv,")
print("                    silhouette_scores.csv, silhouette_plot.png,")
print("                    cluster_plot.png, shock_score_timeline.png")
print("  reports/val/    — report.csv, top_shocks.csv, regime_summary.csv,")
print("                    cluster_plot.png, shock_score_timeline.png")
print("="*60)