import pandas as pd
# import warnings
# warnings.filterwarnings("ignore")

from data.dataloader import GDELTDataLoader, GDELT_COLUMNS
from training.pipeline import ShockDetectionPipeline

# ── 1. Load ───────────────────────────────────────────────────────────────────
PARQUET_PATH = "/mnt/data/TEST_CSV/2000.parquet"

loader = GDELTDataLoader(time_col="Day")
loader.load(PARQUET_PATH)
df = loader.df

# ── 2. Assign column names if integer-indexed ─────────────────────────────────
if isinstance(df.columns[0], int) or str(df.columns[0]).isdigit():
    print(f"[run.py] Integer column names detected — assigning GDELT column names")
    df.columns = GDELT_COLUMNS[: len(df.columns)]

# ── 3. Sort by time after column names are assigned ───────────────────────────
df = df.sort_values("Day").reset_index(drop=True)

print(f"[run.py] Columns: {df.columns.tolist()[:8]} ...")
print(f"[run.py] Full shape: {df.shape}")

# ── 4. Build pipeline ─────────────────────────────────────────────────────────
pipe = ShockDetectionPipeline(
    latent_dim    = 16,
    n_regimes     = 3,
    n_iter        = 1,       # ← increase for better results (50 or 100)
    lr            = 1e-3,
    time_col      = "Day",
    group_col     = "Actor1CountryCode",
    ohe_threshold = 0.20,
    windowed      = True,
    window_size   = 512,
)

# ── 5. Fit ────────────────────────────────────────────────────────────────────
pipe.fit(df)

# ── 6. Save model ─────────────────────────────────────────────────────────────
pipe.save("model.pkl")
print("[run.py] Model saved to model.pkl")

# ── 7. Inference ──────────────────────────────────────────────────────────────
results = pipe.transform(df)

# ── 8. Save results to CSV ────────────────────────────────────────────────────
results.to_csv("results.csv", index=False)
print(f"[run.py] Results saved → results.csv  ({len(results):,} rows)")

# ── 9. Print summary ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ENCODER SUMMARY (what happened to each column):")
print("="*60)
print(pipe.encoder_summary().to_string(index=False))

print("\n" + "="*60)
print("TOP 10 DETECTED SHOCKS:")
print("="*60)
print(pipe.top_shocks(results, top_n=10).to_string(index=False))

print("\n" + "="*60)
print("REGIME SUMMARY:")
print("="*60)
print(pipe.regime_summary(results).to_string())

print("\n" + "="*60)
print(f"OUTPUT SHAPE : {results.shape}")
print(f"OUTPUT COLS  : {list(results.columns)}")
print("="*60)

# ── 10. Load and verify ───────────────────────────────────────────────────────
print("\n[run.py] Verifying save/load round-trip ...")
pipe2    = ShockDetectionPipeline.load("model.pkl")
results2 = pipe2.transform(df)
print(f"[run.py] Loaded model output shape: {results2.shape}  ✓")