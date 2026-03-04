import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from data.dataloader import GDELTDataLoader, GDELT_COLUMNS
from training.pipeline import ShockDetectionPipeline

# ── 1. Load ───────────────────────────────────────────────────────────────────
PARQUET_PATH = "/mnt/data/TEST_CSV/2000.parquet"   # ← change to your actual file path

loader = GDELTDataLoader(time_col="Day")
loader.load(PARQUET_PATH).sort_by_time()
df = loader.df

# ── 2. Assign column names if integer-indexed ─────────────────────────────────
if isinstance(df.columns[0], int) or str(df.columns[0]).isdigit():
    print(f"[run.py] Integer column names detected — assigning GDELT column names")
    df.columns = GDELT_COLUMNS[: len(df.columns)]

print(f"[run.py] Columns: {df.columns.tolist()[:8]} ...")
print(f"[run.py] Full shape: {df.shape}")

# ── 3. Sample for test run ────────────────────────────────────────────────────
# Remove or increase n when running for real
SAMPLE_N = 50_000
df = df.sample(n=SAMPLE_N, random_state=42).sort_values("Day").reset_index(drop=True)
print(f"[run.py] Sampled {len(df):,} rows for test run")

# ── 4. Build pipeline ─────────────────────────────────────────────────────────
pipe = ShockDetectionPipeline(
    latent_dim    = 16,      # shock embedding size
    n_regimes     = 3,       # unsupervised regimes to discover
    n_iter        = 1,       # ← 1 epoch for test; increase for real run
    lr            = 1e-3,
    time_col      = "Day",
    group_col     = "Actor1CountryCode",
    ohe_threshold = 0.20,
    windowed      = True,    # use windowed trainer (essential for large data)
    window_size   = 512,
)

# ── 5. Fit ────────────────────────────────────────────────────────────────────
pipe.fit(df)

# ── 6. Save model ─────────────────────────────────────────────────────────────
pipe.save("model.pkl")     # saved in current directory (gdelt_ssm/)
print("[run.py] Model saved to model.pkl")

# ── 7. Inference ──────────────────────────────────────────────────────────────
results = pipe.transform(df)

# ── 8. Print results ──────────────────────────────────────────────────────────
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

# ── 9. Load and verify ────────────────────────────────────────────────────────
print("\n[run.py] Verifying save/load round-trip ...")
pipe2   = ShockDetectionPipeline.load("model.pkl")
results2 = pipe2.transform(df)
print(f"[run.py] Loaded model output shape: {results2.shape}  ✓")