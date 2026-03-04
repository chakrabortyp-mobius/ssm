# run.py  (place this inside gdelt_ssm/)
import pandas as pd
from training.pipeline import ShockDetectionPipeline

# ── Load your parquet ──────────────────────────────────────────
df = pd.read_parquet("/home/gaian/Downloads/2000.parquet")

# ── Fit ───────────────────────────────────────────────────────
pipe = ShockDetectionPipeline(
    latent_dim    = 16,       # shock embedding size
    n_regimes     = 3,        # unsupervised regimes
    n_iter        = 100,      # training epochs
    lr            = 1e-3,
    time_col      = "Day",    # your date column name
    group_col     = "Actor1CountryCode",
    ohe_threshold = 0.20,
)

pipe.fit(df)

# ── Inference on new data ──────────────────────────────────────
df_new = pd.read_parquet("path/to/new_file.parquet")
results = pipe.transform(df_new)

# ── Inspect results ────────────────────────────────────────────
print(pipe.encoder_summary())          # what happened to each column
print(pipe.top_shocks(results))        # top 10 detected shocks
print(pipe.regime_summary(results))    # stats per discovered regime
print(results.head())