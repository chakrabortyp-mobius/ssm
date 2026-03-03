"""
tests/test_all.py
=================
Unit tests for every module.
Run from project root:

    cd gdelt_ssm
    python -m tests.test_all

Each test is a standalone function.  A PASS/FAIL summary is printed at the end.
No external test framework required.
"""

import sys
import traceback
import numpy as np
import pandas as pd

# ── make sure project root is on path ────────────────────────────────────────
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═════════════════════════════════════════════════════════════════════════════
# Test runner
# ═════════════════════════════════════════════════════════════════════════════

_results = []

def run(fn):
    """Decorator: register and auto-run a test function."""
    name = fn.__name__
    try:
        fn()
        _results.append((name, "PASS", ""))
        print(f"  ✓  {name}")
    except Exception as e:
        tb = traceback.format_exc()
        _results.append((name, "FAIL", str(e)))
        print(f"  ✗  {name}")
        print(f"     {e}")
    return fn


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _make_df(T=100, inject_shocks=True):
    """Create a minimal synthetic GDELT-like DataFrame."""
    np.random.seed(0)
    df = pd.DataFrame({
        "Day":               pd.date_range("2000-01-01", periods=T, freq="W")
                             .strftime("%Y%m%d").astype(int),
        "Actor1CountryCode": np.random.choice(["AFG", "IRQ", "SYR", "USA"], T),
        "GoldsteinScale":    np.random.normal(0, 2, T),
        "NumMentions":       np.random.poisson(5, T),
        "AvgTone":           np.random.normal(0, 4, T),
        "SourceURL":         [f"http://x{i}.com" for i in range(T)],  # should be dropped
    })
    if inject_shocks:
        df.loc[30:32, "GoldsteinScale"] = -9.5
        df.loc[30:32, "NumMentions"]    = 90
        df.loc[30:32, "AvgTone"]        = -15.0
    return df


# ═════════════════════════════════════════════════════════════════════════════
# utils/activations
# ═════════════════════════════════════════════════════════════════════════════

print("\n── utils/activations ──────────────────────────────────────────────")

@run
def test_sigmoid_range():
    from utils.activations import sigmoid
    x = np.array([-100, -1, 0, 1, 100], dtype=float)
    y = sigmoid(x)
    assert (y >= 0).all() and (y <= 1).all(), "sigmoid out of [0,1]"
    assert abs(y[2] - 0.5) < 1e-6, "sigmoid(0) != 0.5"

@run
def test_softplus_positive():
    from utils.activations import softplus
    x = np.linspace(-5, 5, 50)
    assert (softplus(x) > 0).all(), "softplus must be strictly positive"

@run
def test_tanh_grad_range():
    from utils.activations import tanh_grad
    x = np.linspace(-3, 3, 50)
    g = tanh_grad(x)
    assert (g >= 0).all() and (g <= 1).all(), "tanh_grad must be in [0,1]"

@run
def test_softplus_grad_equals_sigmoid():
    from utils.activations import softplus_grad, sigmoid
    x = np.linspace(-3, 3, 20)
    assert np.allclose(softplus_grad(x), sigmoid(x), atol=1e-6), \
        "softplus_grad should equal sigmoid"


# ═════════════════════════════════════════════════════════════════════════════
# utils/adam
# ═════════════════════════════════════════════════════════════════════════════

print("\n── utils/adam ─────────────────────────────────────────────────────")

@run
def test_adam_reduces_loss():
    from utils.adam import AdamParam
    w = AdamParam(np.array([1.0, -1.0, 2.0]), lr=0.1)
    # Gradient of L = ||w||^2 is 2*w
    for _ in range(200):
        grad = 2.0 * w.data
        w.step(grad)
    assert np.linalg.norm(w.data) < 0.1, "Adam did not converge weights to ~0"

@run
def test_adam_shape_mismatch():
    from utils.adam import AdamParam
    w = AdamParam(np.zeros((3, 4)), lr=1e-3)
    try:
        w.step(np.zeros((2, 4)))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

@run
def test_adam_step_counter():
    from utils.adam import AdamParam
    w = AdamParam(np.zeros(5), lr=1e-3)
    for i in range(7):
        w.step(np.ones(5))
    assert w.t == 7


# ═════════════════════════════════════════════════════════════════════════════
# data/encoder
# ═════════════════════════════════════════════════════════════════════════════

print("\n── data/encoder ───────────────────────────────────────────────────")

@run
def test_encoder_fit_transform_shape():
    from data.encoder import AutoColumnEncoder
    df = _make_df(T=300)
    enc = AutoColumnEncoder(time_col="Day", ohe_threshold=0.20)
    X = enc.fit_transform(df)
    assert X.ndim == 2
    assert X.shape[0] == 300

@run
def test_encoder_drops_sparse_strings():
    from data.encoder import AutoColumnEncoder
    df = _make_df(T=300)
    enc = AutoColumnEncoder(time_col="Day", ohe_threshold=0.20, min_rows_for_strategy=100)
    enc.fit(df)
    # SourceURL has unique values → should be dropped or OHE with 0 cats
    decision = enc.col_decisions.get("SourceURL")
    if decision == "ohe":
        assert len(enc.ohe_categories["SourceURL"]) == 0, \
            "Sparse URL should produce 0 OHE categories"
    else:
        assert decision == "drop", f"SourceURL should be dropped, got {decision}"

@run
def test_encoder_ohe_columns():
    from data.encoder import AutoColumnEncoder
    df = _make_df(T=300)
    enc = AutoColumnEncoder(time_col="Day", ohe_threshold=0.20, min_rows_for_strategy=100)
    enc.fit(df)
    # Actor1CountryCode has values appearing >20% → should be OHE
    assert enc.col_decisions.get("Actor1CountryCode") == "ohe"

@run
def test_encoder_no_nans_in_output():
    from data.encoder import AutoColumnEncoder
    df = _make_df(T=200)
    df.loc[10:15, "GoldsteinScale"] = np.nan   # inject some NaNs
    enc = AutoColumnEncoder(time_col="Day")
    X = enc.fit_transform(df)
    assert not np.isnan(X).any(), "Encoder output contains NaNs"

@run
def test_encoder_inference_same_shape():
    from data.encoder import AutoColumnEncoder
    df_train = _make_df(T=200)
    df_infer = _make_df(T=50, inject_shocks=False)
    enc = AutoColumnEncoder(time_col="Day")
    enc.fit(df_train)
    X_train = enc.transform(df_train)
    X_infer = enc.transform(df_infer)
    assert X_train.shape[1] == X_infer.shape[1], \
        "Train and infer encoded dims must match"

@run
def test_encoder_summary_returns_dataframe():
    from data.encoder import AutoColumnEncoder
    df = _make_df(T=200)
    enc = AutoColumnEncoder(time_col="Day")
    enc.fit(df)
    s = enc.summary()
    assert isinstance(s, pd.DataFrame)
    assert "column" in s.columns and "decision" in s.columns


# ═════════════════════════════════════════════════════════════════════════════
# data/dataloader
# ═════════════════════════════════════════════════════════════════════════════

print("\n── data/dataloader ────────────────────────────────────────────────")

@run
def test_dataloader_from_dataframe():
    from data.dataloader import GDELTDataLoader
    df = _make_df(T=150)
    loader = GDELTDataLoader(time_col="Day")
    loader.load_dataframe(df)
    assert loader.shape == (150, len(df.columns))

@run
def test_dataloader_sort_by_time():
    from data.dataloader import GDELTDataLoader
    df = _make_df(T=100)
    df_shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
    loader = GDELTDataLoader(time_col="Day")
    loader.load_dataframe(df_shuffled).sort_by_time()
    days = loader.df["Day"].values
    assert (days[1:] >= days[:-1]).all(), "Not sorted after sort_by_time()"

@run
def test_dataloader_split_by_fraction():
    from data.dataloader import GDELTDataLoader
    df = _make_df(T=100)
    loader = GDELTDataLoader(time_col="Day")
    loader.load_dataframe(df)
    train, infer = loader.split_by_fraction(0.8)
    assert len(train) == 80 and len(infer) == 20

@run
def test_dataloader_split_by_date():
    from data.dataloader import GDELTDataLoader
    df = _make_df(T=100)
    loader = GDELTDataLoader(time_col="Day")
    loader.load_dataframe(df)
    train, infer = loader.split_by_date("20010101")
    assert len(train) + len(infer) == 100
    assert (train["Day"] < 20010101).all()
    assert (infer["Day"] >= 20010101).all()

@run
def test_dataloader_filter_group():
    from data.dataloader import GDELTDataLoader
    df = _make_df(T=200)
    loader = GDELTDataLoader(time_col="Day")
    loader.load_dataframe(df)
    loader.filter_group("Actor1CountryCode", "AFG")
    assert (loader.df["Actor1CountryCode"] == "AFG").all()


# ═════════════════════════════════════════════════════════════════════════════
# model/ssm
# ═════════════════════════════════════════════════════════════════════════════

print("\n── model/ssm ──────────────────────────────────────────────────────")

@run
def test_ssm_forward_shapes():
    from model.ssm import SelectiveSSM
    T, d, s = 50, 8, 4
    ssm = SelectiveSSM(d_input=d, d_state=s)
    X = np.random.randn(T, d).astype(np.float32)
    Z, deltas, recon, _ = ssm.forward(X, store_cache=False)
    assert Z.shape == (T, s), f"Z shape {Z.shape}"
    assert deltas.shape == (T,)
    assert recon.shape == (T, d)

@run
def test_ssm_deltas_positive():
    from model.ssm import SelectiveSSM
    ssm = SelectiveSSM(d_input=6, d_state=4)
    X = np.random.randn(30, 6).astype(np.float32)
    _, deltas, _, _ = ssm.forward(X)
    assert (deltas > 0).all(), "All deltas must be > 0 (softplus output)"

@run
def test_ssm_loss_decreases():
    from model.ssm import SelectiveSSM
    np.random.seed(1)
    X = np.random.randn(80, 6).astype(np.float32)
    ssm = SelectiveSSM(d_input=6, d_state=4, lr=1e-2)
    losses = [ssm.train_step(X) for _ in range(30)]
    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

@run
def test_ssm_no_nans_in_output():
    from model.ssm import SelectiveSSM
    X = np.random.randn(60, 10).astype(np.float32)
    ssm = SelectiveSSM(d_input=10, d_state=6)
    Z, deltas, recon = ssm.predict(X)
    assert not np.isnan(Z).any(), "NaN in Z"
    assert not np.isnan(deltas).any(), "NaN in deltas"
    assert not np.isnan(recon).any(), "NaN in recon"

@run
def test_ssm_cache_stored_in_train():
    from model.ssm import SelectiveSSM
    ssm = SelectiveSSM(d_input=5, d_state=3)
    X = np.random.randn(20, 5).astype(np.float32)
    _, _, _, cache = ssm.forward(X, store_cache=True)
    assert cache is not None and len(cache) == 20

@run
def test_ssm_param_norms_dict():
    from model.ssm import SelectiveSSM
    ssm = SelectiveSSM(d_input=5, d_state=3)
    norms = ssm.param_norms()
    assert isinstance(norms, dict)
    assert "A_log" in norms and "W_in" in norms


# ═════════════════════════════════════════════════════════════════════════════
# model/regime
# ═════════════════════════════════════════════════════════════════════════════

print("\n── model/regime ───────────────────────────────────────────────────")

@run
def test_regime_fit_predict():
    from model.regime import RegimeDiscovery
    Z = np.vstack([
        np.random.randn(30, 4) + np.array([5, 0, 0, 0]),
        np.random.randn(30, 4) + np.array([-5, 0, 0, 0]),
        np.random.randn(30, 4) + np.array([0, 5, 0, 0]),
    ])
    rd = RegimeDiscovery(n_regimes=3)
    rd.fit(Z)
    labels = rd.predict(Z)
    assert labels.shape == (90,)
    # Regimes should separate the three clusters
    for k in range(3):
        assert (labels == k).sum() > 0, f"Regime {k} is empty"

@run
def test_regime_proba_sums_to_one():
    from model.regime import RegimeDiscovery
    Z = np.random.randn(50, 4)
    rd = RegimeDiscovery(n_regimes=3)
    rd.fit(Z)
    probs = rd.predict_proba(Z)
    assert probs.shape == (50, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

@run
def test_regime_centroid_distances_shape():
    from model.regime import RegimeDiscovery
    Z = np.random.randn(40, 6)
    rd = RegimeDiscovery(n_regimes=3)
    rd.fit(Z)
    D = rd.centroid_distances()
    assert D.shape == (3, 3)
    assert np.allclose(np.diag(D), 0.0)   # distance to self = 0


# ═════════════════════════════════════════════════════════════════════════════
# model/scorer
# ═════════════════════════════════════════════════════════════════════════════

print("\n── model/scorer ───────────────────────────────────────────────────")

@run
def test_scorer_output_range():
    from model.scorer import ChangePointScorer
    T = 100
    Z = np.random.randn(T, 8)
    deltas = np.abs(np.random.randn(T))
    recon_err = np.abs(np.random.randn(T))
    sc = ChangePointScorer()
    scores = sc.score(Z, deltas, recon_err)
    assert scores.shape == (T,)
    assert (scores >= 0).all() and (scores <= 1).all()

@run
def test_scorer_detects_injected_shock():
    from model.scorer import ChangePointScorer
    T = 100
    Z = np.random.randn(T, 8) * 0.1
    deltas = np.ones(T) * 0.5
    recon_err = np.ones(T) * 0.1
    # Inject shock at t=50
    Z[50] = np.ones(8) * 10.0
    deltas[50] = 10.0
    recon_err[50] = 20.0
    sc = ChangePointScorer()
    scores = sc.score(Z, deltas, recon_err)
    assert scores[50] == scores.max(), "Injected shock should have highest score"

@run
def test_scorer_signal_breakdown_keys():
    from model.scorer import ChangePointScorer
    T = 50
    sc = ChangePointScorer()
    bd = sc.signal_breakdown(
        np.random.randn(T, 4), np.random.rand(T), np.random.rand(T)
    )
    for key in ["delta_norm", "velocity_norm", "recon_norm", "shock_score"]:
        assert key in bd


# ═════════════════════════════════════════════════════════════════════════════
# training/trainer
# ═════════════════════════════════════════════════════════════════════════════

print("\n── training/trainer ───────────────────────────────────────────────")

@run
def test_trainer_loss_decreases():
    from model.ssm import SelectiveSSM
    from training.trainer import Trainer
    X = np.random.randn(80, 6).astype(np.float32)
    ssm = SelectiveSSM(d_input=6, d_state=4, lr=1e-2)
    tr = Trainer(ssm=ssm, n_iter=30, log_every=30)
    tr.fit(X)
    assert tr.loss_history[-1] < tr.loss_history[0]

@run
def test_trainer_early_stopping():
    from model.ssm import SelectiveSSM
    from training.trainer import Trainer
    X = np.random.randn(60, 5).astype(np.float32)
    ssm = SelectiveSSM(d_input=5, d_state=3, lr=0.0)   # zero lr → never improves
    tr = Trainer(ssm=ssm, n_iter=100, patience=5, log_every=100)
    tr.fit(X)
    assert len(tr.loss_history) < 100, "Early stopping should have fired"

@run
def test_trainer_summary_keys():
    from model.ssm import SelectiveSSM
    from training.trainer import Trainer
    X = np.random.randn(40, 4).astype(np.float32)
    ssm = SelectiveSSM(d_input=4, d_state=3, lr=1e-3)
    tr = Trainer(ssm=ssm, n_iter=5, log_every=5)
    tr.fit(X)
    s = tr.loss_summary()
    for k in ["initial", "final", "min", "reduction_%", "n_epochs"]:
        assert k in s


# ═════════════════════════════════════════════════════════════════════════════
# training/pipeline (integration test)
# ═════════════════════════════════════════════════════════════════════════════

print("\n── training/pipeline (integration) ───────────────────────────────")

@run
def test_pipeline_fit_transform():
    from training.pipeline import ShockDetectionPipeline
    df = _make_df(T=150)
    pipe = ShockDetectionPipeline(
        latent_dim=4, n_regimes=3, n_iter=5, lr=1e-3,
        time_col="Day", group_col="Actor1CountryCode"
    )
    results = pipe.fit_transform(df)
    assert len(results) == 150
    assert "shock_score" in results.columns
    assert "regime" in results.columns
    assert all(f"z_{i}" in results.columns for i in range(4))

@run
def test_pipeline_inference_after_fit():
    from training.pipeline import ShockDetectionPipeline
    df_train = _make_df(T=150)
    df_infer = _make_df(T=40, inject_shocks=False)
    pipe = ShockDetectionPipeline(
        latent_dim=4, n_regimes=3, n_iter=5, time_col="Day"
    )
    pipe.fit(df_train)
    results = pipe.transform(df_infer)
    assert len(results) == 40

@run
def test_pipeline_shock_scores_in_range():
    from training.pipeline import ShockDetectionPipeline
    df = _make_df(T=120)
    pipe = ShockDetectionPipeline(latent_dim=4, n_regimes=3, n_iter=5, time_col="Day")
    results = pipe.fit_transform(df)
    scores = results["shock_score"].values
    assert (scores >= 0).all() and (scores <= 1).all()

@run
def test_pipeline_injected_shock_detected():
    from training.pipeline import ShockDetectionPipeline
    df = _make_df(T=200, inject_shocks=True)   # shock at rows 30-32
    pipe = ShockDetectionPipeline(
        latent_dim=6, n_regimes=3, n_iter=20, lr=5e-3, time_col="Day"
    )
    results = pipe.fit_transform(df)
    top_idx = results["shock_score"].idxmax()
    # The highest shock should be near our injected rows
    assert abs(top_idx - 31) <= 5, \
        f"Top shock at row {top_idx}, expected near row 31"

@run
def test_pipeline_regime_summary():
    from training.pipeline import ShockDetectionPipeline
    df = _make_df(T=150)
    pipe = ShockDetectionPipeline(latent_dim=4, n_regimes=3, n_iter=5, time_col="Day")
    results = pipe.fit_transform(df)
    summary = pipe.regime_summary(results)
    assert len(summary) <= 3
    assert "avg_shock" in summary.columns

@run
def test_pipeline_signal_breakdown():
    from training.pipeline import ShockDetectionPipeline
    df = _make_df(T=100)
    pipe = ShockDetectionPipeline(latent_dim=4, n_regimes=2, n_iter=5, time_col="Day")
    pipe.fit(df)
    results = pipe.transform(df)
    bd = pipe.signal_breakdown(results, df)
    assert "delta_norm" in bd.columns
    assert len(bd) == 100


# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════

def print_summary():
    total  = len(_results)
    passed = sum(1 for _, s, _ in _results if s == "PASS")
    failed = total - passed
    print(f"\n{'═'*55}")
    print(f"  Results: {passed}/{total} passed  |  {failed} failed")
    print(f"{'═'*55}")
    if failed:
        print("\nFailed tests:")
        for name, status, err in _results:
            if status == "FAIL":
                print(f"  ✗ {name}: {err}")
    print()


print_summary()
