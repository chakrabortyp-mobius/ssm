"""
reports/generate_report.py
==========================
Generates per-split report folders with CSVs and plots.

Output layout (created under the project root):
    reports/
    ├── train/
    │   ├── report.csv                  full results
    │   ├── top_shocks.csv              top 20 by shock_score
    │   ├── regime_summary.csv          per-regime statistics
    │   ├── silhouette_scores.csv       K=2..10 scores + best K  (train only)
    │   ├── cluster_plot.png            z_0 vs z_1 coloured by regime
    │   ├── shock_score_timeline.png    shock_score vs date (MM-YY x-axis)
    │   └── silhouette_plot.png         bar chart of silhouette scores (train only)
    └── val/
        ├── report.csv
        ├── top_shocks.csv
        ├── regime_summary.csv
        ├── cluster_plot.png
        └── shock_score_timeline.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from typing import Optional

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY   = '#1B3A5C'
BLUE   = '#2E75B6'
GREEN  = '#1E7145'
ORANGE = '#C55A11'
RED    = '#C0392B'
WHITE  = '#FFFFFF'
REGIME_COLORS = [BLUE, ORANGE, GREEN, RED, '#8E44AD',
                 '#16A085', '#D35400', '#2C3E50', '#7F8C8D', '#27AE60']

def _rcolor(k): return REGIME_COLORS[int(k) % len(REGIME_COLORS)]


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse YYYYMMDD int/str → datetime."""
    try:
        return pd.to_datetime(series.astype(str), format='%Y%m%d', errors='coerce')
    except Exception:
        return pd.to_datetime(series, errors='coerce')


# ── Plot 1: Cluster scatter ───────────────────────────────────────────────────

def plot_cluster(results: pd.DataFrame, out_path: str, title: str, best_k: int):
    z_cols = [c for c in results.columns if c.startswith('z_')]
    if len(z_cols) < 2:
        print(f"  ⚠  cluster_plot skipped — need at least 2 z_ columns")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor('#FAFAFA')

    for r in sorted(results['regime'].unique()):
        sub = results[results['regime'] == r]
        ax.scatter(sub[z_cols[0]], sub[z_cols[1]],
                   c=_rcolor(r), alpha=0.35, s=14, linewidths=0,
                   label=f'Regime {r}  (n={len(sub):,})')
        cx, cy = sub[z_cols[0]].mean(), sub[z_cols[1]].mean()
        ax.scatter(cx, cy, c=_rcolor(r), s=240, marker='X',
                   edgecolors='white', linewidths=2, zorder=5)
        ax.annotate(f' R{r}', (cx, cy), fontsize=11,
                    fontweight='bold', color=_rcolor(r), zorder=6)

    high = results[results['shock_score'] > 0.8]
    if len(high):
        ax.scatter(high[z_cols[0]], high[z_cols[1]],
                   c=RED, s=60, marker='*', zorder=4,
                   label=f'shock_score > 0.8  (n={len(high):,})',
                   edgecolors='white', linewidths=0.5)

    ax.set_xlabel('z_0', color=NAVY, fontsize=11)
    ax.set_ylabel('z_1', color=NAVY, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', color=NAVY, pad=12)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close()


# ── Plot 2: Shock score timeline ──────────────────────────────────────────────

def plot_shock_timeline(results: pd.DataFrame, time_col: str,
                        out_path: str, title: str):
    df = results.copy()
    df['_dt'] = _parse_dates(df[time_col])
    df = df.dropna(subset=['_dt']).sort_values('_dt')

    if df.empty:
        print(f"  ⚠  shock_score_timeline skipped — no valid dates")
        return

    # Aggregate to daily max, then weekly if too many points
    df['_date'] = df['_dt'].dt.to_period('D').dt.to_timestamp()
    agg = (df.groupby('_date')
             .agg(shock_score=('shock_score','max'),
                  regime=('regime', lambda x: x.mode()[0]))
             .reset_index())

    if len(agg) > 2000:
        agg = (agg.set_index('_date')
                  .resample('W')
                  .agg(shock_score=('shock_score','max'),
                       regime=('regime', lambda x: x.mode()[0] if len(x) else 0))
                  .reset_index())

    colors = [_rcolor(int(r)) for r in agg['regime']]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor('#FAFAFA')

    ax.bar(agg['_date'], agg['shock_score'], color=colors, alpha=0.75, width=5)
    ax.axhline(0.75, color=RED, lw=1.8, linestyle='--',
               label='High shock threshold (0.75)', zorder=5)
    ax.set_ylim(0, 1.05)

    # MM-YY format on x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
    n_ticks = max(1, len(agg) // 20)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=n_ticks))
    plt.xticks(rotation=45, ha='right', fontsize=8)

    regimes = sorted(df['regime'].unique())
    handles = [Patch(facecolor=_rcolor(int(r)), alpha=0.75,
                     label=f'Regime {r}') for r in regimes]
    handles.append(plt.Line2D([0],[0], color=RED, lw=1.8,
                               linestyle='--', label='Threshold (0.75)'))
    ax.legend(handles=handles, fontsize=9, loc='upper left', framealpha=0.9)

    ax.set_xlabel('Date (MM-YY)', color=NAVY, fontsize=11)
    ax.set_ylabel('shock_score', color=NAVY, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', color=NAVY, pad=12)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close()


# ── Plot 3: Silhouette bar chart (train only) ─────────────────────────────────

def plot_silhouette(silhouette_results: dict, out_path: str):
    scores = silhouette_results['scores']
    best_k = silhouette_results['best_k']
    ks     = sorted(scores.keys())
    vals   = [scores[k] for k in ks]
    colors = [RED if k == best_k else BLUE for k in ks]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor('#FAFAFA')

    bars = ax.bar(ks, vals, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=1.5)
    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8.5,
                fontweight='bold' if ks[i] == best_k else 'normal',
                color=RED if ks[i] == best_k else NAVY)

    ax.set_xticks(ks)
    ax.set_xlabel('Number of Regimes (K)', color=NAVY, fontsize=11)
    ax.set_ylabel('Silhouette Score', color=NAVY, fontsize=11)
    ax.set_title(
        f'Silhouette Score for K=2..{max(ks)}  —  Best K={best_k}'
        f'  (score={scores[best_k]:.4f})',
        fontsize=12, fontweight='bold', color=NAVY, pad=12)
    ax.legend(handles=[
        Patch(facecolor=RED,  alpha=0.85, label=f'Best K={best_k}'),
        Patch(facecolor=BLUE, alpha=0.85, label='Other K'),
    ], fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close()


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_report(
    results:            pd.DataFrame,
    split_name:         str,           # 'train' or 'val'
    pipe,                              # fitted ShockDetectionPipeline
    out_root:           str = 'reports',
    include_silhouette: bool = False,  # True only for train split
):
    """
    Generate all CSVs + plots for one data split.

    Parameters
    ----------
    results            : output of pipe.transform(df_split)
    split_name         : 'train' or 'val'
    pipe               : fitted ShockDetectionPipeline
    out_root           : parent folder (e.g. 'reports')
    include_silhouette : write silhouette CSV + plot (train split only)
    """
    folder = os.path.join(out_root, split_name)
    os.makedirs(folder, exist_ok=True)
    print(f"\n[Report] Generating '{split_name}' report → {folder}/")

    # ── CSVs ─────────────────────────────────────────────────────────────────
    results.to_csv(os.path.join(folder, 'report.csv'), index=False)
    print(f"  ✓ report.csv  ({len(results):,} rows × {results.shape[1]} cols)")

    pipe.top_shocks(results, top_n=20).to_csv(
        os.path.join(folder, 'top_shocks.csv'), index=False)
    print(f"  ✓ top_shocks.csv")

    pipe.regime_summary(results).to_csv(
        os.path.join(folder, 'regime_summary.csv'))
    print(f"  ✓ regime_summary.csv")

    if include_silhouette and pipe.silhouette_results:
        sil_rows = [{'K': k, 'silhouette_score': v,
                     'best': k == pipe.best_k}
                    for k, v in pipe.silhouette_results['scores'].items()]
        pd.DataFrame(sil_rows).to_csv(
            os.path.join(folder, 'silhouette_scores.csv'), index=False)
        print(f"  ✓ silhouette_scores.csv  (best K={pipe.best_k})")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_cluster(
        results,
        out_path=os.path.join(folder, 'cluster_plot.png'),
        title=f'Regime Clusters ({split_name.capitalize()} Split)  —  K={pipe.best_k}',
        best_k=pipe.best_k,
    )
    print(f"  ✓ cluster_plot.png")

    tc = pipe.time_col
    if tc and tc in results.columns:
        plot_shock_timeline(
            results,
            time_col=tc,
            out_path=os.path.join(folder, 'shock_score_timeline.png'),
            title=f'Shock Score vs Time ({split_name.capitalize()} Split)',
        )
        print(f"  ✓ shock_score_timeline.png")
    else:
        print(f"  ⚠  shock_score_timeline skipped (time_col not in results)")

    if include_silhouette and pipe.silhouette_results:
        plot_silhouette(
            pipe.silhouette_results,
            out_path=os.path.join(folder, 'silhouette_plot.png'),
        )
        print(f"  ✓ silhouette_plot.png")

    print(f"[Report] '{split_name}' done  →  {folder}/")
    return folder