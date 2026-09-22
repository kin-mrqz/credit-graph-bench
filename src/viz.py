"""Plot helpers used by the notebooks. Every function returns the matplotlib Figure."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_roc_pr(y, preds: dict[str, np.ndarray], title: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    y = np.asarray(y).ravel()
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=name)
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate", title=f"ROC {title}")
    axes[1].axhline(y.mean(), color="k", ls="--", lw=0.8, label=f"base rate {y.mean():.3f}")
    axes[1].set(xlabel="recall", ylabel="precision", title=f"PR {title}")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_reliability(tables: dict[str, pd.DataFrame], title: str = ""):
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, t in tables.items():
        ax.plot(t["pred_mean"], t["obs_rate"], marker="o", label=name)
    lim = max([t["pred_mean"].max() for t in tables.values()] + [t["obs_rate"].max() for t in tables.values()] + [0.05])
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set(xlabel="mean predicted PD", ylabel="observed default rate", title=f"Reliability {title}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_degree_hist(degrees: dict[str, np.ndarray], title: str = ""):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, d in degrees.items():
        d = np.asarray(d)
        ax.hist(np.log10(d[d > 0]), bins=40, alpha=0.5, label=f"{name} (isolated {np.mean(d == 0):.1%})")
    ax.set(xlabel="log10 degree", ylabel="nodes", title=f"Degree distribution {title}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_intervals(rows: pd.DataFrame, value: str = "diff", lo: str = "ci_low", hi: str = "ci_high", label: str = "model",
                   title: str = "", xlabel: str = "metric difference vs baseline"):
    """Horizontal error bars, one per row. Rows with intervals excluding zero are drawn darker."""
    rows = rows.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(rows) + 1.5))
    for i, r in rows.iterrows():
        sig = (r[lo] > 0) or (r[hi] < 0)
        ax.errorbar(r[value], i, xerr=[[r[value] - r[lo]], [r[hi] - r[value]]], fmt="o",
                    color="C0" if sig else "grey", capsize=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows[label].astype(str))
    ax.invert_yaxis()
    ax.set(xlabel=xlabel, title=title)
    fig.tight_layout()
    return fig


def plot_training_curves(history: list[dict], keys=("train_loss", "val_auc"), title: str = ""):
    if not history:
        fig, ax = plt.subplots(); ax.set_title("no history"); return fig
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.6))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        if k in h.columns:
            ax.plot(h["epoch"], h[k], marker=".")
        ax.set(xlabel="epoch", ylabel=k, title=f"{k} {title}")
    fig.tight_layout()
    return fig


def plot_bar(series: pd.Series, title: str = "", ylabel: str = "", top: int | None = None, horizontal: bool = True):
    s = series.sort_values(ascending=False)
    if top:
        s = s.head(top)
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(s) + 1.2) if horizontal else (8, 4))
    (s.iloc[::-1].plot.barh(ax=ax) if horizontal else s.plot.bar(ax=ax))
    ax.set(title=title, xlabel=ylabel if horizontal else "", ylabel="" if horizontal else ylabel)
    fig.tight_layout()
    return fig


def plot_cohort(df: pd.DataFrame, metric: str = "auc", title: str = ""):
    fig, ax1 = plt.subplots(figsize=(7, 3.8))
    ax1.plot(df["cohort"].astype(str), df[metric], marker="o", label=metric)
    ax1.set(xlabel="cohort", ylabel=metric, title=title)
    ax2 = ax1.twinx()
    ax2.bar(df["cohort"].astype(str), df["base_rate"], alpha=0.2, color="grey", label="base rate")
    ax2.set_ylabel("base rate")
    ax1.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def savefig(fig, run_dir, name: str):
    from pathlib import Path

    d = Path(run_dir) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    return path
