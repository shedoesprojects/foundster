"""
Evaluation & Visualisation — Credit Card Fraud Detection
=========================================================
Generates presentation-ready, high-resolution visual plots:
  - confusion_matrix_<model>.png
  - pr_curve_comparison.png        (Precision-Recall curves with Average Precision)
  - roc_curve_comparison.png       (ROC curves with AUC metrics)
  - feature_importance.png         (Ranked predictive feature impact)
  - model_comparison.png           (Precision / Recall / F1 paradigm benchmark)
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve,
    average_precision_score, roc_auc_score,
)

PLOT_DIR = "results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

STYLE    = "dark_background"
ACCENT   = "#00D4FF"      # Cyan
ACCENT2  = "#FF6B6B"      # Coral Red
ACCENT3  = "#FFD166"      # Gold
ACCENT4  = "#06D6A0"      # Emerald Green
FONTSIZE = 12


def _safe_fname(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name.lower()).strip("_")


# ── 1. Confusion Matrix ────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name: str, save=True):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax)

        labels = ["Legit (0)", "Fraud (1)"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=FONTSIZE)
        ax.set_yticklabels(labels, fontsize=FONTSIZE)
        ax.set_xlabel("Predicted Label", fontsize=FONTSIZE, labelpad=8)
        ax.set_ylabel("True Label", fontsize=FONTSIZE, labelpad=8)

        for (i, j), val in np.ndenumerate(cm):
            cell_color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=cell_color)

        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)
        fig.tight_layout()

        if save:
            fname = os.path.join(PLOT_DIR, f"confusion_matrix_{_safe_fname(model_name)}.png")
            fig.savefig(fname, dpi=200, bbox_inches="tight")
            print(f"  Saved -> {fname}")
        plt.close(fig)


# ── 2. Precision-Recall Curves ─────────────────────────────────────────────
def plot_pr_curves(pr_data: list, save=True):
    """pr_data: list of (name, y_true, y_prob) tuples."""
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        palette = [ACCENT, ACCENT3, ACCENT4, ACCENT2, "#C77DFF"]

        for (name, y_true, y_prob), color in zip(pr_data, palette):
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            ax.plot(rec, prec, color=color, lw=2.2, label=f"{name} (AP = {ap:.3f})")

        baseline = y_true.mean()
        ax.axhline(baseline, linestyle="--", color="gray", lw=1.2,
                   label=f"Random Chance (P = {baseline:.4f})")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall (Detection Rate)", fontsize=FONTSIZE)
        ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=FONTSIZE)
        ax.set_title("Precision-Recall Curve Comparison", fontsize=14, pad=12)
        ax.legend(fontsize=10.5, loc="lower left")
        ax.grid(alpha=0.25)
        fig.tight_layout()

        if save:
            fname = os.path.join(PLOT_DIR, "pr_curve_comparison.png")
            fig.savefig(fname, dpi=200, bbox_inches="tight")
            print(f"  Saved -> {fname}")
        plt.close(fig)


# ── 3. ROC Curves ──────────────────────────────────────────────────────────
def plot_roc_curves(roc_data: list, save=True):
    """roc_data: list of (name, y_true, y_prob) tuples."""
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        palette = [ACCENT, ACCENT3, ACCENT4, ACCENT2, "#C77DFF"]

        for (name, y_true, y_prob), color in zip(roc_data, palette):
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{name} (AUC = {auc:.4f})")

        ax.plot([0, 1], [0, 1], "w--", lw=1.2, label="Random Guess (AUC = 0.50)")
        ax.set_xlim([-0.01, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=FONTSIZE)
        ax.set_ylabel("True Positive Rate (Recall)", fontsize=FONTSIZE)
        ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, pad=12)
        ax.legend(fontsize=10.5, loc="lower right")
        ax.grid(alpha=0.25)
        fig.tight_layout()

        if save:
            fname = os.path.join(PLOT_DIR, "roc_curve_comparison.png")
            fig.savefig(fname, dpi=200, bbox_inches="tight")
            print(f"  Saved -> {fname}")
        plt.close(fig)


# ── 4. Feature Importance ──────────────────────────────────────────────────
def plot_feature_importance(model, feature_cols: list, top_n: int = 15, save=True):
    importances = model.feature_importances_
    pairs = sorted(zip(feature_cols, importances), key=lambda x: x[1])[-top_n:]
    names  = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8.5, 6))
        bars = ax.barh(names, values, color=ACCENT, edgecolor="none", height=0.65)

        max_val = max(values)
        for bar, val in zip(bars, values):
            ax.text(val + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9.5, color="white")

        ax.set_xlim([0, max_val * 1.15])
        ax.set_xlabel("Feature Importance (Normalized Weight)", fontsize=FONTSIZE)
        ax.set_title(f"Top {top_n} Predictive Fraud Indicators (XGBoost)", fontsize=13, pad=12)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()

        if save:
            fname = os.path.join(PLOT_DIR, "feature_importance.png")
            fig.savefig(fname, dpi=200, bbox_inches="tight")
            print(f"  Saved -> {fname}")
        plt.close(fig)


# ── 5. Model Comparison Bar Chart ──────────────────────────────────────────
def plot_model_comparison(results: list, save=True):
    names      = [r["model"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls    = [r["recall"] for r in results]
    f1s        = [r["f1"] for r in results]

    x = np.arange(len(names))
    w = 0.24

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar(x - w, precisions, w, label="Precision", color=ACCENT)
        ax.bar(x,     recalls,    w, label="Recall",    color=ACCENT3)
        ax.bar(x + w, f1s,        w, label="F1-Score",  color=ACCENT4)

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10.5, rotation=10, ha="right")
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.set_title("Performance Benchmark Across Detection Paradigms", fontsize=14, pad=14)
        ax.legend(fontsize=11, loc="upper left")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()

        if save:
            fname = os.path.join(PLOT_DIR, "model_comparison.png")
            fig.savefig(fname, dpi=200, bbox_inches="tight")
            print(f"  Saved -> {fname}")
        plt.close(fig)
