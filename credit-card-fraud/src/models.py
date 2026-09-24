"""
Credit Card Fraud Detection — Model Training, Validation & Evaluation
======================================================================
Comprehensive evaluation across three paradigm approaches on the Sparkov dataset:

  1. Rule-Based Heuristic Baseline    (Interpretable deterministic rules)
  2. Isolation Forest                 (Unsupervised non-parametric anomaly detection)
  3. XGBoost Tuned                    (Gradient boosted trees with calibrated class weights)

Validation Protocol:
  - Temporal 3-Way Split:
      * Train (first 85% of fraudTrain) -> model fitting
      * Validation (last 15% of fraudTrain) -> threshold & weight calibration
      * Test (fraudTest.csv, 555,719 rows) -> completely held-out final evaluation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix,
    average_precision_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb

from features import FEATURE_COLS
from evaluate import (
    plot_confusion_matrix, plot_pr_curves, plot_roc_curves,
    plot_feature_importance, plot_model_comparison,
)

os.makedirs("results", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

# Business cost parameters (INR)
COST_FP = 50       # Customer friction & verification expense for false positives
COST_FN = 3000     # Net operational chargeback loss when fraud is missed


# ── 1. Rule-Based Baseline ──────────────────────────────────────────────────
def rule_based_baseline(df: pd.DataFrame) -> tuple:
    """
    Deterministic rule-based heuristics:
      Flag transaction if amount is very high OR amount z-score is high
      OR sudden large amount relative to cardholder average.
    """
    flags = (
        (df["amt"] > 400.0) & (
            (df["amount_zscore"] > 3.0) |
            (df["amt_to_card_avg"] > 4.0) |
            (df["distance_km"] > 100.0)
        )
    )
    preds = flags.astype(int).values
    scores = np.clip(
        (df["amt"] / 1000.0) * 0.4 +
        (df["amount_zscore"].clip(0, 10) / 10.0) * 0.3 +
        (df["distance_km"].clip(0, 500) / 500.0) * 0.3,
        0, 1
    ).values
    return preds, scores


# ── 2. Isolation Forest (Unsupervised) ─────────────────────────────────────
def isolation_forest_detector(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    Unsupervised Anomaly Detection using Isolation Forest.
    Zero ground-truth labels used during fitting.
    """
    continuous_features = [
        "amt", "log_amt", "age", "log_city_pop", "hour",
        "distance_km", "amount_zscore", "amt_to_card_avg", "log_time_since_last_txn"
    ]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[continuous_features])
    X_test  = scaler.transform(test[continuous_features])

    # Domain-derived contamination estimate from train (no peeking into test)
    contamination = float(np.clip(train["is_fraud"].mean(), 0.001, 0.05))

    clf = IsolationForest(
        contamination=contamination,
        n_estimators=150,
        max_samples=256,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train)

    preds_raw = clf.predict(X_test)
    preds = (preds_raw == -1).astype(int)  # 1 = outlier/fraud, 0 = inlier
    raw_scores = -clf.score_samples(X_test)  # higher = more anomalous
    # Normalize to [0, 1] range for curve rendering
    scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
    return preds, scores


# ── 3. XGBoost Classifier ──────────────────────────────────────────────────
def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    scale_pos_weight: float = 4.0,
    threshold: float = 0.5,
) -> tuple:
    """Trains an XGBoost model optimized for area under precision-recall curve."""
    clf = xgb.XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        reg_alpha=0.1,
        reg_lambda=1.0,
    )
    clf.fit(X_train, y_train, verbose=False)

    probs = clf.predict_proba(X_eval)[:, 1]
    preds = (probs >= threshold).astype(int)
    return preds, clf, probs


def find_optimal_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple:
    """Identifies the F1-maximizing decision threshold on validation data."""
    thresholds = np.arange(0.20, 0.85, 0.02)
    best_thresh, best_f1 = 0.5, -1.0
    for t in thresholds:
        p_ = (probs >= t).astype(int)
        f1 = f1_score(y_true, p_, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh, best_f1


# ── Performance Evaluation ─────────────────────────────────────────────────
def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    p   = precision_score(y_true, y_pred, zero_division=0)
    r   = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    ap  = average_precision_score(y_true, y_prob) if y_prob is not None else None
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    cost = fp * COST_FP + fn * COST_FN

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Precision : {p:.4f} ({p*100:.2f}%)   Recall : {r:.4f} ({r*100:.2f}%)")
    print(f"  F1-Score  : {f1:.4f}             Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    if ap is not None:
        print(f"  AUC-PR    : {ap:.4f}             AUC-ROC  : {auc:.4f}")
    print(f"  Confusion Matrix -> TP: {tp:>6,} | FP: {fp:>6,} | FN: {fn:>6,} | TN: {tn:>8,}")
    print(f"  Financial Loss   -> Rs {cost:>10,.0f} (False Alarm: Rs{fp*COST_FP:,.0f} + Missed: Rs{fn*COST_FN:,.0f})")

    return {
        "model": name,
        "precision": p, "recall": r, "f1": f1, "accuracy": acc,
        "auc_pr": ap, "auc_roc": auc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "cost_inr": cost,
    }


# ── Main Pipeline Execution ────────────────────────────────────────────────
if __name__ == "__main__":
    t_start = time.time()
    print("Loading feature datasets...")
    train_full = pd.read_csv("data/features_train.csv", parse_dates=["trans_date_trans_time"])
    test       = pd.read_csv("data/features_test.csv", parse_dates=["trans_date_trans_time"])

    # Temporal split: 85% train, 15% validation
    train_full = train_full.sort_values("trans_date_trans_time").reset_index(drop=True)
    cutoff = train_full["trans_date_trans_time"].quantile(0.85)
    train  = train_full[train_full["trans_date_trans_time"] <= cutoff].copy()
    val    = train_full[train_full["trans_date_trans_time"] >  cutoff].copy()

    print(f"\nDataset Splits:")
    print(f"  Train Set      : {len(train):>9,} rows | {train['is_fraud'].sum():>5,} fraud incidents")
    print(f"  Validation Set : {len(val):>9,} rows | {val['is_fraud'].sum():>5,} fraud incidents")
    print(f"  Test Set       : {len(test):>9,} rows | {test['is_fraud'].sum():>5,} fraud incidents")

    y_val  = val["is_fraud"].values
    y_test = test["is_fraud"].values

    results, pr_data, roc_data = [], [], []

    # 1. Rule-Based Baseline
    print("\n[1/3] Evaluating Rule-Based Heuristic Baseline...")
    rb_preds, rb_scores = rule_based_baseline(test)
    rb_res = evaluate("Rule-Based Baseline", y_test, rb_preds, rb_scores)
    results.append(rb_res)
    plot_confusion_matrix(y_test, rb_preds, "Rule-Based Baseline")
    pr_data.append(("Rule-Based", y_test, rb_scores))
    roc_data.append(("Rule-Based", y_test, rb_scores))

    # 2. Isolation Forest (Unsupervised)
    print("\n[2/3] Training & Evaluating Isolation Forest (Unsupervised)...")
    if_preds, if_scores = isolation_forest_detector(train, test)
    if_res = evaluate("Isolation Forest (Unsupervised)", y_test, if_preds, if_scores)
    results.append(if_res)
    plot_confusion_matrix(y_test, if_preds, "Isolation Forest")
    pr_data.append(("Isolation Forest", y_test, if_scores))
    roc_data.append(("Isolation Forest", y_test, if_scores))

    # 3. XGBoost Validation Calibration
    print("\n[3/3] Training XGBoost with Validation Threshold Calibration...")
    X_train, y_train = train[FEATURE_COLS], train["is_fraud"]
    X_val            = val[FEATURE_COLS]

    _, _, val_probs = train_xgboost(X_train, y_train, X_val, scale_pos_weight=4.0, threshold=0.5)
    opt_thresh, opt_val_f1 = find_optimal_threshold(y_val, val_probs)
    print(f"  Optimal Validation Threshold: {opt_thresh:.2f} (Val F1: {opt_val_f1:.4f})")

    # Retrain on full train dataset with optimal parameters and evaluate on untouched test set
    X_train_full, y_train_full = train_full[FEATURE_COLS], train_full["is_fraud"]
    X_test                     = test[FEATURE_COLS]

    print("Fitting final model on full training corpus...")
    xgb_preds, xgb_model, xgb_probs = train_xgboost(
        X_train_full, y_train_full, X_test,
        scale_pos_weight=4.0, threshold=opt_thresh
    )

    xgb_label = f"XGBoost Tuned (Thresh={opt_thresh:.2f})"
    xgb_res = evaluate(xgb_label, y_test, xgb_preds, xgb_probs)
    results.append(xgb_res)
    plot_confusion_matrix(y_test, xgb_preds, "XGBoost Tuned")
    pr_data.append(("XGBoost Tuned", y_test, xgb_probs))
    roc_data.append(("XGBoost Tuned", y_test, xgb_probs))

    # Save model artifact for production & dashboard inference
    model_save_path = "results/xgb_model.json"
    xgb_model.save_model(model_save_path)
    print(f"  Model artifact saved to {model_save_path}")

    # High-Recall Security Setting (threshold 0.35 - prioritized for minimizing False Negatives)
    xgb_high_recall_preds = (xgb_probs >= 0.35).astype(int)
    xgb_high_recall_res = evaluate("XGBoost High-Recall (Thresh=0.35) [Recommended]", y_test, xgb_high_recall_preds, xgb_probs)
    results.append(xgb_high_recall_res)

    # High-Precision Conservative Setting (threshold 0.75)
    xgb_high_p_preds = (xgb_probs >= 0.75).astype(int)
    xgb_high_p_res = evaluate("XGBoost High-Precision (Thresh=0.75)", y_test, xgb_high_p_preds, xgb_probs)
    results.append(xgb_high_p_res)

    # Save a representative scored sample for fast dashboard loading
    test_sample = test.copy()
    test_sample["xgb_prob"] = xgb_probs
    test_sample["predicted_fraud"] = (xgb_probs >= 0.35).astype(int)
    # Ensure all fraud cases + a sample of legitimate cases are saved
    fraud_cases = test_sample[test_sample["is_fraud"] == 1]
    legit_cases = test_sample[test_sample["is_fraud"] == 0].sample(n=min(5000, len(test_sample[test_sample["is_fraud"] == 0])), random_state=42)
    sample_export = pd.concat([fraud_cases, legit_cases]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    sample_export.to_csv("results/test_sample_scored.csv", index=False)
    print(f"  Exported {len(sample_export):,} scored transactions for interactive dashboard to results/test_sample_scored.csv")

    # Visualizations
    print("\nGenerating visual performance charts for presentation...")
    plot_pr_curves(pr_data)
    plot_roc_curves(roc_data)
    plot_feature_importance(xgb_model, FEATURE_COLS)
    plot_model_comparison(results[:3])  # compare baseline, isolation forest, tuned xgboost

    # Summary table
    print("\n" + "="*80)
    print("  EXECUTIVE SUMMARY — FULL TEST BENCHMARK (555,719 REAL TRANSACTIONS)")
    print("="*80)
    summary_df = pd.DataFrame(results)[[
        "model", "precision", "recall", "f1", "accuracy", "auc_pr", "auc_roc", "cost_inr"
    ]]
    summary_df[["precision", "recall", "f1", "accuracy", "auc_pr", "auc_roc"]] = (
        summary_df[["precision", "recall", "f1", "accuracy", "auc_pr", "auc_roc"]]
        .map(lambda x: f"{x:.4f}" if x is not None else "—")
    )
    summary_df["cost_inr"] = summary_df["cost_inr"].map(lambda x: f"Rs {x:,.0f}")
    print(summary_df.to_string(index=False))

    summary_csv_path = "results/model_comparison.csv"
    pd.DataFrame(results).to_csv(summary_csv_path, index=False)
    print(f"\nMetrics saved to {summary_csv_path}")
    print(f"Visual charts saved to results/plots/")
    print(f"Complete run finished in {time.time()-t_start:.1f}s")

    print("\nTop 10 Most Predictive Features:")
    for feat, imp in sorted(zip(FEATURE_COLS, xgb_model.feature_importances_), key=lambda x: -x[1])[:10]:
        bar = "#" * int(imp * 100)
        print(f"  {feat:<26} {imp:.4f}  {bar}")
