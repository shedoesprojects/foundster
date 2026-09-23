"""
Same 3-model comparison as our synthetic-data project, run on REAL data.

Train/test split here is the dataset's OWN provided split (fraudTrain.csv /
fraudTest.csv), which Kaggle already separates by time period -- so we get
time-based separation without having to engineer it ourselves, and it means
our results are directly comparable to any other public work using this
same dataset's standard split.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest
import xgboost as xgb
from features import FEATURE_COLS  # single source of truth -- was previously duplicated here and had drifted out of sync

COST_FALSE_POSITIVE = 50
COST_FALSE_NEGATIVE = 3000


def rule_based_baseline(df):
    flags = (
        (df["amount_zscore"] > 3) |
        (df["category_amt_zscore"] > 5) |
        (df["txn_count_1hr"] > 3)
    )
    return flags.astype(int)


def isolation_forest_detector(train, test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS])
    X_test = scaler.transform(test[FEATURE_COLS])
    contamination = max(train["is_fraud"].mean(), 0.001)  # domain-informed, not peeking at test
    clf = IForest(contamination=contamination, random_state=42, n_estimators=200)
    clf.fit(X_train)
    return clf.predict(X_test)


def xgboost_detector(train, test, scale_pos_weight_mult=1.0, threshold=0.5):
    """scale_pos_weight_mult lets us dial down the automatic imbalance
    correction (full ratio tends to over-flag on real, noisy data).
    threshold lets us pick a cost-optimal cutoff instead of the default 0.5."""
    X_train, y_train = train[FEATURE_COLS], train["is_fraud"]
    X_test = test[FEATURE_COLS]
    ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        scale_pos_weight=ratio * scale_pos_weight_mult, eval_metric="aucpr", random_state=42,
    )
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    return preds, clf, probs


def find_best_threshold_by_f1(y_true, probs, thresholds=None):
    """Sweep thresholds, return the one maximizing F1 -- not cost, since we
    don't have defensible real-world cost figures for this business.
    F1 balances precision and recall without needing that assumption.

    NOTE: this sweep uses the same test set we report final numbers on.
    That's a real limitation (see docs/known_limitations.md) -- properly
    this threshold should be picked on a separate validation split. We
    disclose this rather than quietly presenting it as fully held-out."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.02)
    best_thresh, best_f1 = 0.5, -1
    rows = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        rows.append({"threshold": t, "precision": p, "recall": r, "f1": f1})
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh, best_f1, pd.DataFrame(rows)


def evaluate(name, y_true, y_pred):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\n=== {name} ===")
    print(f"  Precision: {p:.3f}  Recall: {r:.3f}  F1: {f1:.3f}  Accuracy: {acc:.4f}")
    print(f"  Confusion matrix -> TP:{tp}  FP:{fp}  FN:{fn}  TN:{tn}")
    return {"model": name, "precision": p, "recall": r, "f1": f1, "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


if __name__ == "__main__":
    train_full = pd.read_csv("data/features_train.csv")
    test = pd.read_csv("data/features_test.csv")
    train_full["trans_date_trans_time"] = pd.to_datetime(train_full["trans_date_trans_time"])

    # PROPER 3-WAY SPLIT: carve a validation slice out of train, used ONLY
    # for threshold tuning. Test set stays fully held-out until the very
    # final evaluation. This fixes the leakage we flagged earlier -- the
    # threshold is no longer chosen using the same data we report on.
    train_full = train_full.sort_values("trans_date_trans_time").reset_index(drop=True)
    cutoff = train_full["trans_date_trans_time"].quantile(0.85)
    train = train_full[train_full["trans_date_trans_time"] <= cutoff].copy()
    val = train_full[train_full["trans_date_trans_time"] > cutoff].copy()

    print(f"Train: {len(train)} rows ({train['is_fraud'].sum()} fraud) | "
          f"Val: {len(val)} rows ({val['is_fraud'].sum()} fraud) | "
          f"Test: {len(test)} rows ({test['is_fraud'].sum()} fraud)")

    y_val = val["is_fraud"].values
    y_test = test["is_fraud"].values
    results = []

    results.append(evaluate("Rule-based baseline", y_test, rule_based_baseline(test)))

    if_preds = isolation_forest_detector(train, test)
    results.append(evaluate("Isolation Forest (unsupervised)", y_test, if_preds))

    xgb_preds, xgb_model, xgb_probs_test = xgboost_detector(train, test, scale_pos_weight_mult=1.0, threshold=0.5)
    results.append(evaluate("XGBoost (default threshold 0.5)", y_test, xgb_preds))

    for mult in [0.1, 0.25, 0.5]:
        preds, model, probs = xgboost_detector(train, test, scale_pos_weight_mult=mult, threshold=0.5)
        results.append(evaluate(f"XGBoost (scale_pos_weight x{mult})", y_test, preds))

    # tune weight AND threshold on VALIDATION set only, then apply the
    # chosen config ONCE to test set for the final honest number
    print("\n--- Tuning scale_pos_weight + threshold on VALIDATION set ---")
    best_config = {"mult": 1.0, "thresh": 0.5, "f1": -1}
    for mult in [0.05, 0.1, 0.15, 0.25, 0.5, 1.0]:
        _, _, val_probs = xgboost_detector(train, val, scale_pos_weight_mult=mult, threshold=0.5)
        thresh, f1, _ = find_best_threshold_by_f1(y_val, val_probs)
        print(f"  mult={mult}: best_thresh={thresh:.2f}, val_f1={f1:.3f}")
        if f1 > best_config["f1"]:
            best_config = {"mult": mult, "thresh": thresh, "f1": f1}

    print(f"\nChosen config (from validation only): scale_pos_weight_mult={best_config['mult']}, "
          f"threshold={best_config['thresh']:.2f} (val F1={best_config['f1']:.3f})")

    # apply ONCE to the untouched test set -- this is now an honest final number
    final_preds, final_model, final_probs = xgboost_detector(
        train, test, scale_pos_weight_mult=best_config["mult"], threshold=best_config["thresh"])
    results.append(evaluate(
        f"XGBoost (val-tuned: mult={best_config['mult']}, thresh={best_config['thresh']:.2f})",
        y_test, final_preds))

    print("\n=== Summary table (REAL DATA, richer features, leakage-safe tuning) ===")
    summary = pd.DataFrame(results)[["model", "precision", "recall", "f1", "accuracy"]]
    print(summary.to_string(index=False))
    summary.to_csv("data/model_comparison_real.csv", index=False)

    print("\nXGBoost feature importances (final tuned model):")
    for feat, imp in sorted(zip(FEATURE_COLS, final_model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")