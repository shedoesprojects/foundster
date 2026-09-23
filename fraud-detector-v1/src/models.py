import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest
import xgboost as xgb

FEATURES_PATH = "data/features.csv"
FEATURE_COLS = ["amount_zscore", "txn_count_5min", "unique_devices_5min",
                 "is_new_device", "geo_dist_from_home"]

# average cost assumptions for false-positive cost reporting (INR, illustrative)
COST_FALSE_POSITIVE = 50      # customer friction / support cost of wrongly blocking a legit txn
COST_FALSE_NEGATIVE = 3000    # average loss when real fraud is missed


def load_split():
    df = pd.read_csv(FEATURES_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = df["timestamp"].quantile(0.71)  # ~ last 4 of 14 days
    train = df[df["timestamp"] <= cutoff].copy()
    test = df[df["timestamp"] > cutoff].copy()
    print(f"Train: {len(train)} rows ({train['is_fraud'].sum()} fraud) | "
          f"Test: {len(test)} rows ({test['is_fraud'].sum()} fraud)")
    return train, test


def rule_based_baseline(df):
    """Flag if velocity is high OR amount is a big outlier OR device+geo is novel & far."""
    flags = (
        (df["txn_count_5min"] > 8) |
        (df["amount_zscore"].abs() > 5) |
        ((df["is_new_device"] == 1) & (df["geo_dist_from_home"] > 0.5))
    )
    return flags.astype(int)


def isolation_forest_detector(train, test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS])
    X_test = scaler.transform(test[FEATURE_COLS])

    # contamination = expected fraud rate; we set it from domain knowledge (~0.5%),
    # NOT from peeking at test labels -- that would be leakage
    clf = IForest(contamination=0.005, random_state=42, n_estimators=200)
    clf.fit(X_train)  # UNSUPERVISED: labels never passed in here
    preds = clf.predict(X_test)  # 1 = outlier/fraud, 0 = normal (PyOD convention)
    return preds


def xgboost_detector(train, test):
    X_train, y_train = train[FEATURE_COLS], train["is_fraud"]
    X_test = test[FEATURE_COLS]

    # scale_pos_weight compensates for severe class imbalance
    ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        scale_pos_weight=ratio, eval_metric="aucpr", random_state=42,
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return preds, clf


def evaluate(name, y_true, y_pred):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE
    print(f"\n=== {name} ===")
    print(f"  Precision: {p:.3f}  Recall: {r:.3f}  F1: {f1:.3f}")
    print(f"  Confusion matrix -> TP:{tp}  FP:{fp}  FN:{fn}  TN:{tn}")
    print(f"  Estimated cost (FP*Rs{COST_FALSE_POSITIVE} + FN*Rs{COST_FALSE_NEGATIVE}): Rs {total_cost:,.0f}")
    return {"model": name, "precision": p, "recall": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "cost": total_cost}


if __name__ == "__main__":
    train, test = load_split()
    y_test = test["is_fraud"].values

    results = []

    baseline_preds = rule_based_baseline(test)
    results.append(evaluate("Rule-based baseline", y_test, baseline_preds))

    if_preds = isolation_forest_detector(train, test)
    results.append(evaluate("Isolation Forest (unsupervised)", y_test, if_preds))

    xgb_preds, xgb_model = xgboost_detector(train, test)
    results.append(evaluate("XGBoost (supervised ceiling)", y_test, xgb_preds))

    print("\n=== Summary table ===")
    summary = pd.DataFrame(results)[["model", "precision", "recall", "f1", "cost"]]
    print(summary.to_string(index=False))
    summary.to_csv("data/model_comparison.csv", index=False)
    print("\nSaved comparison to data/model_comparison.csv")

    print("\nXGBoost feature importances (explainability):")
    for feat, imp in sorted(zip(FEATURE_COLS, xgb_model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")