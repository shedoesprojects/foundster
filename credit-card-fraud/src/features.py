"""
Feature Engineering — Real-World Credit Card Fraud Detection
============================================================
Vectorized, leakage-safe feature engineering pipeline for the Kaggle Sparkov dataset.

Features engineered:
  1. Transaction Amount Signals:
     - amt: raw transaction amount
     - log_amt: log-transformed amount to normalize heavy right skew
     - amt_to_card_avg: ratio of transaction amount to cardholder's prior average
     - amount_zscore: standardized z-score of amount vs cardholder's historical baseline

  2. Temporal & Velocity Dynamics:
     - hour: transaction hour (0-23)
     - hour_sin / hour_cos: cyclical 24-hr trigonometric embeddings
     - dayofweek: day of week (0-6)
     - is_weekend: weekend binary indicator
     - log_time_since_last_txn: elapsed time since cardholder's previous transaction

  3. Geographic & Demographic Context:
     - distance_km: exact Haversine great-circle distance (cardholder home to merchant)
     - age: cardholder age derived from date of birth (dob)
     - gender: binary gender encoding
     - log_city_pop: log-transformed local population density

  4. Merchant Category Embeddings:
     - cat_*: 14 distinct one-hot encoded spending categories

LEAKAGE DISCIPLINE:
  All cardholder rolling statistics utilize .shift(1) on time-ordered transactions,
  guaranteeing zero future information leakage into historical baselines.
"""

import os
import time
import numpy as np
import pandas as pd

TRAIN_PATH = "../fraud-detector-v2/data/fraudTrain.csv"
TEST_PATH  = "../fraud-detector-v2/data/fraudTest.csv"

CATEGORIES = [
    "entertainment", "food_dining", "gas_transport", "grocery_net",
    "grocery_pos", "health_fitness", "home", "kids_pets", "misc_net",
    "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel"
]

FEATURE_COLS = [
    "amt", "log_amt", "age", "gender", "log_city_pop",
    "hour", "hour_sin", "hour_cos", "dayofweek", "is_weekend", "distance_km",
    "amount_zscore", "amt_to_card_avg", "log_time_since_last_txn"
] + [f"cat_{cat}" for cat in CATEGORIES]

KEEP_COLS = (
    ["trans_num", "cc_num", "merchant", "category", "amt",
     "trans_date_trans_time", "is_fraud"]
    + FEATURE_COLS
)
KEEP_COLS = list(dict.fromkeys(KEEP_COLS))


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance in kilometers."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies fast, vectorized feature extraction with strict temporal ordering."""
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

    # Demographic & geographic features
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365
    df["gender"] = (df["gender"] == "M").astype(int)
    df["log_city_pop"] = np.log1p(df["city_pop"])
    df["log_amt"] = np.log1p(df["amt"])
    df["distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

    # Temporal features
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofweek"] = df["trans_date_trans_time"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Sort strictly by cardholder and timestamp to guarantee no forward-looking leakage
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)

    # Cardholder historical baseline (strictly shifted by 1 transaction)
    df["prior_amt"] = df.groupby("cc_num")["amt"].shift(1)
    df["card_avg_amt"] = df.groupby("cc_num")["prior_amt"].transform(
        lambda x: x.expanding(min_periods=3).mean()
    )
    df["card_std_amt"] = df.groupby("cc_num")["prior_amt"].transform(
        lambda x: x.expanding(min_periods=3).std()
    )
    df["amount_zscore"] = (
        (df["amt"] - df["card_avg_amt"]) / df["card_std_amt"].replace(0, np.nan)
    ).fillna(0)
    df["amt_to_card_avg"] = (
        df["amt"] / df["card_avg_amt"].replace(0, np.nan)
    ).fillna(1.0)

    # Time interval dynamics
    time_diff = df.groupby("cc_num")["trans_date_trans_time"].diff().dt.total_seconds().fillna(86400)
    df["log_time_since_last_txn"] = np.log1p(np.clip(time_diff, 0, 86400 * 7))

    # One-hot encoded category distribution
    for cat in CATEGORIES:
        df[f"cat_{cat}"] = (df["category"] == cat).astype(float)

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("Loading raw training dataset...")
    t0 = time.time()
    train_raw = pd.read_csv(TRAIN_PATH)
    print(f"  Loaded {len(train_raw):,} train rows in {time.time()-t0:.1f}s | "
          f"{train_raw['is_fraud'].sum():,} fraud ({train_raw['is_fraud'].mean()*100:.2f}%)")

    print("Loading raw test dataset...")
    t1 = time.time()
    test_raw = pd.read_csv(TEST_PATH)
    print(f"  Loaded {len(test_raw):,} test rows in {time.time()-t1:.1f}s | "
          f"{test_raw['is_fraud'].sum():,} fraud ({test_raw['is_fraud'].mean()*100:.2f}%)")

    print("\nEngineering features for training dataset...")
    t2 = time.time()
    train_feat = engineer_features(train_raw)
    print(f"  Train features ready in {time.time()-t2:.1f}s")

    print("Engineering features for test dataset...")
    t3 = time.time()
    test_feat = engineer_features(test_raw)
    print(f"  Test features ready in {time.time()-t3:.1f}s")

    train_feat[KEEP_COLS].to_csv("data/features_train.csv", index=False)
    test_feat[KEEP_COLS].to_csv("data/features_test.csv", index=False)

    print(f"\nSaved -> data/features_train.csv ({os.path.getsize('data/features_train.csv') / (1024*1024):.1f} MB)")
    print(f"Saved -> data/features_test.csv ({os.path.getsize('data/features_test.csv') / (1024*1024):.1f} MB)")
    print(f"\nTotal features engineered: {len(FEATURE_COLS)}")
