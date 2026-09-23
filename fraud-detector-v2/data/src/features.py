"""
Feature engineering for the REAL Kaggle Sparkov credit-card fraud dataset.

KEY DIFFERENCE FROM OUR SYNTHETIC VERSION (read this first):

In our synthetic data, we built MERCHANT-relative features (is this
transaction weird for THIS merchant's normal pattern). That made sense
for a "protect this merchant" framing.

Real card fraud works the other way: a stolen card gets used across
MANY merchants. So here we build CARDHOLDER-relative features (is this
transaction weird for THIS card's normal pattern) -- this is the
standard framing in real fraud detection and is a more honest match to
how card-present/card-not-present fraud actually happens.

Features built, all per cc_num (cardholder), using only PAST transactions:
  1. amount_zscore       -- amount vs this card's own rolling mean/std
  2. txn_count_1hr        -- velocity: transactions by this card in trailing 1hr
                             (Sparkov transactions are sparser in time than our
                             synthetic stream, so we widen the window from
                             5min to 1hr -- always size your window to your
                             data's actual density, don't copy a constant blindly)
  3. is_new_merchant      -- has this card ever paid this merchant before
  4. distance_from_home   -- real geographic distance (km, haversine) between
                             cardholder's home lat/long and merchant's lat/long
                             (this is BETTER than our synthetic proxy -- real
                             coordinates, real formula, not a degrees approximation)
  5. category_amt_zscore  -- amount vs this card's own average WITHIN that
                             spending category (a Rs5000 "grocery_pos" txn is
                             far weirder than a Rs5000 "travel" txn for the
                             same person -- category context matters)

LEAKAGE DISCIPLINE (same principle as before): every rolling/expanding
stat uses .shift(1) before computing, so a transaction never sees itself
in its own baseline.
"""

import pandas as pd
import numpy as np

TRAIN_PATH = "data/fraudTrain.csv"
TEST_PATH = "data/fraudTest.csv"


def haversine_km(lat1, lon1, lat2, lon2):
    """Real great-circle distance in km -- more honest than our synthetic
    degrees-based proxy."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_raw(path, sample_frac=None):
    df = pd.read_csv(path)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    if sample_frac:
        # sample by cardholder (not by row) so each card's full history stays intact
        keep_cards = df["cc_num"].drop_duplicates().sample(frac=sample_frac, random_state=42)
        df = df[df["cc_num"].isin(keep_cards)]
    return df


def engineer_features(df):
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)

    df["distance_from_home"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

    out_frames = []
    for cc_num, g in df.groupby("cc_num", sort=False):
        g = g.sort_values("trans_date_trans_time").reset_index(drop=True)

        # --- amount z-score vs THIS CARD's prior rolling mean/std ---
        prior_amt = g["amt"].shift(1)
        roll_mean = prior_amt.expanding(min_periods=5).mean()
        roll_std = prior_amt.expanding(min_periods=5).std().replace(0, np.nan)
        g["amount_zscore"] = ((g["amt"] - roll_mean) / roll_std).fillna(0)

        # --- category-relative amount z-score ---
        g["category_amt_zscore"] = 0.0
        for cat, cg in g.groupby("category"):
            idx = cg.index
            prior = g.loc[idx, "amt"].shift(1)
            m = prior.expanding(min_periods=3).mean()
            s = prior.expanding(min_periods=3).std().replace(0, np.nan)
            g.loc[idx, "category_amt_zscore"] = ((g.loc[idx, "amt"] - m) / s).fillna(0)

        # --- velocity: transactions by this card in trailing 1 hour ---
        g_ts = g.set_index("trans_date_trans_time")
        txn_count = g_ts["trans_num"].rolling("1h").count()
        g["txn_count_1hr"] = (txn_count - 1).values

        # --- merchant novelty: has this card paid this merchant before ---
        seen = set()
        is_new = []
        for m in g["merchant"]:
            is_new.append(0 if m in seen else 1)
            seen.add(m)
        g["is_new_merchant"] = is_new

        out_frames.append(g)

    result = pd.concat(out_frames, ignore_index=True)
    result = result.sort_values("trans_date_trans_time").reset_index(drop=True)

    # --- time-of-day / day-of-week (fraud often clusters at odd hours) ---
    # cyclical encoding (sin/cos) instead of raw hour int -- so the model
    # understands hour 23 and hour 0 are adjacent, not 23 apart
    result["hour"] = result["trans_date_trans_time"].dt.hour
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["is_weekend"] = (result["trans_date_trans_time"].dt.dayofweek >= 5).astype(int)

    # --- category frequency-encoded (how common is this category overall) ---
    # frequency encoding is a simple, leakage-safe way to use a high-cardinality
    # categorical without one-hot blowing up the feature space
    cat_freq = result["category"].value_counts(normalize=True)
    result["category_freq"] = result["category"].map(cat_freq)

    return result


FEATURE_COLS = ["amount_zscore", "category_amt_zscore", "txn_count_1hr",
                 "is_new_merchant", "distance_from_home",
                 "hour_sin", "hour_cos", "is_weekend", "category_freq"]

KEEP_COLS = ["trans_num", "cc_num", "merchant", "category", "amt",
             "trans_date_trans_time", "is_fraud"] + FEATURE_COLS

# de-dupe while preserving order (hour/category_freq aren't in original per-card loop)
KEEP_COLS = list(dict.fromkeys(KEEP_COLS))


if __name__ == "__main__":
    print("Loading train set...")
    train_raw = load_raw(TRAIN_PATH)
    print(f"  {len(train_raw)} rows, {train_raw['is_fraud'].sum()} fraud")

    print("Loading test set...")
    test_raw = load_raw(TEST_PATH)
    print(f"  {len(test_raw)} rows, {test_raw['is_fraud'].sum()} fraud")

    print("Engineering features (train)...")
    train_feat = engineer_features(train_raw)
    print("Engineering features (test)...")
    test_feat = engineer_features(test_raw)

    train_feat[KEEP_COLS].to_csv("data/features_train.csv", index=False)
    test_feat[KEEP_COLS].to_csv("data/features_test.csv", index=False)

    print("\nSaved data/features_train.csv and data/features_test.csv")
    print("\nFeature means, fraud vs legit (train):")
    print(train_feat.groupby("is_fraud")[FEATURE_COLS].mean())