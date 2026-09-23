"""
Feature engineering for the REAL Kaggle Sparkov credit-card fraud dataset.

V2 DESIGN
---------
V1 asks:
    "Is this transaction unusual for this merchant?"

V2 asks:
    "Is this transaction unusual for this cardholder?"

That distinction matters because a compromised card can be used
across many different merchants.

All behavioral features are built from information available before
the current transaction.

IMPORTANT:
    We never allow the current transaction to influence its own
    historical baseline.

FEATURE GROUPS
--------------

1. Cardholder behavioral features
   - amount_zscore
   - txn_count_1hr
   - is_new_merchant
   - seconds_since_last_txn

2. Category-relative behavior
   - category_amt_zscore
   - category_freq

3. Geographic behavior
   - distance_from_home

4. Temporal behavior
   - hour_sin
   - hour_cos
   - is_weekend

5. Demographic/context features
   - age
   - city_pop_log
   - gender_female
   - job_freq

6. Merchant context
   - merchant_risk

merchant_risk is NOT calculated from the transaction's own label.
It is calculated from a historical training period and later applied
to validation/test data.
"""

import pandas as pd
import numpy as np


TRAIN_PATH = "data/fraudTrain.csv"
TEST_PATH = "data/fraudTest.csv"


# ============================================================
# 1. GEOGRAPHIC DISTANCE
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two latitude/longitude
    coordinates.

    Returns:
        Distance in kilometers.
    """

    R = 6371.0

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(a))


# ============================================================
# 2. LOAD RAW DATA
# ============================================================

def load_raw(path, sample_frac=None):
    """
    Load Sparkov transaction data.

    If sampling is requested, sample CARDHOLDERS rather than rows.
    This preserves each selected cardholder's transaction history.
    """

    df = pd.read_csv(path)

    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"]
    )

    df["dob"] = pd.to_datetime(df["dob"])

    if sample_frac is not None:
        keep_cards = (
            df["cc_num"]
            .drop_duplicates()
            .sample(
                frac=sample_frac,
                random_state=42
            )
        )

        df = df[df["cc_num"].isin(keep_cards)]

    return df


# ============================================================
# 3. CARDHOLDER-LEVEL FEATURES
# ============================================================

def engineer_features(df):
    """
    Build transaction-level features.

    Every behavioral feature is based on information that occurred
    before the current transaction.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Sort chronologically within each card
    # --------------------------------------------------------

    df = (
        df.sort_values(
            ["cc_num", "trans_date_trans_time"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Geographic distance
    # --------------------------------------------------------

    df["distance_from_home"] = haversine_km(
        df["lat"],
        df["long"],
        df["merch_lat"],
        df["merch_long"]
    )

    output_frames = []

    # --------------------------------------------------------
    # Process each cardholder independently
    # --------------------------------------------------------

    for cc_num, g in df.groupby(
        "cc_num",
        sort=False
    ):

        g = (
            g.sort_values("trans_date_trans_time")
            .reset_index(drop=True)
        )

        # ====================================================
        # A. AMOUNT Z-SCORE
        # ====================================================

        # IMPORTANT:
        # shift(1) means the current transaction is NOT included
        # in its own historical baseline.

        prior_amount = g["amt"].shift(1)

        rolling_mean = (
            prior_amount
            .expanding(min_periods=5)
            .mean()
        )

        rolling_std = (
            prior_amount
            .expanding(min_periods=5)
            .std()
            .replace(0, np.nan)
        )

        g["amount_zscore"] = (
            (g["amt"] - rolling_mean)
            / rolling_std
        ).fillna(0)

        # ====================================================
        # B. CATEGORY-RELATIVE AMOUNT
        # ====================================================

        g["category_amt_zscore"] = 0.0

        for category, category_group in g.groupby(
            "category"
        ):

            idx = category_group.index

            prior_category_amount = (
                g.loc[idx, "amt"]
                .shift(1)
            )

            category_mean = (
                prior_category_amount
                .expanding(min_periods=3)
                .mean()
            )

            category_std = (
                prior_category_amount
                .expanding(min_periods=3)
                .std()
                .replace(0, np.nan)
            )

            g.loc[idx, "category_amt_zscore"] = (
                (
                    g.loc[idx, "amt"]
                    - category_mean
                )
                / category_std
            ).fillna(0)

        # ====================================================
        # C. TRANSACTION VELOCITY
        # ====================================================

        temp = g.set_index(
            "trans_date_trans_time"
        )

        transaction_count = (
            temp["trans_num"]
            .rolling("1h")
            .count()
        )

        # Rolling window includes current transaction.
        # Subtract one so this represents OTHER transactions
        # in the preceding hour.

        g["txn_count_1hr"] = (
            transaction_count - 1
        ).values

        # ====================================================
        # D. NEW MERCHANT
        # ====================================================

        seen_merchants = set()
        new_merchant_flags = []

        for merchant in g["merchant"]:

            if merchant in seen_merchants:
                new_merchant_flags.append(0)
            else:
                new_merchant_flags.append(1)

            seen_merchants.add(merchant)

        g["is_new_merchant"] = new_merchant_flags

        # ====================================================
        # E. TIME SINCE PREVIOUS TRANSACTION
        # ====================================================

        previous_time = (
            g["trans_date_trans_time"]
            .shift(1)
        )

        gap_seconds = (
            g["trans_date_trans_time"]
            - previous_time
        ).dt.total_seconds()

        # First transaction gets a large value rather than zero.
        g["seconds_since_last_txn"] = (
            gap_seconds
            .fillna(30 * 24 * 3600)
        )

        output_frames.append(g)

    result = pd.concat(
        output_frames,
        ignore_index=True
    )

    result = (
        result
        .sort_values("trans_date_trans_time")
        .reset_index(drop=True)
    )

    # ========================================================
    # 4. TIME FEATURES
    # ========================================================

    result["hour"] = (
        result["trans_date_trans_time"]
        .dt.hour
    )

    # Cyclic encoding:
    #
    # 23:00 and 00:00 are close in reality.
    # Raw integers would incorrectly represent them as far apart.

    result["hour_sin"] = np.sin(
        2 * np.pi * result["hour"] / 24
    )

    result["hour_cos"] = np.cos(
        2 * np.pi * result["hour"] / 24
    )

    result["is_weekend"] = (
        result["trans_date_trans_time"]
        .dt.dayofweek >= 5
    ).astype(int)

    # ========================================================
    # 5. CATEGORY FREQUENCY
    # ========================================================

    category_frequency = (
        result["category"]
        .value_counts(normalize=True)
    )

    result["category_freq"] = (
        result["category"]
        .map(category_frequency)
    )

    # ========================================================
    # 6. DEMOGRAPHIC / CONTEXT FEATURES
    # ========================================================

    result["age"] = (
        (
            result["trans_date_trans_time"]
            - result["dob"]
        ).dt.days
        / 365.25
    )

    result["city_pop_log"] = np.log1p(
        result["city_pop"]
    )

    result["gender_female"] = (
        result["gender"] == "F"
    ).astype(int)

    # ========================================================
    # 7. JOB FREQUENCY
    # ========================================================

    job_frequency = (
        result["job"]
        .value_counts(normalize=True)
    )

    result["job_freq"] = (
        result["job"]
        .map(job_frequency)
    )

    return result


# ============================================================
# 8. MERCHANT RISK
# ============================================================

def add_merchant_risk(
    historical_df,
    apply_to_df
):
    """
    Add historical merchant fraud rate.

    IMPORTANT:
        historical_df must contain ONLY information available
        before the transactions in apply_to_df.

    This prevents target leakage.

    Unknown merchants receive the historical global fraud rate.
    """

    historical_df = historical_df.copy()
    apply_to_df = apply_to_df.copy()

    merchant_rate = (
        historical_df
        .groupby("merchant")["is_fraud"]
        .mean()
    )

    global_rate = (
        historical_df["is_fraud"]
        .mean()
    )

    apply_to_df["merchant_risk"] = (
        apply_to_df["merchant"]
        .map(merchant_rate)
        .fillna(global_rate)
    )

    return apply_to_df


# ============================================================
# 9. FINAL FEATURE LIST
# ============================================================

FEATURE_COLS = [

    # Cardholder behavior
    "amount_zscore",
    "category_amt_zscore",
    "txn_count_1hr",
    "is_new_merchant",
    "seconds_since_last_txn",

    # Geography
    "distance_from_home",

    # Time
    "hour_sin",
    "hour_cos",
    "is_weekend",

    # Category
    "category_freq",

    # Demographics / context
    "age",
    "city_pop_log",
    "gender_female",
    "job_freq",

    # Merchant historical risk
    "merchant_risk",
]


KEEP_COLS = [
    "trans_num",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "trans_date_trans_time",
    "is_fraud",
] + FEATURE_COLS

# Remove accidental duplicates while preserving order.
KEEP_COLS = list(
    dict.fromkeys(KEEP_COLS)
)


# ============================================================
# 10. MAIN
# ============================================================

if __name__ == "__main__":

    print("Loading train set...")

    train_raw = load_raw(TRAIN_PATH)

    print(
        f"  {len(train_raw):,} rows, "
        f"{train_raw['is_fraud'].sum():,} fraud"
    )

    print("\nLoading test set...")

    test_raw = load_raw(TEST_PATH)

    print(
        f"  {len(test_raw):,} rows, "
        f"{test_raw['is_fraud'].sum():,} fraud"
    )

    print("\nEngineering features for train...")

    train_feat = engineer_features(
        train_raw
    )

    print("Engineering features for test...")

    test_feat = engineer_features(
        test_raw
    )

    # --------------------------------------------------------
    # Merchant risk:
    #
    # Train merchant risk comes from TRAIN.
    # Test merchant risk is also calculated from TRAIN only.
    # --------------------------------------------------------

    train_feat = add_merchant_risk(
        train_feat,
        train_feat
    )

    test_feat = add_merchant_risk(
        train_feat,
        test_feat
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train_feat[KEEP_COLS].to_csv(
        "data/features_train.csv",
        index=False
    )

    test_feat[KEEP_COLS].to_csv(
        "data/features_test.csv",
        index=False
    )

    print(
        "\nSaved:"
        "\n  data/features_train.csv"
        "\n  data/features_test.csv"
    )

    print(
        "\nNumber of features:",
        len(FEATURE_COLS)
    )

    print("\nFeatures:")

    for i, feature in enumerate(
        FEATURE_COLS,
        start=1
    ):
        print(
            f"  {i:2d}. {feature}"
        )

    print(
        "\nFeature means, fraud vs legitimate:"
    )

    print(
        train_feat
        .groupby("is_fraud")[FEATURE_COLS]
        .mean()
    )