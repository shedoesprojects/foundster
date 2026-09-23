import pandas as pd
import numpy as np

INPUT_PATH = "data/transactions.csv"
OUTPUT_PATH = "data/features.csv"


def engineer_features(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["merchant_id", "timestamp"]).reset_index(drop=True)

    feature_frames = []

    for merchant_id, g in df.groupby("merchant_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)

        # --- amount z-score vs merchant's PRIOR rolling mean/std ---
        # shift(1) ensures we only use transactions strictly before the current one
        prior_amount = g["amount"].shift(1)
        rolling_mean = prior_amount.expanding(min_periods=5).mean()
        rolling_std = prior_amount.expanding(min_periods=5).std().replace(0, np.nan)
        g["amount_zscore"] = (g["amount"] - rolling_mean) / rolling_std
        g["amount_zscore"] = g["amount_zscore"].fillna(0)

        # --- velocity features using a time-indexed rolling window ---
        g_ts = g.set_index("timestamp")
        # count of prior transactions in trailing 5 min (exclude current row)
        txn_count = g_ts["txn_id"].rolling("5min").count()
        g["txn_count_5min"] = (txn_count - 1).values  # -1 to exclude the current txn itself

        # unique devices in trailing 5 min window
        def rolling_unique_devices(sub):
            return sub.rolling("5min").apply(lambda x: pd.Series(x).nunique(), raw=False)

        # rolling.apply on non-numeric needs a workaround: encode device_id as category codes
        device_codes = pd.Series(g["device_id"].astype("category").cat.codes.values, index=g_ts.index)
        unique_devices = device_codes.rolling("5min").apply(lambda x: len(set(x)), raw=True)
        g["unique_devices_5min"] = unique_devices.values

        # --- device novelty: has this device ever transacted with this merchant before ---
        seen_devices = set()
        is_new = []
        for d in g["device_id"]:
            is_new.append(0 if d in seen_devices else 1)
            seen_devices.add(d)
        g["is_new_device"] = is_new

        # --- geo distance from merchant's own "home" (its most common location) ---
        home_lat = g["geo_lat"].median()
        home_lon = g["geo_lon"].median()
        g["geo_dist_from_home"] = np.sqrt((g["geo_lat"] - home_lat) ** 2 + (g["geo_lon"] - home_lon) ** 2)

        feature_frames.append(g)

    result = pd.concat(feature_frames, ignore_index=True)
    result = result.sort_values("timestamp").reset_index(drop=True)
    return result


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} raw transactions")

    featured = engineer_features(df)

    feature_cols = [
        "txn_id", "merchant_id", "timestamp", "amount", "is_fraud", "fraud_type",
        "amount_zscore", "txn_count_5min", "unique_devices_5min",
        "is_new_device", "geo_dist_from_home",
    ]
    featured[feature_cols].to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(featured)} rows with features to {OUTPUT_PATH}")
    print("\nFeature summary for fraud vs legit (mean values):")
    print(featured.groupby("is_fraud")[
        ["amount_zscore", "txn_count_5min", "unique_devices_5min", "is_new_device", "geo_dist_from_home"]
    ].mean())