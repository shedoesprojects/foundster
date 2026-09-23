"""
Human-readable explainability layer for the fraud detector.

This module supports both:

    V1 -> synthetic / merchant-relative features
    V2 -> real / cardholder-relative features

IMPORTANT
---------
These are SIGNAL explanations, not mathematical XGBoost attributions.

For example:

    "Amount is unusually high"

means the engineered amount anomaly feature crossed a
human-readable threshold.

It does NOT mean:

    "XGBoost assigned exactly 37% of its prediction to amount."

This distinction keeps the demo honest.
"""

import numpy as np


# ============================================================
# V1
# ============================================================

def explain_v1_transaction(row):
    """
    Explain a V1 transaction using the same interpretable
    signals used by the V1 rule-based baseline.
    """

    reasons = []

    # --------------------------------------------------------
    # Rule 1: velocity
    # --------------------------------------------------------

    txn_count = row.get(
        "txn_count_5min",
        0
    )

    if txn_count > 8:

        reasons.append({
            "signal": "velocity",
            "severity": "high",
            "text": (
                f"High transaction velocity: "
                f"{int(txn_count)} other transactions "
                f"occurred within the previous 5 minutes."
            ),
        })

    # --------------------------------------------------------
    # Rule 2: amount anomaly
    # --------------------------------------------------------

    amount_z = row.get(
        "amount_zscore",
        0
    )

    if abs(amount_z) > 5:

        direction = (
            "above"
            if amount_z > 0
            else "below"
        )

        reasons.append({
            "signal": "amount_anomaly",
            "severity": "high",
            "text": (
                f"Transaction amount is "
                f"{abs(amount_z):.1f}σ {direction} "
                f"the merchant's historical amount pattern."
            ),
        })

    # --------------------------------------------------------
    # Rule 3: new device + geographic anomaly
    # --------------------------------------------------------

    new_device = row.get(
        "is_new_device",
        0
    )

    geo_distance = row.get(
        "geo_dist_from_home",
        0
    )

    if (
        new_device == 1
        and geo_distance > 0.5
    ):

        reasons.append({
            "signal": "new_device_geo",
            "severity": "medium",
            "text": (
                "Transaction came from a new device "
                "and a location substantially different "
                "from the merchant's usual location."
            ),
        })

    # --------------------------------------------------------
    # No individual rule triggered
    # --------------------------------------------------------

    if not reasons:

        reasons.append({
            "signal": "combined_pattern",
            "severity": "none",
            "text": (
                "No individual rule crossed its threshold. "
                "The model result may come from a combination "
                "of weaker signals."
            ),
        })

    return _build_result(
        reasons,
        model_name="V1"
    )


# ============================================================
# V2
# ============================================================

def reconstruct_hour(
    hour_sin,
    hour_cos
):
    """
    Convert cyclical sin/cos representation back into
    an approximate hour.
    """

    angle = np.arctan2(
        hour_sin,
        hour_cos
    )

    hour = (
        angle
        / (2 * np.pi)
        * 24
    )

    return int(round(hour)) % 24


def explain_v2_transaction(row):
    """
    Explain a V2 transaction using interpretable risk signals.
    """

    reasons = []

    # --------------------------------------------------------
    # Amount anomaly
    # --------------------------------------------------------

    amount_z = row.get(
        "amount_zscore",
        0
    )

    if amount_z > 3:

        reasons.append({
            "signal": "amount_zscore",
            "severity": "high",
            "text": (
                f"Amount is {amount_z:.1f}σ above "
                f"this cardholder's historical spending pattern."
            ),
        })

    # --------------------------------------------------------
    # Category-relative amount
    # --------------------------------------------------------

    category_z = row.get(
        "category_amt_zscore",
        0
    )

    if category_z > 5:

        reasons.append({
            "signal": "category_amt_zscore",
            "severity": "high",
            "text": (
                f"Amount is unusually high for this "
                f"spending category ({category_z:.1f}σ above normal)."
            ),
        })

    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    txn_count = row.get(
        "txn_count_1hr",
        0
    )

    if txn_count > 3:

        reasons.append({
            "signal": "txn_count_1hr",
            "severity": "medium",
            "text": (
                f"{int(txn_count)} other transactions "
                f"occurred on this card in the previous hour."
            ),
        })

    # --------------------------------------------------------
    # New merchant
    # --------------------------------------------------------

    if row.get(
        "is_new_merchant",
        0
    ) == 1:

        reasons.append({
            "signal": "is_new_merchant",
            "severity": "low",
            "text": (
                "This is the first observed transaction "
                "between this card and this merchant."
            ),
        })

    # --------------------------------------------------------
    # Dormant -> sudden activity
    # --------------------------------------------------------

    gap = row.get(
        "seconds_since_last_txn",
        0
    )

    if gap >= 24 * 3600:

        reasons.append({
            "signal": "transaction_recency",
            "severity": "medium",
            "text": (
                "The card had been inactive for more than "
                "24 hours before this transaction."
            ),
        })

    # --------------------------------------------------------
    # Unusual hour
    # --------------------------------------------------------

    if (
        "hour_sin" in row
        and "hour_cos" in row
    ):

        hour = reconstruct_hour(
            row["hour_sin"],
            row["hour_cos"]
        )

        if hour >= 23 or hour <= 4:

            reasons.append({
                "signal": "hour",
                "severity": "medium",
                "text": (
                    f"Transaction occurred at approximately "
                    f"{hour:02d}:00, an unusual hour."
                ),
            })

    # --------------------------------------------------------
    # Merchant historical risk
    #
    # We display this as context rather than declaring a hard
    # fraud threshold because merchant_risk is a continuous
    # historical statistic.
    # --------------------------------------------------------

    merchant_risk = row.get(
        "merchant_risk",
        0
    )

    if merchant_risk >= 0.01:

        reasons.append({
            "signal": "merchant_risk",
            "severity": "medium",
            "text": (
                f"This merchant's historical fraud rate in "
                f"the training data was approximately "
                f"{merchant_risk:.2%}."
            ),
        })

    # --------------------------------------------------------
    # No individual signal
    # --------------------------------------------------------

    if not reasons:

        reasons.append({
            "signal": "combined_pattern",
            "severity": "none",
            "text": (
                "No individual explanation threshold was "
                "crossed. The model may be responding to "
                "a combination of weaker signals."
            ),
        })

    return _build_result(
        reasons,
        model_name="V2"
    )


# ============================================================
# COMMON RESULT FORMAT
# ============================================================

def _build_result(
    reasons,
    model_name
):

    signal_reasons = [
        r for r in reasons
        if r["signal"] != "combined_pattern"
    ]

    high_count = sum(
        1
        for r in signal_reasons
        if r["severity"] == "high"
    )

    if high_count >= 1:

        summary = "Strong risk signal"

    elif len(signal_reasons) >= 2:

        summary = "Multiple risk signals"

    elif len(signal_reasons) == 1:

        summary = "Single risk signal"

    else:

        summary = "Combined model pattern"

    return {
        "model": model_name,
        "summary": summary,
        "reasons": reasons,
        "num_signals": len(signal_reasons),
    }


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def explain_transaction(
    row,
    version
):
    """
    Main public explanation function.

    version:
        "v1"
        "v2"
    """

    version = version.lower()

    if version == "v1":

        return explain_v1_transaction(row)

    if version == "v2":

        return explain_v2_transaction(row)

    raise ValueError(
        "version must be either 'v1' or 'v2'"
    )