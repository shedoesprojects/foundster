"""
Explainability Engine - Credit Card Fraud Detection
====================================================
Transforms raw feature signals and model probabilities into human-auditable,
plain-English risk explanations for risk analysts, bank ops, and presentation demos.
"""

def explain_transaction(row, threshold=0.35, prob=None):
    """
    Evaluates risk signals for a single transaction.
    Returns:
        dict with keys:
            - is_flagged: bool
            - confidence: float (percentage)
            - risk_level: str ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
            - summary: str
            - signals: list of dicts [{'signal': name, 'severity': 'high'|'medium'|'low', 'text': description}]
    """
    if prob is None:
        prob = row.get("xgb_prob", 0.0)

    is_flagged = prob >= threshold
    reasons = []

    # 1. Amount to Card Average Ratio (Strongest single indicator)
    amt_ratio = row.get("amt_to_card_avg", 1.0)
    amt = row.get("amt", 0.0)
    if amt_ratio >= 4.0:
        reasons.append({
            "signal": "amt_to_card_avg",
            "severity": "high",
            "text": f"Amount (Rs {amt:,.2f}) is {amt_ratio:.1f}x higher than cardholder's usual average"
        })
    elif amt_ratio >= 2.5:
        reasons.append({
            "signal": "amt_to_card_avg",
            "severity": "medium",
            "text": f"Amount (Rs {amt:,.2f}) is {amt_ratio:.1f}x higher than normal card spend"
        })

    # 2. Amount Z-score
    zscore = row.get("amount_zscore", 0.0)
    if zscore >= 3.0:
        reasons.append({
            "signal": "amount_zscore",
            "severity": "high",
            "text": f"Spending magnitude is {zscore:.1f} standard deviations above card historical baseline"
        })

    # 3. High-Risk Merchant Category
    category = row.get("category", "")
    high_risk_cats = ["shopping_net", "misc_net", "grocery_pos"]
    if category in high_risk_cats or any(row.get(f"cat_{c}", 0) == 1 for c in high_risk_cats):
        cat_name = category if category else "online retail / grocery POS"
        reasons.append({
            "signal": "category",
            "severity": "medium",
            "text": f"Target merchant is in high-vulnerability category: '{cat_name}'"
        })

    # 4. Geodesic Distance
    dist = row.get("distance_km", 0.0)
    if dist >= 150.0:
        reasons.append({
            "signal": "distance_km",
            "severity": "high",
            "text": f"Terminal location is {dist:.1f} km away from cardholder registered home location"
        })
    elif dist >= 80.0:
        reasons.append({
            "signal": "distance_km",
            "severity": "medium",
            "text": f"Terminal location is {dist:.1f} km away from usual residential zone"
        })

    # 5. Circadian / Odd Hours
    hour = int(row.get("hour", 12))
    if hour in [23, 0, 1, 2, 3, 4, 5]:
        reasons.append({
            "signal": "hour",
            "severity": "medium",
            "text": f"Transaction initiated during high-fraud window ({hour:02d}:00 hours)"
        })

    # 6. Rapid Velocity (Time since last transaction)
    log_time = row.get("log_time_since_last_txn", 10.0)
    if log_time < 5.5:  # under 4 minutes
        reasons.append({
            "signal": "velocity",
            "severity": "high",
            "text": "Rapid consecutive transaction detected shortly after prior payment"
        })

    # 7. Fallback if no single rule triggered but combination did
    if not reasons:
        if is_flagged:
            reasons.append({
                "signal": "multivariate_pattern",
                "severity": "medium",
                "text": "Subtle multi-variable anomaly flagged across combined demographic and spending dimensions"
            })
        else:
            reasons.append({
                "signal": "normal_behavior",
                "severity": "low",
                "text": "Transaction aligns with cardholder historical patterns, location, and normal merchant parameters"
            })

    high_count = sum(1 for r in reasons if r["severity"] == "high")
    if prob >= 0.70:
        risk_level = "CRITICAL"
        summary = "High-Risk Fraud Pattern Identified"
    elif prob >= threshold:
        risk_level = "HIGH"
        summary = "Suspicious Transaction Flagged for Review"
    elif prob >= 0.15:
        risk_level = "MEDIUM"
        summary = "Moderate Risk: Standard Automated Clearance"
    else:
        risk_level = "LOW"
        summary = "Legitimate Transaction Verified"

    return {
        "is_flagged": is_flagged,
        "confidence": prob,
        "risk_level": risk_level,
        "summary": summary,
        "signals": reasons,
    }
