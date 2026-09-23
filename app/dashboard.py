"""
Streamlit demo dashboard for the fraud detector.

ONE-TAB DEMO

The user chooses:

    V1 - Synthetic dataset
    V2 - Real Sparkov dataset

Then the dashboard:

    1. selects a random transaction
    2. displays transaction details
    3. scores it with the corresponding XGBoost model
    4. displays the prediction
    5. displays human-readable risk signals

This dashboard is intentionally minimal.

It is a demo layer over the existing ML pipeline,
not a replacement for the training/evaluation code.
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb


# ============================================================
# PATHS
# ============================================================

DASHBOARD_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    DASHBOARD_DIR
)

V1_DIR = os.path.join(
    PROJECT_ROOT,
    "fraud-detector-v1"
)

V2_DIR = os.path.join(
    PROJECT_ROOT,
    "fraud-detector-v2"
)

V1_DATA_DIR = os.path.join(
    V1_DIR,
    "data"
)

V1_SRC_DIR = os.path.join(
    V1_DIR,
    "src"
)

V2_DATA_DIR = os.path.join(
    V2_DIR,
    "data"
)

V2_SRC_DIR = os.path.join(
    V2_DIR,
    "data",
    "src"
)


# Make dashboard directory importable.
sys.path.insert(
    0,
    DASHBOARD_DIR
)


from explain import explain_transaction


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fraud Detector",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🔍 Fraud Detector")

st.caption(
    "Compare a synthetic fraud benchmark with "
    "real-world cardholder-relative fraud detection."
)


# ============================================================
# V1 DATA
# ============================================================

@st.cache_data
def load_v1_data():

    path = os.path.join(
        V1_DATA_DIR,
        "features.csv"
    )

    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    return df


# ============================================================
# V2 DATA
# ============================================================

@st.cache_data
def load_v2_data():

    path = os.path.join(
        V2_DATA_DIR,
        "features_test.csv"
    )

    df = pd.read_csv(path)

    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"]
    )

    return df


# ============================================================
# V1 MODEL
# ============================================================

@st.cache_resource
def train_v1_model():

    df = load_v1_data()

    features = [
        "amount_zscore",
        "txn_count_5min",
        "unique_devices_5min",
        "is_new_device",
        "geo_dist_from_home",
    ]

    cutoff = df["timestamp"].quantile(
        0.71
    )

    train = df[
        df["timestamp"] <= cutoff
    ]

    X_train = train[features]
    y_train = train["is_fraud"]

    ratio = (
        (y_train == 0).sum()
        /
        max(
            (y_train == 1).sum(),
            1
        )
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=ratio,
        eval_metric="aucpr",
        random_state=42,
    )

    model.fit(
        X_train,
        y_train
    )

    return model, features


# ============================================================
# V2 MODEL
# ============================================================

@st.cache_resource
def train_v2_model():

    df = load_v2_data()

    features = [
        "amount_zscore",
        "category_amt_zscore",
        "txn_count_1hr",
        "is_new_merchant",
        "distance_from_home",
        "hour_sin",
        "hour_cos",
        "is_weekend",
        "category_freq",
        "seconds_since_last_txn",
        "age",
        "city_pop_log",
        "gender_female",
        "job_freq",
        "merchant_risk",
    ]

    # --------------------------------------------------------
    # Demo model
    #
    # The dashboard trains on the available feature dataset.
    # The formal train/validation/test evaluation remains in
    # models.py.
    # --------------------------------------------------------

    train_path = os.path.join(
        V2_DATA_DIR,
        "features_train.csv"
    )

    train = pd.read_csv(
        train_path
    )

    X_train = train[features]
    y_train = train["is_fraud"]

    ratio = (
        (y_train == 0).sum()
        /
        max(
            (y_train == 1).sum(),
            1
        )
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=ratio,
        eval_metric="aucpr",
        random_state=42,
    )

    model.fit(
        X_train,
        y_train
    )

    return model, features


# ============================================================
# LOAD DATA
# ============================================================

version = st.radio(
    "Choose fraud detection track",
    [
        "V1 — Synthetic",
        "V2 — Real"
    ],
    horizontal=True
)


# ============================================================
# V1
# ============================================================

if version.startswith("V1"):

    try:

        data = load_v1_data()

        model, feature_cols = (
            train_v1_model()
        )

    except Exception as e:

        st.error(
            "Could not load V1 data/model."
        )

        st.exception(e)

        st.stop()

    version_key = "v1"

    timestamp_col = "timestamp"

    id_col = "txn_id"


# ============================================================
# V2
# ============================================================

else:

    try:

        data = load_v2_data()

        model, feature_cols = (
            train_v2_model()
        )

    except Exception as e:

        st.error(
            "Could not load V2 data/model."
        )

        st.exception(e)

        st.stop()

    version_key = "v2"

    timestamp_col = (
        "trans_date_trans_time"
    )

    id_col = "trans_num"


# ============================================================
# RANDOM TRANSACTION
# ============================================================

st.divider()

st.subheader(
    "🎲 Select a transaction"
)

if (
    "current_version" not in
    st.session_state
    or
    st.session_state[
        "current_version"
    ] != version_key
):

    st.session_state[
        "current_version"
    ] = version_key

    st.session_state[
        "current_row"
    ] = None


if st.button(
    "🎲 Choose Random Transaction",
    type="primary"
):

    st.session_state[
        "current_row"
    ] = data.sample(
        1,
        random_state=None
    ).iloc[0]


if (
    st.session_state.get(
        "current_row"
    ) is None
):

    st.info(
        "Click the button above to select "
        "a random transaction."
    )

    st.stop()


row = st.session_state[
    "current_row"
]


# ============================================================
# TRANSACTION INFORMATION
# ============================================================

st.divider()

st.subheader(
    "💳 Transaction"
)


col1, col2, col3 = st.columns(3)


with col1:

    if version_key == "v1":

        st.metric(
            "Amount",
            f"{row.get('amount', 0):,.2f}"
        )

    else:

        st.metric(
            "Amount",
            f"{row.get('amt', 0):,.2f}"
        )


with col2:

    actual = row.get(
        "is_fraud",
        None
    )

    if actual == 1:

        st.metric(
            "Actual label",
            "FRAUD"
        )

    elif actual == 0:

        st.metric(
            "Actual label",
            "LEGITIMATE"
        )

    else:

        st.metric(
            "Actual label",
            "Unknown"
        )


with col3:

    st.metric(
        "Dataset",
        version_key.upper()
    )


# ============================================================
# TRANSACTION DETAILS
# ============================================================

with st.expander(
    "View transaction details"
):

    details = {}

    if id_col in row.index:

        details["Transaction ID"] = (
            row[id_col]
        )

    if timestamp_col in row.index:

        details["Timestamp"] = (
            str(row[timestamp_col])
        )

    if version_key == "v1":

        details["Merchant"] = (
            row.get(
                "merchant_id",
                "N/A"
            )
        )

        details["Amount z-score"] = (
            f"{row.get('amount_zscore', 0):.3f}"
        )

        details["Transactions / 5 min"] = (
            int(
                row.get(
                    "txn_count_5min",
                    0
                )
            )
        )

        details["Unique devices / 5 min"] = (
            int(
                row.get(
                    "unique_devices_5min",
                    0
                )
            )
        )

        details["New device"] = (
            "Yes"
            if row.get(
                "is_new_device",
                0
            ) == 1
            else "No"
        )

        details["Geographic distance"] = (
            f"{row.get('geo_dist_from_home', 0):.3f}"
        )

    else:

        details["Merchant"] = (
            row.get(
                "merchant",
                "N/A"
            )
        )

        details["Category"] = (
            row.get(
                "category",
                "N/A"
            )
        )

        details["Amount z-score"] = (
            f"{row.get('amount_zscore', 0):.3f}"
        )

        details["Category amount z-score"] = (
            f"{row.get('category_amt_zscore', 0):.3f}"
        )

        details["Transactions / hour"] = (
            int(
                row.get(
                    "txn_count_1hr",
                    0
                )
            )
        )

        details["New merchant"] = (
            "Yes"
            if row.get(
                "is_new_merchant",
                0
            ) == 1
            else "No"
        )

        details["Distance from home"] = (
            f"{row.get('distance_from_home', 0):.2f} km"
        )

        details["Age"] = (
            f"{row.get('age', 0):.1f}"
        )

        details["Merchant historical risk"] = (
            f"{row.get('merchant_risk', 0):.2%}"
        )

    st.json(details)


# ============================================================
# MODEL SCORING
# ============================================================

st.divider()

st.subheader(
    "🤖 Model decision"
)


X = pd.DataFrame(
    [row]
)[feature_cols]


probability = float(
    model.predict_proba(X)[0, 1]
)


prediction = int(
    probability >= 0.5
)


# ============================================================
# RESULT
# ============================================================

result_col, confidence_col = (
    st.columns([1, 1])
)


with result_col:

    if prediction == 1:

        st.error(
            "🚩 FRAUD FLAGGED"
        )

    else:

        st.success(
            "✅ NOT FLAGGED"
        )


with confidence_col:

    if prediction == 1:

        st.metric(
            "XGBoost score",
            f"{probability:.1%}"
        )

    else:

        st.metric(
            "XGBoost score",
            f"{probability:.1%}"
        )


st.caption(
    "The displayed score is the model's XGBoost "
    "positive-class score, not a guaranteed real-world "
    "probability of fraud."
)


# ============================================================
# EXPLANATION
# ============================================================

st.divider()

st.subheader(
    "💡 Why did the model see risk?"
)

explanation = explain_transaction(
    row,
    version_key
)


st.write(
    f"### {explanation['summary']}"
)


for reason in explanation["reasons"]:

    severity = reason["severity"]

    if severity == "high":

        icon = "🔴"

    elif severity == "medium":

        icon = "🟡"

    elif severity == "low":

        icon = "🟢"

    else:

        icon = "⚪"

    st.write(
        f"{icon} **{reason['text']}**"
    )


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

actual = row.get(
    "is_fraud",
    None
)

if actual is not None:

    st.divider()

    st.subheader(
        "🔎 Ground truth"
    )

    if prediction == 1 and actual == 1:

        st.success(
            "Model flagged it and the dataset label is FRAUD."
        )

    elif prediction == 1 and actual == 0:

        st.warning(
            "Model flagged it, but the dataset label is "
            "LEGITIMATE — this is a false positive."
        )

    elif prediction == 0 and actual == 1:

        st.warning(
            "Model did not flag it, but the dataset label "
            "is FRAUD — this is a false negative."
        )

    else:

        st.info(
            "Model did not flag it and the dataset label "
            "is LEGITIMATE."
        )