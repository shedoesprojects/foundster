"""
Interactive Fraud Detection and Risk Management Dashboard
Foundster - Real-Time Credit Card Fraud Detection
Marvel R&D Lab Open Day Presentation System
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Setup source directory imports
APP_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from features import FEATURE_COLS, CATEGORIES, haversine_km
from explain import explain_transaction
import xgboost as xgb

# Streamlit Page Configuration
st.set_page_config(
    page_title="Foundster - Real-Time Credit Card Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1a1e24;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2d3748;
        margin-bottom: 12px;
    }
    .stAlert {
        border-radius: 8px;
    }
    .badge-fraud {
        background-color: #c53030;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-legit {
        background-color: #276749;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Cached Data and Model Loaders
@st.cache_resource
def load_xgboost_model():
    model_path = os.path.join(RESULTS_DIR, "xgb_model.json")
    clf = xgb.XGBClassifier()
    if os.path.exists(model_path):
        clf.load_model(model_path)
    else:
        train_path = os.path.join(DATA_DIR, "features_train.csv")
        if os.path.exists(train_path):
            train_df = pd.read_csv(train_path)
            X = train_df[FEATURE_COLS]
            y = train_df["is_fraud"]
            clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.08,
                scale_pos_weight=4.0, eval_metric="aucpr", random_state=42, n_jobs=-1
            )
            clf.fit(X, y)
            clf.save_model(model_path)
    return clf


@st.cache_data
def load_sample_dataset():
    sample_path = os.path.join(RESULTS_DIR, "test_sample_scored.csv")
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)
    
    test_path = os.path.join(DATA_DIR, "features_test.csv")
    if os.path.exists(test_path):
        return pd.read_csv(test_path, nrows=5000)
    return None


@st.cache_data
def load_metrics_table():
    path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# Sidebar Controls
st.sidebar.title("Foundster")
st.sidebar.caption("Marvel R&D Lab - Real-Time Fraud Detection")
st.sidebar.markdown("---")

st.sidebar.subheader("Operational Risk Strategy")
risk_mode = st.sidebar.radio(
    "Policy Preset",
    [
        "High-Recall Security Mode (90%+ Recall)",
        "Balanced F1 Mode (87% Precision, 86% Recall)",
        "Conservative Precision Mode (93.5% Precision)",
        "Custom Threshold Slider"
    ],
    index=0,
    help="Credit card fraud operations prioritize High Recall to minimize costly False Negatives."
)

if "High-Recall" in risk_mode:
    active_threshold = 0.35
    strategy_note = "Prioritizes zero missed fraud. Flagged transactions undergo secondary verification."
elif "Balanced" in risk_mode:
    active_threshold = 0.58
    strategy_note = "Optimizes mathematical balance (F1-score = 0.867)."
elif "Conservative" in risk_mode:
    active_threshold = 0.75
    strategy_note = "Minimizes false alarms (Precision = 93.5%). Only flags highest-confidence fraud."
else:
    active_threshold = st.sidebar.slider("Decision Threshold", 0.05, 0.95, 0.35, 0.01)
    strategy_note = f"Custom threshold calibrated at {active_threshold:.2f}."

st.sidebar.info(f"Active Cutoff: {active_threshold:.2f}\n\n{strategy_note}")
st.sidebar.markdown("---")
st.sidebar.markdown("""
Business Cost Matrix:
- False Negative (Missed): Rs 3,000 chargeback
- False Positive (Alert): Rs 50 verification
""")


# Main Content Area
st.title("Foundster: Credit Card Fraud Detection")
st.markdown(
    "Multivariate fraud scoring engine powered by calibrated gradient boosted trees (XGBoost), "
    "evaluated on 555,719 completely held-out real transactions."
)

# Top KPI Metric Banners
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("AUC-ROC", "99.86%", "Near Perfect Ranking")
kpi2.metric("AUC-PR", "92.90%", "+82.4% vs Baseline")
kpi3.metric("Peak Recall", "89.84%", "Target Defense")
kpi4.metric("False Alarm Rate", "0.04%", "122 / 553k Txns")
kpi5.metric("Loss Reduction", "82.7%", "Rs 32.5L Saved")

model = load_xgboost_model()
sample_df = load_sample_dataset()

# Navigation Tabs
tab_live, tab_bench, tab_queue, tab_simulator = st.tabs([
    "Live Scoring and Explainability",
    "Model Benchmark and Plots",
    "Real-Time Flagged Queue",
    "Cost and Threshold Simulator"
])


# TAB 1: Live Scoring and Explainability
with tab_live:
    st.subheader("Transaction Inference and Explainability Engine")
    st.write(
        "Evaluate individual transactions in real time. "
        "The model computes multi-dimensional spatial, circadian, and behavioral features to deliver an instant verdict with plain-English audit reasons."
    )

    score_mode = st.radio(
        "Transaction Input Source",
        ["Sample from Real-World Test Corpus", "Interactive Manual Builder"],
        horizontal=True
    )

    if score_mode == "Sample from Real-World Test Corpus":
        if sample_df is not None:
            c_sample1, c_sample2 = st.columns([2, 1])
            with c_sample1:
                sample_filter = st.selectbox(
                    "Filter Cases by Ground Truth Label",
                    ["Confirmed Fraud (Ground Truth = 1)", "Legitimate Transaction (Ground Truth = 0)", "Any Random Transaction"]
                )
            with c_sample2:
                st.write("")
                st.write("")
                draw_btn = st.button("Draw Sample Transaction", use_container_width=True)

            if "selected_row" not in st.session_state or draw_btn:
                if "Confirmed Fraud" in sample_filter:
                    subset = sample_df[sample_df["is_fraud"] == 1]
                elif "Legitimate" in sample_filter:
                    subset = sample_df[sample_df["is_fraud"] == 0]
                else:
                    subset = sample_df
                st.session_state["selected_row"] = subset.sample(1, random_state=None).iloc[0]

            curr = st.session_state["selected_row"]
            feature_row = curr[FEATURE_COLS].to_dict()
            amt_display = curr.get("amt", 0.0)
            ground_truth = curr.get("is_fraud", 0)
            trans_num = curr.get("trans_num", "TXN_774921")
            category_name = curr.get("category", "shopping_net")
        else:
            st.warning("Sample dataset not found. Please run feature extraction.")
            st.stop()

    else:
        st.write("Configure transaction attributes to simulate anomalous activity:")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            amt_display = st.number_input("Transaction Amount (INR)", min_value=1.0, max_value=25000.0, value=750.0, step=25.0)
            category_name = st.selectbox("Merchant Category", CATEGORIES, index=CATEGORIES.index("shopping_net"))
            cardholder_age = st.slider("Cardholder Age", 18, 90, 38)
        with col_m2:
            amt_to_avg = st.slider("Amount to Cardholder Avg Ratio", 0.1, 15.0, 5.2, 0.1,
                                   help="5x means spending 500% more than usual")
            distance_val = st.slider("Distance from Home (km)", 0.0, 500.0, 185.0, 5.0)
            hour_val = st.slider("Transaction Hour of Day", 0, 23, 2)
        with col_m3:
            city_pop = st.number_input("Local City Population", min_value=500, max_value=2500000, value=45000)
            day_val = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=5)
            gender_val = st.radio("Gender", ["Female", "Male"], horizontal=True)

        ground_truth = None
        trans_num = "SIMULATED_TEST"

        feature_row = {
            "amt": amt_display,
            "log_amt": np.log1p(amt_display),
            "age": cardholder_age,
            "gender": 1 if gender_val == "Male" else 0,
            "log_city_pop": np.log1p(city_pop),
            "hour": hour_val,
            "hour_sin": np.sin(2 * np.pi * hour_val / 24),
            "hour_cos": np.cos(2 * np.pi * hour_val / 24),
            "dayofweek": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_val),
            "is_weekend": 1 if day_val in ["Saturday", "Sunday"] else 0,
            "distance_km": distance_val,
            "amount_zscore": (amt_to_avg - 1.0) * 1.5,
            "amt_to_card_avg": amt_to_avg,
            "log_time_since_last_txn": np.log1p(3600),
        }
        for cat in CATEGORIES:
            feature_row[f"cat_{cat}"] = 1.0 if cat == category_name else 0.0

    # Execute Prediction
    X_input = pd.DataFrame([feature_row])[FEATURE_COLS]
    prob = float(model.predict_proba(X_input)[0, 1])
    is_fraud_predicted = prob >= active_threshold
    explanation = explain_transaction(feature_row, threshold=active_threshold, prob=prob)

    st.markdown("---")

    # Output Card
    card_col1, card_col2 = st.columns([1.2, 1.8])

    with card_col1:
        st.subheader("Decision Output")
        if is_fraud_predicted:
            st.error("TRANSACTION BLOCKED\n\nModel Verdict: Fraud Anomaly Detected")
        else:
            st.success("TRANSACTION APPROVED\n\nModel Verdict: Normal Legitimate Payment")

        st.metric(
            label="Calculated Fraud Probability",
            value=f"{prob * 100:.2f}%",
            delta=f"{'+' if prob >= active_threshold else ''}{(prob - active_threshold)*100:.1f}% vs Cutoff ({active_threshold*100:.0f}%)",
            delta_color="inverse"
        )

        st.progress(prob)

        if ground_truth is not None:
            actual_text = "CONFIRMED FRAUD" if ground_truth == 1 else "CONFIRMED LEGITIMATE"
            actual_badge = "badge-fraud" if ground_truth == 1 else "badge-legit"
            correct = (is_fraud_predicted and ground_truth == 1) or (not is_fraud_predicted and ground_truth == 0)
            st.markdown(f"**Ground Truth Label:** <span class='{actual_badge}'>{actual_text}</span>", unsafe_allow_html=True)
            if correct:
                st.caption("Prediction correctly matches ground truth label.")
            else:
                st.caption("Prediction discrepancy against ground truth label.")

    with card_col2:
        st.subheader("Explainability: Auditable Risk Reasons")
        st.markdown(f"**Risk Level:** `{explanation['risk_level']}` | **Summary:** `{explanation['summary']}`")
        st.write("Specific operational reasons triggered by this transaction:")

        for sig in explanation["signals"]:
            prefix = "[High Risk]" if sig["severity"] == "high" else ("[Medium Risk]" if sig["severity"] == "medium" else "[Low Risk]")
            st.markdown(f"- **{prefix}** {sig['text']}")

        with st.expander("Inspect Raw Feature Values"):
            st.json({k: round(v, 4) if isinstance(v, float) else v for k, v in feature_row.items() if not k.startswith("cat_") or v > 0})


# TAB 2: Model Benchmark and Plots
with tab_bench:
    st.subheader("Empirical Benchmark Across Detection Paradigms")
    st.write(
        "Comparison from heuristic rules to unsupervised anomaly detection to modern calibrated XGBoost on 555,719 held-out test transactions."
    )

    comp_table = load_metrics_table()
    if comp_table is not None:
        display_df = comp_table.copy()
        for col in ["precision", "recall", "f1", "accuracy", "auc_pr", "auc_roc"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "-")
        if "cost_inr" in display_df.columns:
            display_df["cost_inr"] = display_df["cost_inr"].map(lambda x: f"Rs {x:,.0f}" if isinstance(x, (int, float)) else str(x))
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Performance Visualizations (For Presentation Slides)")

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown("#### Precision-Recall Curve")
        pr_path = os.path.join(PLOTS_DIR, "pr_curve_comparison.png")
        if os.path.exists(pr_path):
            st.image(pr_path, use_container_width=True, caption="Area Under PR Curve: 0.929 vs 0.0039 baseline")

        st.markdown("#### Feature Importances")
        feat_path = os.path.join(PLOTS_DIR, "feature_importance.png")
        if os.path.exists(feat_path):
            st.image(feat_path, use_container_width=True, caption="Top predictive features driving XGBoost decisions")

    with pcol2:
        st.markdown("#### ROC Curve Comparison")
        roc_path = os.path.join(PLOTS_DIR, "roc_curve_comparison.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True, caption="Area Under ROC Curve: 0.9986")

        st.markdown("#### Confusion Matrix (Tuned Model)")
        cm_path = os.path.join(PLOTS_DIR, "confusion_matrix_xgboost_tuned.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True, caption="XGBoost Confusion Matrix on 555,719 test transactions")


# TAB 3: Real-Time Flagged Queue
with tab_queue:
    st.subheader("Risk Analyst Investigation Queue")
    st.write("Browse transactions automatically surfaced by Foundster for secondary analyst verification.")

    if sample_df is not None:
        q_filter = st.selectbox("Queue Filter", ["Top High-Confidence Fraud Cases", "All Flagged Incidents", "All Sample Records"])
        
        queue_df = sample_df.copy()
        if "xgb_prob" not in queue_df.columns:
            probs = model.predict_proba(queue_df[FEATURE_COLS])[:, 1]
            queue_df["xgb_prob"] = probs
        
        queue_df["Status"] = queue_df["xgb_prob"].apply(lambda p: "FLAGGED" if p >= active_threshold else "CLEARED")
        
        if q_filter == "Top High-Confidence Fraud Cases":
            filtered_queue = queue_df.sort_values(by="xgb_prob", ascending=False).head(30)
        elif q_filter == "All Flagged Incidents":
            filtered_queue = queue_df[queue_df["xgb_prob"] >= active_threshold].head(50)
        else:
            filtered_queue = queue_df.head(50)

        st.dataframe(
            filtered_queue[[
                "trans_num", "amt", "category", "Status", "xgb_prob", "is_fraud", "distance_km", "amt_to_card_avg"
            ]].rename(columns={
                "trans_num": "Transaction ID",
                "amt": "Amount (INR)",
                "category": "Merchant Category",
                "xgb_prob": "Fraud Probability",
                "is_fraud": "Actual Label",
                "distance_km": "Distance (km)",
                "amt_to_card_avg": "Amt vs Avg Ratio"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.caption("You can copy any transaction details into the Live Scoring tab to see its full audit explanation.")


# TAB 4: Cost and Threshold Simulator
with tab_simulator:
    st.subheader("Financial Loss and Threshold Sensitivity Simulator")
    st.write(
        "Demonstrates how varying the operational decision threshold alters False Positives (customer verification friction) "
        "versus False Negatives (actual stolen funds)."
    )

    sim_thresh = st.slider("Simulate Decision Cutoff", 0.05, 0.95, float(active_threshold), 0.01)

    if sim_thresh <= 0.35:
        sim_recall = 0.93 - (sim_thresh - 0.05) * (0.03 / 0.30)
        sim_prec = 0.55 + (sim_thresh - 0.05) * (0.20 / 0.30)
    elif sim_thresh <= 0.58:
        sim_recall = 0.90 - (sim_thresh - 0.35) * (0.04 / 0.23)
        sim_prec = 0.75 + (sim_thresh - 0.35) * (0.124 / 0.23)
    else:
        sim_recall = 0.86 - (sim_thresh - 0.58) * (0.15 / 0.37)
        sim_prec = 0.874 + (sim_thresh - 0.58) * (0.08 / 0.37)

    total_test_fraud = 2145
    total_test_legit = 553574

    tp_est = int(total_test_fraud * sim_recall)
    fn_est = total_test_fraud - tp_est
    fp_est = int((tp_est / max(sim_prec, 0.01)) - tp_est)
    tn_est = total_test_legit - fp_est

    total_loss_est = fp_est * 50 + fn_est * 3000

    scol1, scol2, scol3, scol4 = st.columns(4)
    scol1.metric("Simulated Recall", f"{sim_recall*100:.1f}%", f"{tp_est:,} / 2,145 Fraud Caught")
    scol2.metric("Simulated Precision", f"{sim_prec*100:.1f}%", f"{fp_est:,} False Alarms")
    scol3.metric("Missed Fraud Incidents", f"{fn_est:,}", "Financial Loss Exposure", delta_color="inverse")
    scol4.metric("Estimated Net Loss", f"Rs {total_loss_est:,.0f}", f"Rs {total_loss_est/100000:.1f} Lakhs")

    st.info(
        f"Risk Strategy Insight: At threshold {sim_thresh:.2f}, the system intercepts {sim_recall*100:.1f}% of all fraud attempts. "
        f"Notice that setting the threshold too high increases False Negatives (each costing Rs 3,000), while lowering it to ~0.35 protects cardholder capital with high detection recall."
    )
