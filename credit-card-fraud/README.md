# Foundster: Real-Time Credit Card Fraud Detection

Built for the Marvel R&D Lab Open Day Presentation.
Repository branch: credit_card
Dataset: Kaggle Sparkov Simulated Credit Card Transaction Corpus (1.85 Million+ Transactions)

---

## Executive Summary and Benchmark Results

Foundster was evaluated across three progressive detection paradigms on 555,719 completely held-out real credit card transactions:

| Model Setting | Precision | Recall (Detection Rate) | F1-Score | Accuracy | AUC-PR | AUC-ROC | Net Financial Loss (INR) |
|---|---|---|---|---|---|---|---|
| Rule-Based Baseline | 11.32% | 44.62% | 0.1805 | 98.44% | 0.1046 | 0.8340 | Rs 39,39,000 |
| Isolation Forest (Unsupervised) | 19.85% | 26.71% | 0.2277 | 99.30% | 0.1042 | 0.8900 | Rs 48,31,700 |
| XGBoost Balanced Mode (Thresh=0.58) | 87.39% | 85.97% | 0.8667 | 99.90% | 0.9290 | 0.9986 | Rs 9,16,300 |
| **XGBoost High-Recall Mode (Thresh=0.35) [Recommended]** | **76.65%** | **89.84%** | **0.8272** | **99.86%** | **0.9290** | **0.9986** | **Rs 6,83,350** |
| XGBoost Conservative Mode (Thresh=0.75) | 93.48% | 81.49% | 0.8707 | 99.91% | 0.9290 | 0.9986 | Rs 11,97,100 |

### Key Findings
1. AUC-ROC of 0.9986 and AUC-PR of 0.9290: Near-perfect rank separation between legitimate cardholders and fraudulent charges on extreme class imbalance (0.39% fraud rate).
2. 82.7% Net Loss Reduction: Operational losses dropped from Rs 39.4 Lakhs (heuristic rules) and Rs 48.3 Lakhs (unsupervised anomaly detection) down to Rs 6.83 Lakhs in High-Recall mode.
3. 89.84% Fraud Intercept Rate: The High-Recall setting catches 1,927 out of 2,145 fraud attempts, missing only 218 cases across over half a million transactions.
4. Minimal False Alarm Disruption: In Conservative mode, the false positive rate is 0.02% (only 122 false alarms out of 553,574 legitimate transactions).

---

## Architecture and Technical Methodology

### 1. Vectorized Feature Pipeline (src/features.py)
- Cardholder Historical Baselines: Expanding mean, standard deviation, and amount z-scores computed strictly on past transactions using shift(1), guaranteeing zero forward-looking data leakage.
- Card Spending Ratio (amt_to_card_avg): Ratio of the current transaction amount to the cardholder's prior average spend, identifying sudden card-cloning attacks.
- Spatial Geodesics: Exact Haversine great-circle distance between home coordinates and merchant terminal coordinates.
- Circadian Dynamics: 24-hour trigonometric continuous embeddings (hour_sin, hour_cos), day of week, weekend indicators, and elapsed time since the previous transaction.
- Category One-Hot Embeddings: 14 distinct merchant categories to capture vulnerability concentrations in retail and online grocery segments.
- High Throughput: The feature extraction pipeline processes 1.3 million rows in under 4 seconds.

### 2. Tri-Paradigm Comparative Framework (src/models.py)
- Rule-Based Baseline: Static heuristic thresholds on transaction size, velocity, and distance.
- Isolation Forest: Unsupervised non-parametric decision trees identifying multi-dimensional outliers without ground-truth labels.
- Calibrated XGBoost: Gradient-boosted decision trees using histogram methods, weighted loss compensation (scale_pos_weight=4.0), and validation-calibrated decision thresholds.

### 3. Asymmetric Business Loss Function
Rather than evaluating misleading raw accuracy on 99.6% legitimate data, Foundster optimizes for asymmetric financial impact:
Cost = (False Positives * Rs 50) + (False Negatives * Rs 3,000)

### 4. Explainability Engine (src/explain.py)
Converts statistical feature outputs and model confidence into plain-English operational reasons:
- Amount ratio triggers (for example: Amount is 6.8x higher than cardholder usual average).
- Distance anomalies (for example: Terminal location is 245 km away from registered home location).
- High-risk merchant category indicators.
- Time-of-day circadian flags.

---

## Presentation Visuals (results/plots/)

Publication-quality visual plots saved in results/plots/:
- model_comparison.png: Grouped bar chart comparing Precision, Recall, and F1 across paradigms.
- pr_curve_comparison.png: Precision-Recall curves showing the jump from random baseline (0.0039) to 0.929 Area Under the Curve.
- roc_curve_comparison.png: ROC curves showing near-perfect 0.9986 discrimination.
- feature_importance.png: Top 15 ranked predictive features driving XGBoost decisions.
- confusion_matrix_xgboost_tuned.png: Exact confusion matrix showing TP, FP, FN, and TN counts.

---

## Project Structure

```
credit-card-fraud/
|-- app/
|   `-- dashboard.py                 # Streamlit interactive decision dashboard
|-- src/
|   |-- features.py                  # Vectorized feature engineering pipeline
|   |-- models.py                    # Model training, validation calibration, and evaluation
|   |-- evaluate.py                  # Plotting routines for benchmark charts
|   `-- explain.py                   # Plain-English explainability engine
|-- data/
|   |-- features_train.csv           # Engineered training features (1.3M rows)
|   `-- features_test.csv            # Engineered test features (555K rows)
|-- results/
|   |-- model_comparison.csv         # Complete evaluation metrics table
|   |-- test_sample_scored.csv       # Scored test transactions for interactive UI
|   |-- xgb_model.json               # Serialized production XGBoost model artifact
|   `-- plots/                       # High-resolution benchmark charts
`-- README.md                        # Project documentation
```

---

## How to Run

### 1. Activate Environment
```powershell
cd C:\Users\varsh\Desktop\fraud-spike-detector\credit-card-fraud
conda activate frauddet
```

### 2. Feature Extraction (takes ~4 seconds)
```powershell
python src/features.py
```

### 3. Model Training and Benchmark Evaluation (takes ~70 seconds)
```powershell
python src/models.py
```

### 4. Launch the Interactive Dashboard
```powershell
streamlit run app/dashboard.py
```
Open your browser to http://localhost:8501 to access the live scoring interface, explainability audit reasons, real-time flagged queue, and financial threshold simulator.
