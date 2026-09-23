# Fraud-Spike Detector

A multivariate fraud detection system developed in two iterations, progressing from a controlled transaction environment to a real-world credit-card fraud dataset.

The project compares **rule-based detection, unsupervised anomaly detection, and supervised machine learning**, with a focus on time-aware feature engineering, leakage prevention, class imbalance, and interpretable fraud signals.

---

## Project Overview

Fraud detection is difficult because fraudulent transactions are rare, behavior changes over time, and the same transaction can look normal or suspicious depending on the context of the merchant, cardholder, device, amount, and location.

This project was developed in two stages:

1. **`fraud-detector-v1/`** — a controlled transaction environment used to develop and validate the initial fraud-detection pipeline.
2. **`fraud-detector-v2/`** — a real-data implementation using the Kaggle Sparkov credit-card fraud dataset, with richer cardholder-relative features and a more rigorous validation workflow.

The two versions use the same overall philosophy while adapting their features to the structure of the underlying data.

---

# Architecture

```text
Raw Transactions
       │
       ▼
Feature Engineering
       │
       ├── Behavioral features
       ├── Velocity features
       ├── Amount anomalies
       ├── Novelty features
       └── Geographic / temporal context
       │
       ▼
 ┌─────────────────────────────┐
 │     Fraud Detection Models  │
 ├─────────────────────────────┤
 │ Rule-based baseline         │
 │ Isolation Forest            │
 │ XGBoost                     │
 └─────────────────────────────┘
       │
       ▼
Evaluation
       │
       ├── Precision
       ├── Recall
       ├── F1
       ├── Accuracy
       └── Confusion Matrix
```

---

# Version 1 — Controlled Fraud Detector

Location:

```text
fraud-detector-v1/
```

Version 1 establishes the initial fraud-detection pipeline using a controlled transaction dataset.

The feature engineering pipeline is designed around **merchant-relative behavior**: the system asks whether a transaction is unusual compared with the historical behavior of that merchant.

## Features

The initial detector uses five primary features:

### 1. Amount Z-Score

Measures how unusual the current transaction amount is compared with the merchant's prior transaction amounts.

The historical mean and standard deviation are calculated using transactions that occurred before the current transaction.

### 2. Transaction Velocity

Counts transactions occurring within a trailing **5-minute window**.

This captures bursts of activity that may indicate automated or fraudulent behavior.

### 3. Unique Devices

Measures the number of distinct devices appearing in the recent transaction window.

### 4. New Device Detection

Checks whether a device has previously been associated with the merchant.

### 5. Geographic Distance

Measures the distance of the transaction from the merchant's typical geographic location.

All rolling historical features are constructed using prior transactions so that the current transaction does not contribute to its own baseline.

## Models

Version 1 compares three approaches:

### Rule-Based Baseline

A simple interpretable detector that flags transactions when:

* transaction velocity is unusually high,
* transaction amount is a large outlier, or
* a new device appears sufficiently far from the merchant's typical location.

### Isolation Forest

An unsupervised anomaly-detection model.

The model does not use fraud labels during training and instead learns patterns of normal behavior.

### XGBoost

A supervised classifier trained using the available fraud labels.

Class imbalance is addressed through `scale_pos_weight`.

---

# Version 2 — Real-World Credit Card Fraud Detector

Location:

```text
fraud-detector-v2/
```

Version 2 applies the fraud-detection pipeline to the **Kaggle Sparkov credit-card fraud dataset**.

The feature-engineering approach changes significantly from v1.

Instead of asking:

> "Is this transaction unusual for this merchant?"

v2 asks:

> "Is this transaction unusual for this cardholder?"

This reflects the fact that a compromised card can be used across many different merchants.

---

## Version 2 Features

The real-data pipeline creates several cardholder-relative features.

### Amount Anomaly

Measures how unusual the transaction amount is compared with the cardholder's historical spending.

### Category-Relative Amount Anomaly

Measures whether the amount is unusual **within the transaction category**.

This provides additional context because an amount that is unusual for one category may be normal for another.

### Transaction Velocity

Counts transactions made by the cardholder within a trailing **1-hour window**.

The window is wider than v1 because the real-world dataset has a different transaction density.

### New Merchant Detection

Checks whether the cardholder has previously transacted with the merchant.

### Geographic Distance

Calculates the real geographic distance between the cardholder's location and the merchant location using the **Haversine formula**.

### Time Features

The pipeline additionally captures:

* hour of day
* cyclical hour encoding
* weekend indicator

The cyclical representation allows the model to understand that 23:00 and 00:00 are adjacent times rather than numerically distant values.

### Category Frequency

Encodes how frequently a transaction category occurs in the dataset.

---

# Leakage Prevention

A major design principle throughout the project is preventing future information from influencing the current transaction.

For rolling and expanding behavioral statistics, the current transaction is excluded using a prior-transaction shift.

For example:

```python
prior_amt = g["amt"].shift(1)
```

This ensures that the transaction being evaluated does not influence its own historical baseline.

Version 2 also uses a three-way workflow:

```text
Training Data
     │
     ├──────────────► Model training
     │
     ▼
Validation Data
     │
     └──────────────► Hyperparameter / threshold selection
     
Test Data
     │
     └──────────────► Final evaluation
```

The final test set is kept separate from the validation-based tuning process.

---

# Model Comparison

Both versions compare multiple detection strategies:

| Approach            | Type          | Purpose                                        |
| ------------------- | ------------- | ---------------------------------------------- |
| Rule-based baseline | Deterministic | Simple and interpretable control               |
| Isolation Forest    | Unsupervised  | Detect anomalous behavior without fraud labels |
| XGBoost             | Supervised    | Learn fraud patterns from labeled examples     |

This makes it possible to compare a simple business-rule approach against unsupervised anomaly detection and supervised machine learning.

---

# Evaluation

The project evaluates models using:

* Precision
* Recall
* F1 score
* Accuracy
* Confusion matrix

For fraud detection, these metrics should be interpreted together because fraudulent transactions represent a small proportion of the overall transaction population.

The implementation also tracks false positives and false negatives and includes illustrative cost calculations in the model evaluation code.

---

# Running Version 1

From the `fraud-detector-v1` directory:

```bash
cd fraud-detector-v1
```

Install the project dependencies:

```bash
pip install -r ../requirements.txt
```

Generate or use the transaction data and run feature engineering:

```bash
python src/features.py
```

Then train and evaluate the models:

```bash
python src/models.py
```

The resulting feature dataset is written to:

```text
fraud-detector-v1/data/features.csv
```

and the model comparison is written to:

```text
fraud-detector-v1/data/model_comparison.csv
```

---

# Running Version 2

The v2 pipeline requires the Kaggle Sparkov credit-card fraud dataset.

The expected files are:

```text
fraud-detector-v2/
└── data/
    ├── fraudTrain.csv
    ├── fraudTest.csv
    └── src/
        ├── features.py
        └── models.py
```

The large raw CSV files are **not included in this GitHub repository** because they exceed GitHub's standard individual-file size limit.

After placing the datasets in the expected location, run:

```bash
cd fraud-detector-v2/data
python src/features.py
```

This generates:

```text
features_train.csv
features_test.csv
```

Then run:

```bash
python src/models.py
```

The model-comparison output is written to:

```text
data/model_comparison_real.csv
```

---

# Repository Structure

```text
foundster/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── fraud-detector-v1/
│   ├── data/
│   │   ├── features.csv
│   │   ├── model_comparison.csv
│   │   └── transactions.csv
│   │
│   ├── notes/
│   │   ├── intro1.txt
│   │   └── resources.txt
│   │
│   └── src/
│       ├── features.py
│       ├── generate_data.py
│       └── models.py
│
└── fraud-detector-v2/
    └── data/
        ├── README.md
        └── src/
            ├── features.py
            └── models.py
```

The large v2 raw datasets are intentionally excluded from the repository but are required locally to reproduce the complete v2 pipeline.

---

# Key Design Principles

### Time-aware features

Behavioral features are calculated from historical transactions rather than future information.

### Leakage prevention

The transaction currently being evaluated is excluded from its own historical statistics.

### Multiple detection strategies

The project does not rely on a single modeling technique. Rule-based, unsupervised, and supervised approaches are compared.

### Context-aware fraud detection

The feature perspective evolves from merchant-relative behavior in v1 to cardholder-relative behavior in v2.

### Class-imbalance awareness

Fraud is a rare event, so precision, recall, F1, and confusion matrices are considered alongside accuracy.

### Interpretable signals

Many of the model inputs correspond directly to understandable fraud indicators such as unusual amounts, transaction bursts, new merchants/devices, and geographic anomalies.

---

# Future Improvements

Potential extensions include:

* real-time transaction scoring
* streaming feature computation
* model monitoring and drift detection
* additional behavioral features
* calibrated probability outputs
* automated threshold selection using business-specific costs
* deployment as an API service
* integration with real transaction-processing systems

---

## Summary

**Fraud-Spike Detector** explores how fraud detection changes when moving from a controlled environment to a real-world dataset.

Version 1 establishes the core methodology using merchant-relative behavioral features.

Version 2 adapts the same principles to real credit-card transactions by using cardholder-relative behavior, richer temporal and geographic features, and a more rigorous validation workflow.

The result is a progressively developed fraud-detection pipeline that combines interpretable rules, unsupervised anomaly detection, and supervised machine learning.
