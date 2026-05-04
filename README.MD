---

# Credit Card Customer Segmentation
### K-Means Clustering + PCA | Unsupervised Machine Learning

---

## Overview

Customer segmentation on credit card transaction behavior data over 6 months using K-Means Clustering and PCA. Dataset contains 17 features across ~4,500 rows, sourced from Google BigQuery — only odd-numbered `CUST_ID` rows were extracted via SQL query. The raw CSV is available in this repository as `credit_card_data.csv`.

**Goal:** help a financial institution identify distinct customer behavioral profiles for more targeted marketing strategies.

---

## Results

Model with **k=3** produces three distinct customer profiles:

| Cluster | Profile | Description |
|---|---|---|
| 0 | Passive Low-Transaction | Lowest spending and balance, most inactive users |
| 1 | Cash Advance Focused | Heavy cash withdrawal, high balance, rarely purchases |
| 2 | High-Value Top Shopper | Most active buyer, largest credit limit, highest transaction count |

> **Key Insight:** `CREDIT_LIMIT` has almost no correlation with purchase frequency (Spearman: 0.085) — raising credit limits alone will not drive more transactions.

---

## ML Pipeline

```
Raw Data
   -> Missing Value  : Fill MINIMUM_PAYMENTS with median
   -> Outlier        : Log Transform (log1p) — preserves extreme values instead of removing them
   -> Scaling        : StandardScaler
   -> Dim. Reduction : PCA 5 components (~85% variance retained)
   -> Clustering     : K-Means k=3
```

**k=3 selection** based on Elbow Method + Silhouette Analysis. Although k=2 yields a higher Silhouette Score, k=3 was chosen for richer and more actionable business segmentation.

**Model evaluation:**

| Metric | Score | Note |
|---|---|---|
| Silhouette Score | 0.2898 | Decent cluster separation |
| Davies-Bouldin Index | 1.3608 | Lower is better |
| Calinski-Harabasz Index | 1999.66 | Higher is better |

---

## Inference

`inference.ipynb` accepts new customer data and predicts its cluster using saved model artifacts.

```python
medians = joblib.load('medians.pkl')   # impute missing values
scaler  = joblib.load('scaler.pkl')    # standardization
pca     = joblib.load('pca.pkl')       # dimensionality reduction
kmeans  = joblib.load('kmeans.pkl')    # clustering model

# pipeline: fillna -> log1p -> scale -> pca -> predict
```

---

## File Structure

```
credit-card-customer-segmentation/
├── notebook.ipynb        # EDA, feature engineering, modeling
├── inference.ipynb       # Predict cluster for new customer
├── credit_card_data.csv  # Dataset
├── medians.pkl           # Saved medians for imputation
├── scaler.pkl            # Saved StandardScaler
├── pca.pkl               # Saved PCA (5 components)
├── kmeans.pkl            # Saved KMeans (k=3)
└── README.md
```

---

## Tech Stack

- **Language:** Python 3.9
- **Libraries:** `pandas` `numpy` `scikit-learn` `matplotlib` `seaborn` `joblib`

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/your-username/credit-card-customer-segmentation.git
cd credit-card-customer-segmentation
```

2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

3. Run main notebook
```bash
jupyter notebook notebook.ipynb
```

4. Run inference
```bash
jupyter notebook inference.ipynb
```

---
