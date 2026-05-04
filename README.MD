```markdown
# Credit Card Customer Segmentation
### Using K-Means Clustering + PCA | Unsupervised Machine Learning

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Type](https://img.shields.io/badge/Type-Unsupervised%20ML-purple)

---

## Project Overview

This project performs **customer segmentation** on credit card usage data over the last 6 months using the **K-Means clustering** algorithm combined with **Principal Component Analysis (PCA)** for dimensionality reduction.

The goal is to help a financial institution identify distinct behavioral profiles among their cardholders, enabling more **personalized and targeted marketing strategies**.

> **Dataset source:** Google BigQuery (`ftds-hacktiv8-project.phase1_ftds_037_hck.credit-card-information`)  
> Only odd-numbered `CUST_ID` rows were used (filtered via SQL query).

---

## Repository Structure

```
credit-card-customer-segmentation/
├── notebook.ipynb        # Main notebook: EDA, Feature Engineering, Modeling
├── inference.ipynb       # Inference notebook: predict cluster for new customer
├── credit_card_data.csv  # Dataset
├── medians.pkl           # Saved median values (for missing value imputation)
├── scaler.pkl            # Saved StandardScaler
├── pca.pkl               # Saved PCA model (5 components)
├── kmeans.pkl            # Saved KMeans model (k=3)
└── README.md
```

---

## Dataset Features

| Feature | Description |
|---|---|
| `BALANCE` | Balance amount left in account |
| `PURCHASES` | Amount of purchases made |
| `ONEOFF_PURCHASES` | Maximum purchase amount in one-off |
| `INSTALLMENTS_PURCHASES` | Amount of purchase in installments |
| `CASH_ADVANCE` | Cash advance amount given by user |
| `PURCHASES_FREQUENCY` | How frequently purchases are made |
| `CASH_ADVANCE_FREQUENCY` | How frequently cash advances happen |
| `CREDIT_LIMIT` | Credit limit of the card |
| `PAYMENTS` | Amount of payment done |
| `MINIMUM_PAYMENTS` | Minimum payment amount |
| `PRC_FULL_PAYMENT` | Percentage of full payment paid by user |
| `TENURE` | Tenure of credit card service |
| ... | *(17 features total)* |

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Analyzed distribution skewness across all features
- Investigated effect of `TENURE` on `PURCHASES`, `BALANCE`, and `PAYMENTS` using **Spearman correlation** (due to non-normal distributions)
- Analyzed `CREDIT_LIMIT` vs `PURCHASES_FREQUENCY` relationship — result: very weak correlation (0.085)

### 2. Feature Engineering Pipeline

```
Raw Data
   -> Handle Missing Values     : Fill MINIMUM_PAYMENTS with median
   -> Outlier Handling          : Log Transform (log1p) — preserves all data
   -> Scaling                   : StandardScaler
   -> Dimensionality Reduction  : PCA (5 components, ~85% variance retained)
```

> **Why Log Transform instead of removing outliers?**  
> In customer segmentation, extreme values represent real behavioral segments (e.g., heavy cash advance users). Removing them would cause information loss. Log transform compresses the scale while keeping all data points.

### 3. Optimal K Selection

| Method | Result |
|---|---|
| Elbow Method | Elbow point at k=3 |
| Silhouette Score | k=2 is highest mathematically, but k=3 gives richer segmentation |

**Final choice: k=3** — balances statistical validity with business interpretability.

### 4. Model Evaluation

| Metric | Value | Interpretation |
|---|---|---|
| Silhouette Score | **0.2898** | Decent cluster separation |
| Davies-Bouldin Index | **1.3608** | Low = compact & well-separated clusters |
| Calinski-Harabasz Index | **1999.66** | High = clusters are distinct |

---

## Clustering Results

Three distinct customer profiles were identified:

| Cluster | Label | Profile |
|---|---|---|
| **Cluster 0** | Passive Low-Transaction | Lowest spending, low balance, most inactive users |
| **Cluster 1** | Cash Advance Focused | High cash advance usage, high balance, low purchases |
| **Cluster 2** | High-Value Top Shopper | Most active buyers, highest credit limit, most frequent transactions |

### Key Business Insight
> Credit limit (`CREDIT_LIMIT`) has almost no correlation with purchase frequency (Spearman: 0.085). Increasing credit limits alone will not drive more transactions — personalized strategies per cluster are necessary.

---

## Inference: Predict Cluster for New Customer

The `inference.ipynb` notebook demonstrates how to classify a **new, unseen customer** using the saved model artifacts.

### Inference Pipeline

```python
import joblib, numpy as np, pandas as pd

# Load all saved components
medians = joblib.load('medians.pkl')
scaler  = joblib.load('scaler.pkl')
pca     = joblib.load('pca.pkl')
kmeans  = joblib.load('kmeans.pkl')

# New customer data
new_customer = {
    'BALANCE': 2000, 'PURCHASES': 1000, 'CASH_ADVANCE': 500,
    'CREDIT_LIMIT': 5000, 'PAYMENTS': 800, 'TENURE': 12,
    # ... (all 17 features)
}

# Pipeline
df_new    = pd.DataFrame([new_customer])
df_filled = df_new.fillna(medians)       # Step 1: Missing value
df_log    = np.log(df_filled + 1)        # Step 2: Log transform
df_scaled = scaler.transform(df_log)     # Step 3: Scaling
df_pca    = pca.transform(df_scaled)     # Step 4: PCA
cluster   = kmeans.predict(df_pca)       # Step 5: Predict
```

**Example Output:**
```
=============================================
HASIL PREDIKSI UNTUK NASABAH ID 9999
---------------------------------------------
Cluster ID : 2
Kategori   : High-Value (Top Shopper)
=============================================
```

---

## Tech Stack

- **Language:** Python 3.9
- **Libraries:** `pandas` `numpy` `scikit-learn` `matplotlib` `seaborn` `joblib`
- **Algorithm:** K-Means Clustering
- **Dimensionality Reduction:** PCA (Principal Component Analysis)

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/credit-card-customer-segmentation.git
   cd credit-card-customer-segmentation
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
   ```

3. **Run the main notebook**
   ```bash
   jupyter notebook notebook.ipynb
   ```

4. **Run inference for a new customer**
   ```bash
   jupyter notebook inference.ipynb
   ```

---

