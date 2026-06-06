# Dubai Property Price Intelligence

> End-to-end ML project predicting Dubai real estate sale prices using Bayut listing data.
> Covers the full data science lifecycle: SQL → EDA → feature engineering → model training → hyperparameter tuning → live Streamlit dashboard.

---

## Live Demo
🔗 *[Add your Streamlit Cloud link here after deployment]*

---

## Project Motivation

Bayut & dubizzle is the UAE's largest property listings platform. This project
uses their listing data to build a price prediction model — demonstrating the
exact skills required for Bayut's ML Intern role:

- Querying large datasets with SQL and feeding ML models
- Data exploration and quality assessment
- Building and evaluating statistical models in Python
- Hyperparameter tuning with cross-validation
- Building a customer-facing reporting tool (Streamlit dashboard)

---

## Project Structure

```
dubai-property-ml/
├── data/
│   ├── DOWNLOAD_INSTRUCTIONS.md   # How to get the dataset
│   ├── properties.csv             # Raw data (not committed to git)
│   ├── properties.db              # SQLite database (auto-generated)
│   ├── best_model.joblib          # Trained model (auto-generated)
│   ├── model_meta.json            # Model performance metadata
│   └── plot_*.png                 # EDA charts (auto-generated)
├── src/
│   ├── __init__.py
│   └── database.py                # SQLite loader + all SQL query functions
├── 01_eda.py                      # Exploratory data analysis (VS Code interactive)
├── 02_models.py                   # ML pipeline, evaluation, tuning
├── app.py                         # Streamlit prediction dashboard
├── requirements.txt
└── README.md
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the dataset
Follow `data/DOWNLOAD_INSTRUCTIONS.md` — download a Dubai properties CSV from
Kaggle and place it at `data/properties.csv`.

### 3. Run EDA
```bash
# In VS Code: open 01_eda.py and run cells with Shift+Enter
# Or from terminal:
python 01_eda.py
```

### 4. Train the model
```bash
python 02_models.py
```

### 5. Launch the dashboard
```bash
streamlit run app.py
```

---

## Methodology

### Data layer (SQL)
All data is loaded into SQLite via `src/database.py`. Every pull into pandas
goes through a named SQL query — mirroring how real BI teams work with
data warehouses. This ensures the SQL → ML handoff is explicit and auditable.

### Exploratory Data Analysis
- Target distribution: highly right-skewed → log-transform applied
- Strongest correlate: `area_sqft` (Pearson r ≈ 0.7+)
- Location premium: Palm Jumeirah / Downtown 3-5× suburban areas
- Outlier removal: listings outside 1st–99th percentile of price/sqft

### Feature Engineering

| Feature | Type | Notes |
|---|---|---|
| `area_sqft` | Numerical | Strongest predictor |
| `bedrooms` | Numerical | Collinear with area — kept for model |
| `bathrooms` | Numerical | |
| `is_luxury` | Binary | Top 20% price/sqft properties |
| `location` | Categorical | OHE — large price signal |
| `property_type` | Categorical | OHE |
| `furnished` | Categorical | OHE — ~15% premium |
| `bedroom_bucket` | Categorical | Bucketed to reduce sparsity |
| **log(price)** | Target | Applied before training |

### Models

| Model | Description |
|---|---|
| Ridge Regression | L2-regularised linear baseline |
| Random Forest | 100–300 trees, non-linear, robust to outliers |
| **Gradient Boosting** | Sequential boosting — best performer on tabular data |

### Evaluation
- **5-fold cross-validation** on training data for unbiased model selection
- **GridSearchCV** tunes: `n_estimators`, `learning_rate`, `max_depth`, `subsample`
- Final metrics on held-out 20% test set: RMSE, MAE, R²
- Residuals analysis confirms no systematic bias

### Deployment
Full sklearn Pipeline (`ColumnTransformer` → `GradientBoostingRegressor`) is
serialised with `joblib`. The Streamlit app loads it once and calls
`model.predict(input_df)` — no preprocessing mismatch is possible.

---

## Results

| Model | Test R² | Test RMSE (AED) | Test MAE (AED) |
|---|---|---|---|
| Ridge Regression | 0.929 | 4,438,793 | 753,248 |
| Gradient Boosting | 0.949 | 1,338,070 | 528,515 |
| **Random Forest (tuned)** | **0.970** | **930,937** | **273,730** |

Random Forest outperformed Gradient Boosting on this dataset — common with ~5K records where boosting can overfit the noise. The tuned model explains **97% of price variance** with a mean absolute error of **AED 274K** on properties averaging AED 3.6M (~7.6% error).

---

## Skills Demonstrated

`Python` · `SQL / SQLite` · `pandas` · `numpy` · `scikit-learn` · `Gradient Boosting` ·
`GridSearchCV` · `Cross-validation` · `Feature Engineering` · `Streamlit` · `Plotly` ·
`Data Cleaning` · `Statistical Modelling` · `ML in Production`

---

## Author

**Chiranjith Pradeep**
B.E. Computer Science — BITS Pilani Dubai
[github.com/Chiranjith0349](https://github.com/Chiranjith0349)
