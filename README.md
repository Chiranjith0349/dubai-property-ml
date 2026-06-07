# Dubai Property Price Intelligence

> Price prediction model for Dubai real estate, trained on ~5,000 property listings scraped from local portals.
> Full pipeline from raw CSV to a live Streamlit dashboard - SQLite data layer, EDA, feature engineering, model comparison, and hyperparameter tuning.

---

## Live Demo
🔗 [dubai-property-ml.streamlit.app](https://dubai-property-ml-nqcgjvbkvsum8akhzmhd23.streamlit.app/)

---

## Project Structure

```
dubai-property-ml/
├── data/
│   ├── DOWNLOAD_INSTRUCTIONS.md   # How to get the dataset
│   ├── properties.csv             # Raw data
│   ├── properties.db              # SQLite database (auto-generated, gitignored)
│   ├── best_model.joblib          # Trained model
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
The raw CSV is loaded into SQLite via `src/database.py`. Every data pull into
pandas goes through a named SQL query rather than reading the CSV directly —
makes it easy to swap to a real database and keeps the data layer reusable.

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
The full sklearn Pipeline (preprocessor + model) is serialised with `joblib`.
The Streamlit app loads it once at startup and calls `model.predict(input_df)` —
the same object that was trained, so there's no preprocessing mismatch.

---

## Results

| Model | Test R² | Test RMSE (AED) | Test MAE (AED) |
|---|---|---|---|
| Ridge Regression | 0.929 | 4,438,793 | 753,248 |
| Gradient Boosting | 0.949 | 1,338,070 | 528,515 |
| **Random Forest (tuned)** | **0.970** | **930,937** | **273,730** |

Random Forest outperformed Gradient Boosting on this dataset — common with ~5K records where boosting can overfit the noise. The tuned model explains **97% of price variance** with a mean absolute error of **AED 274K** on properties averaging AED 3.6M (~7.6% error).

---

## Tech Stack

`Python` · `SQL / SQLite` · `pandas` · `numpy` · `scikit-learn` · `Streamlit` · `Plotly`

---

## Author

**Chiranjith Pradeep**
B.E. Computer Science — BITS Pilani Dubai
[github.com/Chiranjith0349](https://github.com/Chiranjith0349)
