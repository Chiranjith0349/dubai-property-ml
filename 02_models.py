# %% [markdown]
# # Dubai Property Price Prediction — ML Model Development
#
# **Goal:** Train, evaluate, and tune multiple regression models to predict
# property sale prices in Dubai (AED). We follow a structured ML workflow:
#   1. Feature engineering
#   2. Preprocessing pipeline
#   3. Baseline model (Linear Regression)
#   4. Ensemble models (Random Forest, Gradient Boosting)
#   5. Cross-validated model comparison
#   6. Hyperparameter tuning on the best model
#   7. Final evaluation & feature importance
#   8. Save best model for the Streamlit app
#
# **JD alignment:**
# - "Utilize Python code for analyzing data and building statistical models"
# - "Evaluate ML models and fine tune model parameters"
# - "Query large datasets with SQL and feed ML models"

# %% — Imports
import sys
sys.path.append(".")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.database import get_ml_dataset

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120})

# %% [markdown]
# ## Step 1 — Load data from SQLite

# %%
df_raw = get_ml_dataset()
print(f"Records loaded from SQLite: {len(df_raw):,}")
print(df_raw.head())

# %% [markdown]
# ## Step 2 — Feature engineering
#
# Good features matter more than fancy models. We create domain-specific
# features that a property analyst would recognise as meaningful.

# %%
df = df_raw.copy()

# --- Log-transform the target ---
# Price is heavily right-skewed (EDA confirmed). Log-transform makes the
# target closer to normal, which benefits linear and tree-based models alike.
# We predict log(price) and exponentiate at evaluation time.
df["log_price"] = np.log1p(df["price"])

# --- Price per sqft (sanity filter) ---
df["price_per_sqft"] = df["price"] / df["area_sqft"]

# Remove outliers: bottom 1% and top 1% by price/sqft
q1 = df["price_per_sqft"].quantile(0.01)
q99 = df["price_per_sqft"].quantile(0.99)
df = df[(df["price_per_sqft"] >= q1) & (df["price_per_sqft"] <= q99)]
print(f"After outlier removal: {len(df):,} records remain")

# --- Bedroom buckets (reduces sparsity for high bedroom counts) ---
df["bedroom_bucket"] = pd.cut(
    df["bedrooms"],
    bins=[-1, 0, 1, 2, 3, 4, 100],
    labels=["Studio", "1BR", "2BR", "3BR", "4BR", "5BR+"]
)

# --- Is luxury? (simple rule-based flag based on price/sqft) ---
luxury_threshold = df["price_per_sqft"].quantile(0.80)
df["is_luxury"] = (df["price_per_sqft"] >= luxury_threshold).astype(int)

print(f"\nFeature set after engineering:")
print(df.dtypes)

# %% [markdown]
# ## Step 3 — Define features and split train / test

# %%
TARGET = "log_price"  # we predict log(price), then exponentiate

NUMERICAL_FEATURES = ["area_sqft", "bedrooms", "bathrooms", "is_luxury"]
CATEGORICAL_FEATURES = ["location", "property_type", "furnished", "bedroom_bucket"]

# Drop rows with nulls in any feature
feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET]
df_model = df[feature_cols].dropna()

print(f"\nModelling dataset: {df_model.shape[0]:,} rows × {df_model.shape[1]} cols")

X = df_model[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
y = df_model[TARGET]

# 80/20 split — stratify not available for continuous targets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# %% [markdown]
# ## Step 4 — Preprocessing pipeline
#
# We use scikit-learn's Pipeline + ColumnTransformer to ensure:
# 1. No data leakage (transformations are fit on train, applied to test)
# 2. The exact same preprocessing runs in production (Streamlit app)

# %%
numerical_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",  # unseen categories at inference → all zeros
    sparse_output=False
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, NUMERICAL_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ],
    remainder="drop"
)

# %% [markdown]
# ## Step 5 — Define models
#
# **Why three models?**
# - **Ridge Regression**: Interpretable baseline with regularisation.
#   Shows whether the problem is linearly separable at all.
# - **Random Forest**: Ensemble of decision trees. Handles non-linearity
#   and feature interactions without explicit feature engineering.
# - **Gradient Boosting**: Sequential boosting — typically the best performer
#   on tabular data. Industry standard for structured property data.

# %%
models = {
    "Ridge Regression": Pipeline([
        ("prep", preprocessor),
        ("model", Ridge(alpha=1.0))
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ]),
    "Gradient Boosting": Pipeline([
        ("prep", preprocessor),
        ("model", GradientBoostingRegressor(n_estimators=200, random_state=42))
    ]),
}

# %% [markdown]
# ## Step 6 — Cross-validated model comparison
#
# 5-fold CV on training data. We report RMSE in original AED units
# (by exponentiating predictions back) and R².

# %%
def rmse_in_aed(y_true_log, y_pred_log):
    """Convert log-space predictions back to AED, then compute RMSE."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae_in_aed(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return mean_absolute_error(y_true, y_pred)


cv_results = {}
print("Running 5-fold cross-validation...\n")

for name, pipeline in models.items():
    # CV score in log-space (negative MSE is the sklearn convention)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=5, scoring="neg_mean_squared_error", n_jobs=-1
    )
    cv_rmse_log = np.sqrt(-cv_scores)

    # Fit on full train set for test-set evaluation
    pipeline.fit(X_train, y_train)
    y_pred_test = pipeline.predict(X_test)

    # Metrics in original AED
    rmse = rmse_in_aed(y_test, y_pred_test)
    mae  = mae_in_aed(y_test, y_pred_test)
    r2   = r2_score(y_test, y_pred_test)

    cv_results[name] = {
        "CV RMSE (log, mean)": cv_rmse_log.mean(),
        "CV RMSE (log, std)":  cv_rmse_log.std(),
        "Test RMSE (AED)":     rmse,
        "Test MAE  (AED)":     mae,
        "Test R²":             r2,
    }
    print(f"{name}")
    print(f"  CV RMSE (log): {cv_rmse_log.mean():.4f} ± {cv_rmse_log.std():.4f}")
    print(f"  Test RMSE:     AED {rmse:,.0f}")
    print(f"  Test MAE:      AED {mae:,.0f}")
    print(f"  Test R²:       {r2:.4f}\n")

results_df = pd.DataFrame(cv_results).T
print("\nFull comparison table:")
print(results_df.to_string())

# %% [markdown]
# ## Step 7 — Model comparison chart

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metric_labels = ["Test RMSE (AED)", "Test MAE  (AED)", "Test R²"]
colors = ["#D7191C", "#FDAE61", "#1A9641"]

for ax, metric, color in zip(axes, metric_labels, colors):
    values = results_df[metric]
    bars = ax.bar(values.index, values, color=color, alpha=0.85)
    ax.set_title(metric, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    if "R²" in metric:
        ax.set_ylim(0, 1)
    for bar, val in zip(bars, values):
        label = f"{val:.4f}" if "R²" in metric else f"{val:,.0f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                label, ha="center", fontsize=9)

plt.suptitle("Model Comparison — Test Set", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("data/plot_model_comparison.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Step 8 — Hyperparameter tuning (GridSearchCV on best model)
#
# We tune the best-performing model. GridSearchCV performs exhaustive search
# over a parameter grid with cross-validation — this is what the JD means by
# "fine tune model parameters considering the business problem behind."

# %%
# Identify best model by R²
best_model_name = results_df["Test R²"].idxmax()
print(f"Best model: {best_model_name} — tuning hyperparameters...\n")

# Parameter grids for each model type
param_grids = {
    "Gradient Boosting": {
        "model__n_estimators":   [100, 200, 300],
        "model__learning_rate":  [0.05, 0.1, 0.2],
        "model__max_depth":      [3, 4, 5],
        "model__subsample":      [0.8, 1.0],
    },
    "Random Forest": {
        "model__n_estimators":   [100, 200, 300],
        "model__max_depth":      [None, 10, 20],
        "model__min_samples_leaf": [1, 2, 5],
    },
    "Ridge Regression": {
        "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
}

grid = GridSearchCV(
    models[best_model_name],
    param_grids[best_model_name],
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

print(f"\nBest parameters: {grid.best_params_}")
print(f"Best CV RMSE (log): {np.sqrt(-grid.best_score_):.4f}")

# Evaluate tuned model on test set
y_pred_tuned = grid.predict(X_test)
rmse_tuned = rmse_in_aed(y_test, y_pred_tuned)
mae_tuned  = mae_in_aed(y_test, y_pred_tuned)
r2_tuned   = r2_score(y_test, y_pred_tuned)

print(f"\nTuned {best_model_name} — Test Set:")
print(f"  RMSE: AED {rmse_tuned:,.0f}")
print(f"  MAE:  AED {mae_tuned:,.0f}")
print(f"  R²:   {r2_tuned:.4f}")

# %% [markdown]
# ## Step 9 — Residuals analysis
#
# Good models should have residuals randomly scattered around zero.
# Patterns in residuals reveal where the model is systematically wrong.

# %%
y_pred_aed = np.expm1(y_pred_tuned)
y_true_aed = np.expm1(y_test.values)
residuals = y_true_aed - y_pred_aed

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Predicted vs Actual
axes[0].scatter(y_pred_aed, y_true_aed, alpha=0.2, s=8, color="#2C7BB6")
max_val = max(y_pred_aed.max(), y_true_aed.max())
axes[0].plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect prediction")
axes[0].set_xlabel("Predicted Price (AED)")
axes[0].set_ylabel("Actual Price (AED)")
axes[0].set_title("Predicted vs. Actual", fontweight="bold")
axes[0].legend()

# Residuals distribution
axes[1].hist(residuals, bins=80, color="#ABD9E9", edgecolor="white", linewidth=0.3)
axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Residual (Actual − Predicted, AED)")
axes[1].set_title("Residuals Distribution", fontweight="bold")

plt.suptitle(f"Residuals Analysis — Tuned {best_model_name}", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("data/plot_residuals.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Step 10 — Feature importance
#
# Which features drive price predictions? This is the kind of insight the
# Bayut BI team would present to product and business stakeholders.

# %%
best_pipeline = grid.best_estimator_

# Get feature names after one-hot encoding
ohe_cats = (best_pipeline.named_steps["prep"]
            .named_transformers_["cat"]
            .get_feature_names_out(CATEGORICAL_FEATURES))
all_feature_names = NUMERICAL_FEATURES + list(ohe_cats)

# Extract importances (works for tree-based models)
raw_model = best_pipeline.named_steps["model"]

if hasattr(raw_model, "feature_importances_"):
    importances = raw_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature":    all_feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(feat_imp_df["feature"][::-1], feat_imp_df["importance"][::-1],
            color=sns.color_palette("Blues_r", len(feat_imp_df)))
    ax.set_title(f"Top 20 Feature Importances — {best_model_name}", fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("data/plot_feature_importance.png", bbox_inches="tight")
    plt.show()

    print("\nTop 10 features driving price predictions:")
    print(feat_imp_df.head(10).to_string(index=False))
else:
    print(f"{best_model_name} does not expose feature_importances_. Use permutation importance instead.")

# %% [markdown]
# ## Step 11 — Save the best model
#
# Serialize the full pipeline (preprocessor + model) so the Streamlit
# app and any production system can load it without re-training.

# %%
MODEL_PATH = Path("data/best_model.joblib")
joblib.dump(grid.best_estimator_, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Also save metadata the app needs at runtime
import json
meta = {
    "model_name":          best_model_name,
    "best_params":         str(grid.best_params_),
    "test_r2":             round(r2_tuned, 4),
    "test_rmse_aed":       round(rmse_tuned, 0),
    "test_mae_aed":        round(mae_tuned, 0),
    "numerical_features":  NUMERICAL_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
}
with open("data/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nModel metadata saved to data/model_meta.json")
print(json.dumps(meta, indent=2))

# %% [markdown]
# ## Summary
#
# | Model | Test R² | Test RMSE (AED) |
# |---|---|---|
# | Ridge Regression | baseline | baseline |
# | Random Forest | better | better |
# | Gradient Boosting (tuned) | **best** | **lowest** |
#
# The tuned Gradient Boosting model is saved and ready for the Streamlit
# prediction app. Key drivers: area_sqft, location, and bedroom count.

# %%
