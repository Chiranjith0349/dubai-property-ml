"""
app.py — Dubai Property Price Intelligence Dashboard
=====================================================
Streamlit app that serves the trained ML model as a live price predictor
and provides market overview analytics for Dubai real estate.

Deploy for free at: https://streamlit.io/cloud
  1. Push this repo to GitHub
  2. Go to share.streamlit.io → New app → point to app.py
  3. Done — shareable link in 2 minutes.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.database import (
    get_connection,
    q_price_by_location,
    q_price_by_type,
    q_price_by_bedrooms,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dubai Property Price Intelligence",
    page_icon="🏙️",
    layout="wide",
)

# ── Load model & metadata ──────────────────────────────────────────────────────
MODEL_PATH = Path("data/best_model.joblib")
META_PATH  = Path("data/model_meta.json")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_meta():
    if not META_PATH.exists():
        return {}
    with open(META_PATH) as f:
        return json.load(f)

model = load_model()
meta  = load_meta()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🏙️ Dubai Property Price Intelligence")
st.markdown(
    "**ML-powered price prediction** for Dubai real estate — trained on ~5,000 public property listings."
)

if model is None:
    st.warning(
        "⚠️ Model not found. Run `02_models.py` first to train and save the model, "
        "then relaunch this app."
    )
    st.stop()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_predict, tab_market, tab_methodology = st.tabs(
    ["💰 Price Predictor", "📊 Market Overview", "⚙️ Methodology"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRICE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("Predict a Property's Sale Price")
    st.markdown("Fill in the property details and click **Predict**.")

    # Load location + type options from the database
    @st.cache_data
    def get_options():
        try:
            loc_df  = q_price_by_location(top_n=50)
            type_df = q_price_by_type()
            locations = sorted(loc_df["location"].dropna().tolist())
            types     = sorted(type_df["property_type"].dropna().tolist())
        except Exception:
            locations = ["Downtown Dubai", "Dubai Marina", "JBR", "Palm Jumeirah", "JVC", "Business Bay"]
            types     = ["Apartment", "Villa", "Townhouse", "Penthouse", "Studio"]
        return locations, types

    locations, prop_types = get_options()

    col1, col2 = st.columns(2)

    with col1:
        location      = st.selectbox("Neighbourhood / Area", locations)
        property_type = st.selectbox("Property Type", prop_types)
        furnished     = st.selectbox("Furnishing", ["Furnished", "Unfurnished", "Semi-Furnished"])

    with col2:
        bedrooms  = st.slider("Bedrooms", 0, 7, 2, help="0 = Studio")
        bathrooms = st.slider("Bathrooms", 1, 8, 2)
        area_sqft = st.number_input("Area (sqft)", min_value=200, max_value=30000,
                                    value=1200, step=50)

    # Derived features (must match 02_models.py engineering exactly)
    bed_labels = ["Studio", "1BR", "2BR", "3BR", "4BR", "5BR+"]
    bed_bucket = bed_labels[min(bedrooms, 5)]

    # We don't know is_luxury until the model runs — use a placeholder
    # The model was trained with is_luxury based on price_per_sqft percentile.
    # At inference, we default to 0; users can override with the checkbox.
    is_luxury = int(st.checkbox("Mark as luxury / premium property", value=False))

    predict_clicked = st.button("🔮 Predict Price", type="primary", use_container_width=True)

    if predict_clicked:
        input_df = pd.DataFrame([{
            "area_sqft":     area_sqft,
            "bedrooms":      bedrooms,
            "bathrooms":     bathrooms,
            "is_luxury":     is_luxury,
            "location":      location,
            "property_type": property_type,
            "furnished":     furnished,
            "bedroom_bucket": bed_bucket,
        }])

        log_pred = model.predict(input_df)[0]
        predicted_price = np.expm1(log_pred)

        # Confidence band: ±MAE from model metadata
        mae = meta.get("test_mae_aed", predicted_price * 0.1)
        low  = max(0, predicted_price - mae)
        high = predicted_price + mae

        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Estimated Price", f"AED {predicted_price:,.0f}")
        col_b.metric("Low estimate",    f"AED {low:,.0f}")
        col_c.metric("High estimate",   f"AED {high:,.0f}")

        st.caption(
            f"Model: {meta.get('model_name', 'ML Model')}  |  "
            f"Test R²: {meta.get('test_r2', '—')}  |  "
            f"MAE band: ±AED {mae:,.0f}"
        )

        price_per_sqft = predicted_price / area_sqft
        st.info(f"📐 Implied price per sqft: **AED {price_per_sqft:,.0f}**")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_market:
    st.subheader("Dubai Real Estate Market — Key Metrics")

    @st.cache_data
    def load_market_data():
        try:
            loc_df  = q_price_by_location(top_n=20)
            type_df = q_price_by_type()
            bed_df  = q_price_by_bedrooms()
            return loc_df, type_df, bed_df
        except Exception as e:
            st.error(f"Could not load market data: {e}")
            return None, None, None

    loc_df, type_df, bed_df = load_market_data()

    if loc_df is not None:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                loc_df.sort_values("avg_price_aed"),
                x="avg_price_aed", y="location",
                orientation="h",
                title="Average Sale Price by Neighbourhood (Top 20)",
                labels={"avg_price_aed": "Avg Price (AED)", "location": ""},
                color="avg_price_per_sqft",
                color_continuous_scale="Blues",
                text="listings",
            )
            fig.update_traces(texttemplate="%{text} listings", textposition="outside")
            fig.update_layout(height=550, coloraxis_colorbar_title="AED/sqft")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.scatter(
                loc_df,
                x="avg_area_sqft" if "avg_area_sqft" in loc_df.columns else "listings",
                y="avg_price_aed",
                size="listings",
                color="avg_price_per_sqft",
                hover_name="location",
                title="Location Bubble Chart — Price vs. Size",
                labels={
                    "avg_area_sqft": "Avg Area (sqft)",
                    "avg_price_aed": "Avg Price (AED)",
                    "avg_price_per_sqft": "AED/sqft",
                },
                color_continuous_scale="Reds",
            )
            fig2.update_layout(height=550)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            if type_df is not None:
                fig3 = px.bar(
                    type_df,
                    x="property_type", y="avg_price_per_sqft",
                    title="Price per Sqft by Property Type",
                    labels={"avg_price_per_sqft": "AED/sqft", "property_type": ""},
                    color="avg_price_per_sqft",
                    color_continuous_scale="Greens",
                    text_auto=".0f",
                )
                fig3.update_layout(showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig3, use_container_width=True)

        with col4:
            if bed_df is not None:
                fig4 = px.line(
                    bed_df,
                    x="bedrooms", y="avg_price_aed",
                    markers=True,
                    title="How Price Scales with Bedrooms",
                    labels={"avg_price_aed": "Avg Price (AED)", "bedrooms": "Bedrooms"},
                )
                fig4.update_traces(line_color="#D7191C", marker_size=8)
                st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
with tab_methodology:
    st.subheader("How the Model Works")

    st.markdown("""
    ### Data Pipeline
    Raw CSV → **SQLite database** → pandas DataFrame → ML model.
    All data pulls go through named SQL queries — no raw CSV reads in the ML code.

    ### Feature Engineering
    | Feature | Reasoning |
    |---|---|
    | `area_sqft` | Strongest single predictor of price |
    | `bedrooms / bedroom_bucket` | Categorical bucketing reduces sparsity for 5+ BR |
    | `location` | One-hot encoded — neighbourhood commands a large price premium |
    | `property_type` | Villas vs Apartments have structurally different pricing |
    | `furnished` | ~15-20% premium for furnished properties |
    | `is_luxury` | Binary flag for top 20% price/sqft properties |
    | **log(price)** | Target transformation — raw price is highly right-skewed |

    ### Models Compared
    | Model | Why included |
    |---|---|
    | Ridge Regression | Interpretable linear baseline with L2 regularisation |
    | Random Forest | Non-linear ensemble, robust to outliers |
    | **Gradient Boosting** | Sequential boosting — typically wins on tabular data |

    ### Evaluation
    - **5-fold cross-validation** on training set for unbiased model selection
    - **GridSearchCV** for hyperparameter tuning (n_estimators, learning_rate, max_depth)
    - Final metrics on held-out 20% test set: RMSE, MAE, R²
    - Residuals analysis to identify systematic bias

    ### Deployment
    Full sklearn Pipeline (preprocessor + model) serialised with `joblib`.
    One `model.predict(input_df)` call at inference — no preprocessing mismatch possible.
    """)

    if meta:
        st.markdown("### Model Performance (last training run)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Model", meta.get("model_name", "—"))
        col2.metric("Test R²", meta.get("test_r2", "—"))
        col3.metric("Test MAE", f"AED {meta.get('test_mae_aed', 0):,.0f}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built by Chiranjith Pradeep · BITS Pilani Dubai · github.com/Chiranjith0349")
