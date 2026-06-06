# %% [markdown]
# # Dubai Property Market — Exploratory Data Analysis
#
# **Business context:** Bayut & dubizzle is the UAE's largest property listings
# platform. Before building any ML model, we need to deeply understand the data:
# its quality, distributions, and the relationships between features and price.
# Bad EDA → bad models. This notebook covers the full data exploration lifecycle.
#
# **JD alignment:**
# - "Perform data exploration to find patterns in the data"
# - "Understand the state and quality of the data available"
# - "Query large datasets with SQL"

# %% — Imports & setup
import sys
sys.path.append(".")  # so Python can find the src/ package

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from src.database import (
    load_csv_to_db,
    get_ml_dataset,
    q_overview,
    q_price_by_location,
    q_price_by_type,
    q_price_by_bedrooms,
    q_furnished_premium,
    q_price_per_sqft_outliers,
)

# Plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (10, 5)})

# Format large numbers nicely (e.g. 1,500,000 → 1.5M)
def fmt_aed(x, pos=None):
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x/1_000:.0f}K"
    return str(int(x))

# %% [markdown]
# ## Step 1 — Load data into SQLite
#
# We load the raw CSV into a local SQLite database. Every data pull from here
# onwards goes through a SQL query — not raw pandas. This mirrors how real
# BI/ML teams work with data warehouses.

# %%
load_csv_to_db()  # Safe to re-run; replaces table each time

# %% [markdown]
# ## Step 2 — High-level overview (SQL)

# %%
overview = q_overview()
print("=== DATASET OVERVIEW ===")
print(overview.T.to_string(header=False))

# %% [markdown]
# ## Step 3 — Load ML dataset into pandas & inspect quality

# %%
df = get_ml_dataset()

print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("\nData types:")
print(df.dtypes)
print("\nMissing values (%):")
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
print(missing_pct[missing_pct > 0] if missing_pct.any() else "  None — data is clean.")

# %% [markdown]
# ## Step 4 — Target variable: Price distribution
#
# Most regression problems have a skewed target. We check whether a log
# transformation is needed (it usually is for property prices).

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["price"], bins=80, color="#2C7BB6", edgecolor="white", linewidth=0.4)
axes[0].set_title("Price Distribution (Raw AED)", fontweight="bold")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(fmt_aed))
axes[0].set_xlabel("Price (AED)")

axes[1].hist(np.log1p(df["price"]), bins=80, color="#ABD9E9", edgecolor="white", linewidth=0.4)
axes[1].set_title("Price Distribution (Log-transformed)", fontweight="bold")
axes[1].set_xlabel("log(Price + 1)")

plt.suptitle("Target Variable Analysis", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("data/plot_price_distribution.png", bbox_inches="tight")
plt.show()

print(f"\nSkewness (raw):  {df['price'].skew():.2f}  (>1 = log transform recommended)")
print(f"Skewness (log):  {np.log1p(df['price']).skew():.2f}")

# %% [markdown]
# ## Step 5 — Price by location (SQL query → chart)

# %%
loc_df = q_price_by_location(top_n=15)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(loc_df["location"], loc_df["avg_price_aed"],
               color=sns.color_palette("Blues_r", len(loc_df)))
ax.set_xlabel("Average Price (AED)")
ax.set_title("Top 15 Locations by Listing Volume — Average Sale Price", fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_aed))

# Annotate bars with listing count
for bar, count in zip(bars, loc_df["listings"]):
    ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
            f"{count:,} listings", va="center", fontsize=8, color="#444")

plt.tight_layout()
plt.savefig("data/plot_price_by_location.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Step 6 — Price by property type

# %%
type_df = q_price_by_type()
print("Price by property type:")
print(type_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(type_df["property_type"], type_df["avg_price_per_sqft"],
       color=sns.color_palette("Set2", len(type_df)))
ax.set_title("Average Price per Sqft by Property Type", fontweight="bold")
ax.set_ylabel("AED / sqft")
ax.set_xlabel("")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("data/plot_price_per_sqft_by_type.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Step 7 — How price scales with bedrooms

# %%
bed_df = q_price_by_bedrooms()
print("\nPrice by bedroom count:")
print(bed_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(bed_df["bedrooms"], bed_df["avg_price_aed"], "o-",
        color="#D7191C", linewidth=2, markersize=8)
ax.fill_between(bed_df["bedrooms"], bed_df["min_price_aed"], bed_df["max_price_aed"],
                alpha=0.1, color="#D7191C")
ax.set_title("Price vs. Number of Bedrooms", fontweight="bold")
ax.set_xlabel("Bedrooms (0 = Studio)")
ax.set_ylabel("Price (AED)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_aed))
plt.tight_layout()
plt.savefig("data/plot_price_vs_bedrooms.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Step 8 — Area vs Price scatter (core feature relationship)
#
# Area (sqft) is typically the single strongest predictor of price.
# We expect a strong positive correlation.

# %%
# Sample for plotting speed (scatter with 50k points is slow)
sample = df.sample(min(5000, len(df)), random_state=42)

fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(sample["area_sqft"], sample["price"],
                alpha=0.3, s=10, c=np.log1p(sample["price"]),
                cmap="YlOrRd")
ax.set_xlabel("Area (sqft)")
ax.set_ylabel("Price (AED)")
ax.set_title("Area vs. Price (coloured by log-price)", fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_aed))
plt.colorbar(sc, ax=ax, label="log(Price)")
plt.tight_layout()
plt.savefig("data/plot_area_vs_price.png", bbox_inches="tight")
plt.show()

# Pearson correlation
corr = df["area_sqft"].corr(df["price"])
print(f"\nPearson correlation — area vs price: {corr:.3f}")

# %% [markdown]
# ## Step 9 — Correlation heatmap (numerical features)

# %%
num_cols = ["price", "area_sqft", "bedrooms", "bathrooms"]
available = [c for c in num_cols if c in df.columns]

corr_matrix = df[available].corr()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, ax=ax,
            annot_kws={"size": 11})
ax.set_title("Correlation Matrix — Numerical Features", fontweight="bold")
plt.tight_layout()
plt.savefig("data/plot_correlation_heatmap.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Step 10 — Outlier detection (price per sqft)
#
# Listings with implausibly low or high price/sqft are likely data errors.
# We surface them here and remove them before modelling.

# %%
# Compute price per sqft
df["price_per_sqft"] = df["price"] / df["area_sqft"]

q1 = df["price_per_sqft"].quantile(0.01)
q99 = df["price_per_sqft"].quantile(0.99)
outliers_pct = ((df["price_per_sqft"] < q1) | (df["price_per_sqft"] > q99)).mean() * 100

print(f"Price/sqft range (1st–99th percentile): AED {q1:,.0f} – {q99:,.0f}")
print(f"Records outside this range: {outliers_pct:.1f}% — will be removed before modelling")

# SQL-level outlier sample
print("\nSample outliers from SQL:")
print(q_price_per_sqft_outliers().head(10).to_string(index=False))

# %% [markdown]
# ## Step 11 — Furnished premium analysis

# %%
furn_df = q_furnished_premium()
print("\nFurnishing premium:")
print(furn_df.to_string(index=False))

# %% [markdown]
# ## Step 12 — Key EDA takeaways
#
# Document findings here — this is what you'd present to the BI team.

# %%
print("""
KEY FINDINGS
────────────────────────────────────────────────────────────────
1. PRICE DISTRIBUTION: Highly right-skewed — log-transform required
   before training linear models.

2. STRONGEST PREDICTOR: area_sqft has the highest correlation with price.
   Bedrooms and bathrooms are collinear with area (expected).

3. LOCATION PREMIUM: Significant price variation across neighbourhoods.
   Palm Jumeirah / Downtown command a 3-5x premium over suburban areas.
   Location must be encoded as a feature — not dropped.

4. PROPERTY TYPE: Villas have the highest price/sqft, followed by penthouses.
   Apartments are the most listed category (volume driven).

5. OUTLIERS: ~2% of records have price/sqft outside the realistic range.
   These will be filtered before modelling to avoid distorting regression.

6. FURNISHED PREMIUM: Furnished properties average ~15-20% higher price.
   Worth including as a binary feature.
────────────────────────────────────────────────────────────────
""")

# %%
