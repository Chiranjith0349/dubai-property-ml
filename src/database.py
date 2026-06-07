"""
database.py
-----------
Loads the Dubai properties CSV into a local SQLite database and exposes
named SQL query functions used by both the EDA notebook and the ML pipeline.

All data pulls go through SQL queries rather than reading the CSV directly —
this keeps the data layer consistent and makes it easy to swap in a real
database later without touching the notebook code.
"""

import re
import sqlite3
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "properties.db"
CSV_PATH = DATA_DIR / "properties.csv"

# ── Column mapping ─────────────────────────────────────────────────────────────
# If your CSV column names differ from the ones below, update the VALUES here.
# Keys are the standardised names used throughout the project.
COLUMN_MAP = {
    "price":         "price",           # sale price in AED
    "area_sqft":     "area(sqft)",      # size in square feet
    "bedrooms":      "bedroom",         # note: singular in this dataset
    "bathrooms":     "bathroom",        # note: singular in this dataset
    "location":      "address",         # address used as neighbourhood proxy
    "property_type": "propert_type",    # note: typo in source dataset
    "furnished":     "furnishing",      # Furnished / Unfurnished / Semi-Furnished
    "purpose":       "purpose",         # for-sale / for-rent
}


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_csv_to_db(csv_path: Path = CSV_PATH, db_path: Path = DB_PATH) -> None:
    """
    Read the raw CSV, rename columns to the project standard, and write
    to SQLite. Safe to call multiple times — replaces the table each run.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}.\n"
            "Follow data/DOWNLOAD_INSTRUCTIONS.md to get the file."
        )

    df = pd.read_csv(csv_path, low_memory=False)

    # Rename only columns that exist in this CSV
    reverse_map = {v: k for k, v in COLUMN_MAP.items()}
    df = df.rename(columns=reverse_map)

    # Keep only the standard columns that are present
    standard_cols = [c for c in COLUMN_MAP.keys() if c in df.columns]
    df = df[standard_cols]

    # ── Data cleaning ────────────────────────────────────────────────────────
    # Price: "1,800,000" → 1800000.0
    if "price" in df.columns:
        df["price"] = (df["price"].astype(str)
                       .str.replace(",", "", regex=False)
                       .str.strip())
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Area: "1,208 sqft" → 1208.0
    if "area_sqft" in df.columns:
        df["area_sqft"] = (df["area_sqft"].astype(str)
                           .str.replace(",", "", regex=False)
                           .str.replace("sqft", "", regex=False)
                           .str.strip())
        df["area_sqft"] = pd.to_numeric(df["area_sqft"], errors="coerce")

    # Bedrooms: "3 beds" → 3, "Studio" → 0, "1 bed" → 1
    def parse_bedrooms(val):
        val = str(val).lower().strip()
        if "studio" in val:
            return 0
        match = re.search(r"\d+", val)
        return int(match.group()) if match else None

    if "bedrooms" in df.columns:
        df["bedrooms"] = df["bedrooms"].apply(parse_bedrooms)

    # Bathrooms: same pattern — "2 baths" → 2
    def parse_bathrooms(val):
        val = str(val).lower().strip()
        match = re.search(r"\d+", val)
        return int(match.group()) if match else None

    if "bathrooms" in df.columns:
        df["bathrooms"] = df["bathrooms"].apply(parse_bathrooms)

    conn = sqlite3.connect(db_path)
    df.to_sql("properties", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Loaded {len(df):,} records into {db_path}")
    print(f"Columns: {standard_cols}")


def get_connection() -> sqlite3.Connection:
    """Return a live connection to the SQLite database."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            "Database not found. Run load_csv_to_db() first."
        )
    return sqlite3.connect(DB_PATH)


def run_query(sql: str) -> pd.DataFrame:
    """Execute any SQL query and return a DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df


# ── Named SQL queries ──────────────────────────────────────────────────────────

def q_overview() -> pd.DataFrame:
    """High-level counts and price statistics for a quick sanity check."""
    return run_query("""
        SELECT
            COUNT(*)                        AS total_listings,
            COUNT(DISTINCT location)        AS unique_locations,
            COUNT(DISTINCT property_type)   AS property_types,
            ROUND(MIN(price), 0)            AS min_price_aed,
            ROUND(MAX(price), 0)            AS max_price_aed,
            ROUND(AVG(price), 0)            AS avg_price_aed,
            ROUND(AVG(area_sqft), 0)        AS avg_area_sqft
        FROM properties
        WHERE price > 0 AND area_sqft > 0
    """)


def q_price_by_location(top_n: int = 15) -> pd.DataFrame:
    """Average sale price per neighbourhood, top N by listing volume."""
    return run_query(f"""
        SELECT
            location,
            COUNT(*)                        AS listings,
            ROUND(AVG(price), 0)            AS avg_price_aed,
            ROUND(AVG(price / area_sqft), 0) AS avg_price_per_sqft
        FROM properties
        WHERE price > 0 AND area_sqft > 0
        GROUP BY location
        ORDER BY listings DESC
        LIMIT {top_n}
    """)


def q_price_by_type() -> pd.DataFrame:
    """Average price and size breakdown by property type."""
    return run_query("""
        SELECT
            property_type,
            COUNT(*)                        AS listings,
            ROUND(AVG(price), 0)            AS avg_price_aed,
            ROUND(AVG(area_sqft), 0)        AS avg_area_sqft,
            ROUND(AVG(price / area_sqft), 0) AS avg_price_per_sqft
        FROM properties
        WHERE price > 0 AND area_sqft > 0
        GROUP BY property_type
        ORDER BY avg_price_aed DESC
    """)


def q_price_by_bedrooms() -> pd.DataFrame:
    """How price scales with bedroom count — key feature insight."""
    return run_query("""
        SELECT
            bedrooms,
            COUNT(*)                AS listings,
            ROUND(AVG(price), 0)    AS avg_price_aed,
            ROUND(MIN(price), 0)    AS min_price_aed,
            ROUND(MAX(price), 0)    AS max_price_aed
        FROM properties
        WHERE price > 0 AND bedrooms IS NOT NULL
        GROUP BY bedrooms
        ORDER BY bedrooms
    """)


def q_furnished_premium() -> pd.DataFrame:
    """Price premium for furnished vs unfurnished properties."""
    return run_query("""
        SELECT
            furnished,
            COUNT(*)                AS listings,
            ROUND(AVG(price), 0)    AS avg_price_aed
        FROM properties
        WHERE price > 0 AND furnished IS NOT NULL
        GROUP BY furnished
        ORDER BY avg_price_aed DESC
    """)


def q_price_per_sqft_outliers() -> pd.DataFrame:
    """
    Listings where price/sqft is implausibly low or high — likely data errors.
    Used during EDA to decide the outlier filter thresholds before modelling.
    """
    return run_query("""
        SELECT
            location,
            property_type,
            bedrooms,
            area_sqft,
            price,
            ROUND(price / area_sqft, 0) AS price_per_sqft
        FROM properties
        WHERE price > 0 AND area_sqft > 0
          AND (price / area_sqft < 200 OR price / area_sqft > 10000)
        ORDER BY price_per_sqft
        LIMIT 20
    """)


def get_ml_dataset() -> pd.DataFrame:
    """
    Return a clean DataFrame ready for the ML pipeline.
    Filters applied here match the business logic — we only model
    for-sale properties with valid prices and sizes.
    """
    sql = """
        SELECT
            price,
            area_sqft,
            bedrooms,
            bathrooms,
            location,
            property_type,
            furnished
        FROM properties
        WHERE
            price > 100000              -- minimum realistic sale price (AED)
            AND price < 100000000       -- cap at 100M to remove data errors
            AND area_sqft > 100
            AND area_sqft < 50000
            AND bedrooms IS NOT NULL
    """
    # Filter to for-sale only if the purpose column exists
    conn = get_connection()
    cols = pd.read_sql_query("SELECT * FROM properties LIMIT 1", conn).columns
    conn.close()

    if "purpose" in cols:
        sql += " AND (purpose LIKE '%sale%' OR purpose LIKE '%buy%')"

    return run_query(sql)


if __name__ == "__main__":
    # Quick smoke test — run `python src/database.py` after downloading data
    load_csv_to_db()
    print("\n--- Overview ---")
    print(q_overview().to_string())
    print("\n--- Top locations ---")
    print(q_price_by_location(5).to_string())
