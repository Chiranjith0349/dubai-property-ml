# Dataset Download Instructions

## Dataset: Dubai Properties (Bayut Listings)

This project uses a Dubai real estate listings dataset scraped from Bayut.com — 
the exact platform you're applying to. Using their own data is a strong talking point.

### Steps

1. Go to Kaggle and search: **"dubai properties bayut"**
   - Direct link option 1: https://www.kaggle.com/datasets/azharsaleem/real-estate-gold-mine-in-uae-bayut-com
   - Direct link option 2: Search "dubai real estate listings" and pick the largest dataset

2. Download the CSV file(s)

3. Rename the main file to **`properties.csv`** and place it in this `/data/` folder

4. Open `src/database.py` and update `COLUMN_MAP` at the top to match your 
   CSV's actual column names (instructions are in the file)

### Expected columns (adjust if yours differ)
The code expects these concepts — column names may vary by dataset:

| Concept | Common column names |
|---|---|
| Price (AED) | `price`, `Price`, `sale_price` |
| Area (sqft) | `area_in_sqft`, `area`, `size` |
| Bedrooms | `bedrooms`, `beds`, `Bedrooms` |
| Bathrooms | `bathrooms`, `baths` |
| Location/Area | `location`, `neighborhood`, `area_name` |
| Property type | `type`, `property_type`, `Type` |
| Furnished | `furnishing`, `furnished` |
| Purpose | `purpose`, `rent_or_buy` |

### Minimum viable dataset
You need at least: **price, area_sqft, bedrooms, location, property_type**
Everything else enriches the model but isn't required.
