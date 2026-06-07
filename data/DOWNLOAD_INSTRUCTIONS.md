# Dataset Download Instructions

## Dataset: Dubai Real Estate Listings

This project uses a public Dubai property listings dataset available on Kaggle.

### Steps

1. Go to Kaggle and search: **"dubai real estate listings"** or **"dubai properties"**
   - One option: https://www.kaggle.com/datasets/azharsaleem/real-estate-gold-mine-in-uae-bayut-com

2. Download the CSV file

3. Rename it to **`properties.csv`** and place it in this `/data/` folder

4. Open `src/database.py` and check that `COLUMN_MAP` matches your CSV's column names
   (the comments in the file explain what each key maps to)

### Expected columns (adjust if yours differ)

The code expects these concepts — column names vary by dataset:

| Concept | Common column names |
|---|---|
| Price (AED) | `price`, `Price`, `sale_price` |
| Area (sqft) | `area_in_sqft`, `area(sqft)`, `size` |
| Bedrooms | `bedrooms`, `bedroom`, `beds` |
| Bathrooms | `bathrooms`, `bathroom`, `baths` |
| Location | `location`, `address`, `neighborhood` |
| Property type | `property_type`, `propert_type`, `type` |
| Furnished | `furnishing`, `furnished` |
| Purpose | `purpose`, `rent_or_buy` |

### Minimum viable dataset
You need at least: **price, area_sqft, bedrooms, location, property_type**
Everything else enriches the model but isn't required.
