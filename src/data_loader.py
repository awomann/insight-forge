"""
data_loader.py
--------------
Loads and prepares the Superstore dataset for InsightForge.
Remaps original column names to internal standard names on load.
"""

# IMPORTS & CONFIG

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


# Maps Superstore column names → internal standard names
SUPERSTORE_COLUMN_MAP = {
    "Order ID":      "order_id",
    "Order Date":    "order_date",
    "Region":        "region",
    "City":          "city",
    "State":         "state",
    "Segment":       "customer_segment",
    "Category":      "product_category",
    "Sub-Category":  "product_name",
    "Product Name":  "product_full_name",
    "Sales":         "sales",
    "Quantity":      "quantity",
    "Discount":      "discount",
    "Profit":        "profit",
}

def load_sales_data() -> pd.DataFrame:
    """Load sales.csv and remap column names to internal standard."""
    df = pd.read_csv(DATA_DIR / "sales.csv", encoding="latin-1")
    rename = {k: v for k, v in SUPERSTORE_COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["sales", "order_date"])
    return df