# imports
import pandas as pd
import numpy as np
from pathlib import Path

# to reach local files in the data folder
DATA_DIR = Path(__file__).parent.parent / "data"

# function to load sales data; isolates only relevant columns
def load_sales_data():
    df = pd.read_csv(DATA_DIR / "sales.csv", encoding="latin-1")
    df = df[['Order ID', 'Order Date', 'Region', 'City', 'State', 'Segment', 'Category', 'Sub-Category', 'Product Name', 'Sales', 'Quantity', 'Discount', 'Profit']]
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    return df

# ['Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Sales', 'Quantity', 'Discount', 'Profit']
# sales trends—product Name, sales, qty, discount, category, geo data
# regional analysis—all geo data
# product categories—category, sub-category, sales 
# customer segments—segment, geo data
# fairness checks—not sure