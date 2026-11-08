
# ==========================================
# HOUSE PRICE PREDICTION
# Step 1: Loading and Understanding the Data
# ==========================================

# Import the Pandas library
import pandas as pd

# Location of our dataset
DATA_PATH = "data/modified_data.csv"

# Read the CSV file
df = pd.read_csv(DATA_PATH)

# Display the first five rows
print("\n========== FIRST FIVE ROWS ==========")
print(df.head())

# Display the number of rows and columns
print("\n========== DATASET SHAPE ==========")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# Display all column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# Display information about the dataset
print("\n========== DATASET INFORMATION ==========")
df.info()

# Check for missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Display statistical information
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# Display statistics about house prices
print("\n========== PRICE SUMMARY ==========")
print(df["price"].describe())

# Count invalid prices (zero or negative)
print("\n========== INVALID PRICES ==========")
print("Zero or negative prices:", (df["price"] <= 0).sum())
