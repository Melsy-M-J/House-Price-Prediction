
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

# ==========================================
# STEP 2: DATA CLEANING
# ==========================================

# Check duplicate rows
print("\n========== DUPLICATE RECORDS ==========")

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)

# Check invalid prices
print("\n========== INVALID PRICE RECORDS ==========")

invalid_prices = df[df["price"] <= 0]

print("Invalid price records:", len(invalid_prices))

# Display a few invalid records
print(invalid_prices[["price", "city", "statezip", "sqft_living"]].head(10))


# Remove records with zero or negative prices
df_clean = df[df["price"] > 0].copy()

print("\n========== AFTER REMOVING INVALID PRICES ==========")

print("Original number of rows:", len(df))
print("Rows after removing invalid prices:", len(df_clean))

# Remove duplicate records
df_clean = df_clean.drop_duplicates()

print("\n========== FINAL CLEANED DATASET ==========")

print("Rows after cleaning:", len(df_clean))
print("Columns:", len(df_clean.columns))

# Check whether invalid prices remain
print("Invalid prices remaining:", (df_clean["price"] <= 0).sum())

# Check whether duplicates remain
print("Duplicate rows remaining:", df_clean.duplicated().sum())

# Save the cleaned dataset
df_clean.to_csv("results/cleaned_house_data.csv", index=False)

print("\nCleaned dataset saved successfully!")
