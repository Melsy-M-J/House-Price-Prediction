
# ==========================================
# HOUSE PRICE PREDICTION
# Step 1: Loading and Understanding the Data
# ==========================================

# Import the Pandas library
import pandas as pd

# To split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

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


# ==========================================
# STEP 3: SELECT INPUT FEATURES AND TARGET
# ==========================================

# Select the input columns that the model will use
feature_columns = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "city",
    "statezip"
]

# Select the target column
target_column = "price"

# Create X (input features)
X = df_clean[feature_columns].copy()

# Create y (target variable)
y = df_clean[target_column].copy()


# Separate numerical and categorical features
categorical_features = ["city", "statezip"]

numeric_features = [
    column for column in feature_columns
    if column not in categorical_features
]


# Display the results
print("\n========== FEATURE SELECTION ==========")

print("Input features (X):")
print(feature_columns)

print("\nTarget variable (y):")
print(target_column)

print("\n========== FEATURE TYPES ==========")

print("Numerical features:", numeric_features)
print("Categorical features:", categorical_features)

print("\n========== INPUT DATA ==========")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst five input records:")
print(X.head())

print("\nFirst five target values:")
print(y.head())


# ==========================================
# STEP 4: SPLITTING THE DATASET
# ==========================================

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Display the sizes of the datasets
print("\n========== DATASET SPLITTING ==========")

print("Original input data:", X.shape)
print("Original target data:", y.shape)

print("\nTraining input data:", X_train.shape)
print("Testing input data:", X_test.shape)

print("\nTraining target data:", y_train.shape)
print("Testing target data:", y_test.shape)
