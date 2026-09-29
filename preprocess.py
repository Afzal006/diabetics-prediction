import pandas as pd

# Load dataset
df = pd.read_csv("data/diabetes.csv")

print("Original dataset shape:", df.shape)

# Columns where 0 represents an invalid/missing measurement
columns_to_replace = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Replace 0 with NaN
df[columns_to_replace] = df[columns_to_replace].replace(0, pd.NA)

print("\nMissing values after replacing invalid zeros:")
print(df.isnull().sum())

# Fill missing values with median
for column in columns_to_replace:
    df[column] = df[column].fillna(df[column].median())

print("\nMissing values after median imputation:")
print(df.isnull().sum())

print("\nCleaned dataset:")
print(df.head())