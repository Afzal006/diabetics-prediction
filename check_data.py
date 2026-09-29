import pandas as pd

# Load dataset
df = pd.read_csv("data/diabetes.csv")

print("=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("DATASET SHAPE")
print("=" * 50)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n" + "=" * 50)
print("COLUMN NAMES")
print("=" * 50)
print(df.columns.tolist())

print("\n" + "=" * 50)
print("DATA TYPES")
print("=" * 50)
print(df.dtypes)

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("DUPLICATE ROWS")
print("=" * 50)
print("Duplicates:", df.duplicated().sum())

print("\n" + "=" * 50)
print("TARGET DISTRIBUTION")
print("=" * 50)
print(df["Outcome"].value_counts())

print("\n" + "=" * 50)
print("STATISTICAL SUMMARY")
print("=" * 50)
print(df.describe())