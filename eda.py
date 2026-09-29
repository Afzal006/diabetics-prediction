import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/diabetes.csv")

print("Dataset loaded successfully!")

# --------------------------------------------------
# 1. Check zero values
# --------------------------------------------------

print("\nZero values in each column:")
print((df == 0).sum())

# --------------------------------------------------
# 2. Check target distribution
# --------------------------------------------------

print("\nTarget distribution:")
print(df["Outcome"].value_counts())

# --------------------------------------------------
# 3. Target distribution plot
# --------------------------------------------------

plt.figure(figsize=(6, 4))

sns.countplot(x="Outcome", data=df)

plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Number of Patients")

plt.show()

# --------------------------------------------------
# 4. Correlation heatmap
# --------------------------------------------------

plt.figure(figsize=(10, 7))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.show()

# --------------------------------------------------
# 5. Glucose distribution
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Glucose",
    hue="Outcome",
    kde=True
)

plt.title("Glucose Distribution by Diabetes Outcome")
plt.xlabel("Glucose")
plt.ylabel("Count")

plt.show()

# --------------------------------------------------
# 6. BMI distribution
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="BMI",
    hue="Outcome",
    kde=True
)

plt.title("BMI Distribution by Diabetes Outcome")
plt.xlabel("BMI")
plt.ylabel("Count")

plt.show()