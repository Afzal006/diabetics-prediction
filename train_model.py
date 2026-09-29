import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ==================================================
# 1. LOAD DATASET
# ==================================================

df = pd.read_csv("data/diabetes.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==================================================
# 2. REPLACE INVALID ZERO VALUES
# ==================================================

columns_with_invalid_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

df[columns_with_invalid_zero] = df[
    columns_with_invalid_zero
].replace(0, np.nan)


# ==================================================
# 3. SEPARATE FEATURES AND TARGET
# ==================================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# ==================================================
# 4. TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==================================================
# 5. CREATE MODELS
# ==================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}


# ==================================================
# 6. TRAIN AND EVALUATE
# ==================================================

results = {}

for name, model in models.items():

    pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            model
        )
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Probability
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    results[name] = {
        "pipeline": pipeline,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# ==================================================
# 7. MODEL COMPARISON
# ==================================================

print("\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

print(
    f"{'Model':<22}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'ROC-AUC':<12}"
)

print("-" * 75)

for name, result in results.items():

    print(
        f"{name:<22}"
        f"{result['accuracy']:<12.4f}"
        f"{result['precision']:<12.4f}"
        f"{result['recall']:<12.4f}"
        f"{result['f1']:<12.4f}"
        f"{result['roc_auc']:<12.4f}"
    )


# ==================================================
# 8. SELECT MODEL USING F1 SCORE
# ==================================================

best_model_name = max(
    results,
    key=lambda name: results[name]["f1"]
)

best_pipeline = results[best_model_name]["pipeline"]


print("\n" + "=" * 60)
print("SELECTED MODEL")
print("=" * 60)

print("Model:", best_model_name)
print("F1 Score:", f"{results[best_model_name]['f1']:.4f}")


# ==================================================
# 9. CREATE MODEL FOLDER
# ==================================================

os.makedirs("model", exist_ok=True)


# ==================================================
# 10. SAVE COMPLETE PIPELINE
# ==================================================

model_path = "model/diabetes_model.pkl"

joblib.dump(
    best_pipeline,
    model_path
)

print("\nModel saved successfully!")
print("Location:", model_path)


# ==================================================
# 11. SAVE MODEL INFORMATION
# ==================================================

model_info = {
    "model_name": best_model_name,
    "accuracy": results[best_model_name]["accuracy"],
    "precision": results[best_model_name]["precision"],
    "recall": results[best_model_name]["recall"],
    "f1": results[best_model_name]["f1"],
    "roc_auc": results[best_model_name]["roc_auc"]
}

joblib.dump(
    model_info,
    "model/model_info.pkl"
)

print("Model information saved successfully!")
print("\nTraining completed!")