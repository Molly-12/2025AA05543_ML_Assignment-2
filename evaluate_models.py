import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
df = pd.read_csv("data/bank-additional-full.csv", sep=";")

df["y"] = df["y"].map({"yes": 1, "no": 0})

X = df.drop("y", axis=1)
y = df["y"]

# ---------------------------------------------------
# Train-test split (NO preprocessing here!)
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = [
    "Logistic_Regression",
    "Decision_Tree",
    "kNN",
    "Naive_Bayes",
    "Random_Forest",
    "XGBoost"
]

# ---------------------------------------------------
# Evaluate models
# ---------------------------------------------------
for m in models:

    model = joblib.load(f"model/{m}.pkl")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(m)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_prob))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("MCC:", matthews_corrcoef(y_test, y_pred))
    print("-" * 40)
