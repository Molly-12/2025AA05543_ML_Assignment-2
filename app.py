import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Bank Marketing Classification")

st.title("Bank Marketing Classification")

st.info("The app automatically evaluates the selected model on test data. "
        "You may also download the test dataset below.")

# ---------------------------------------------------
# Load Dataset Automatically
# ---------------------------------------------------
df = pd.read_csv("data/bank-additional-full.csv", sep=";")
df["y"] = df["y"].map({"yes": 1, "no": 0})

X = df.drop("y", axis=1)
y = df["y"]

# Same split as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------------------------------
# Download Button for Examiner
# ---------------------------------------------------
st.subheader("Download Test Dataset")

test_df = X_test.copy()
test_df["y"] = y_test

csv = test_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Test CSV",
    data=csv,
    file_name="bank_test_data.csv",
    mime="text/csv"
)

# ---------------------------------------------------
# Model Selection
# ---------------------------------------------------
st.subheader("Select Model")

model_name = st.selectbox(
    "",
    ["Logistic_Regression","Decision_Tree","kNN",
     "Naive_Bayes","Random_Forest","XGBoost"]
)

# ---------------------------------------------------
# Load Model & Predict Automatically
# ---------------------------------------------------
model = joblib.load(f"model/{model_name}.pkl")

preds = model.predict(X_test)

st.subheader("Classification Report")
st.text(classification_report(y_test, preds))

st.subheader("Confusion Matrix")
st.write(confusion_matrix(y_test, preds))
