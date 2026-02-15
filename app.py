import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Bank Marketing Classification")

st.title("Bank Marketing Classification")

# ---------------------------------------------------------
# Load Full Dataset Automatically (from repo)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/bank-additional-full.csv", sep=";")
    df["y"] = df["y"].map({"yes": 1, "no": 0})
    return df

df = load_data()

# ---------------------------------------------------------
# Create Default Test Split
# ---------------------------------------------------------
X = df.drop("y", axis=1)
y = df["y"]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

default_test_df = X_test.copy()
default_test_df["y"] = y_test

# ---------------------------------------------------------
# Download Button 
# ---------------------------------------------------------
st.subheader("Download Test Dataset")

csv = default_test_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Default Test CSV",
    data=csv,
    file_name="bank_test_data.csv",
    mime="text/csv"
)

# ---------------------------------------------------------
# Upload Option 
# ---------------------------------------------------------
st.subheader("Upload Test CSV (Optional)")

uploaded_file = st.file_uploader("Upload CSV with y column", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.success("Uploaded file loaded successfully!")
else:
    test_df = default_test_df
    st.info("Using default test dataset.")

# ---------------------------------------------------------
# Model Selection
# ---------------------------------------------------------
st.subheader("Select Model")

model_name = st.selectbox(
    "",
    ["Logistic_Regression",
     "Decision_Tree",
     "kNN",
     "Naive_Bayes",
     "Random_Forest",
     "XGBoost"]
)

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
model = joblib.load(f"model/{model_name}.pkl")

X_input = test_df.drop("y", axis=1)
y_true = test_df["y"]

preds = model.predict(X_input)

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

# Calculate metrics
accuracy = accuracy_score(y_true, preds)
precision = precision_score(y_true, preds)
recall = recall_score(y_true, preds)
f1 = f1_score(y_true, preds)
mcc = matthews_corrcoef(y_true, preds)

# AUC (only if model supports probability)
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X_input)[:, 1]
    auc = roc_auc_score(y_true, probs)
else:
    auc = None

# ---------------------------------------------------------
# Display Evaluation Metrics 
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Evaluation Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", round(accuracy, 3))
col2.metric("Precision", round(precision, 3))
col3.metric("Recall", round(recall, 3))

col4, col5, col6 = st.columns(3)
col4.metric("F1 Score", round(f1, 3))
col5.metric("MCC", round(mcc, 3))

if auc is not None:
    col6.metric("AUC", round(auc, 3))
else:
    col6.metric("AUC", "N/A")

# ---------------------------------------------------------
# Classification Report
# ---------------------------------------------------------
st.subheader("📋 Classification Report")

report_dict = classification_report(y_true, preds, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(3)

st.dataframe(report_df, use_container_width=True)

# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------
st.subheader("📌 Confusion Matrix")

cm = confusion_matrix(y_true, preds)

cm_df = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

st.dataframe(cm_df, use_container_width=True)
