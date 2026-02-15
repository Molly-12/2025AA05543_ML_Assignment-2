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
# Download Button (Examiner Friendly)
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
# Upload Option (Optional Override)
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
st.subheader("Classification Report")
st.text(classification_report(y_true, preds))

st.subheader("Confusion Matrix")
st.write(confusion_matrix(y_true, preds))
