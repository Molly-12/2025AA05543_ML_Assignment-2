import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report

st.title("Bank Marketing Classification")

model_name = st.selectbox(
    "Select Model",
    ["Logistic_Regression","Decision_Tree","kNN","Naive_Bayes","Random_Forest","XGBoost"]
)

uploaded_file = st.file_uploader("Upload Test CSV (with y column)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file,  sep=";")
   y_true = data["y"].map({"yes": 1, "no": 0})
    X = data.drop("y", axis=1)

    model = joblib.load(f"model/{model_name}.pkl")
    preds = model.predict(X)

    st.text("Classification Report")
    st.text(classification_report(y_true, preds))

    st.text("Confusion Matrix")
    st.write(confusion_matrix(y_true, preds))


