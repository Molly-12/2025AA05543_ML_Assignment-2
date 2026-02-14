Machine Learning Assignment 2 – Bank Marketing Classification

a. Problem Statement

The objective of this assignment is to build and evaluate multiple machine learning
classification models to predict whether a bank customer will subscribe to a term
deposit. The project also involves deploying the trained models using a Streamlit
web application to demonstrate model predictions and performance metrics in an
interactive manner.

b. Dataset Description

The dataset used for this assignment is the Bank Marketing Dataset obtained from
the UCI Machine Learning Repository. The dataset contains information related to
direct marketing campaigns (phone calls) of a Portuguese banking institution.

Each record represents a client and includes demographic details, campaign-related
information, and socio-economic indicators. The target variable `y` indicates whether
the client subscribed to a term deposit (`yes` or `no`).

- Type: Binary Classification  
- Number of instances: 45211  
- Number of features: 16
- Target variable: `y` (yes / no)


c. Models used and their metrics

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression      | 0.9162   | 0.9425 | 0.7095    | 0.4343 | 0.5388 | 0.5137 |
| Decision Tree            | 0.8945   | 0.7411 | 0.5311    | 0.5431 | 0.5370 | 0.4775 |
| kNN                      | 0.9035   | 0.8768 | 0.5982    | 0.4364 | 0.5047 | 0.4596 |
| Naive Bayes              | 0.8440   | 0.8493 | 0.3839    | 0.6358 | 0.4787 | 0.4108 |
| Random Forest (Ensemble) | 0.9175   | 0.9484 | 0.6824    | 0.5000 | 0.5771 | 0.5405 |
| XGBoost (Ensemble)       | 0.9175   | 0.9496 | 0.6590    | 0.5539 | 0.6019 | 0.5588 |



d. Observation of each model

| ML Model Name            | Observation about Model Performance                                                                                                                         |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Achieves high accuracy and strong AUC with good precision, but relatively lower recall, indicating that some positive cases are not detected.               |
| Decision Tree            | Provides balanced precision and recall while capturing non-linear patterns, but overall performance is lower than ensemble models.                          |
| kNN                      | Shows improved AUC compared to earlier results and moderate performance overall, but F1 score indicates sensitivity to class imbalance and feature scaling. |
| Naive Bayes              | Demonstrates high recall but comparatively low precision, meaning it identifies many positives but produces more false positives.                           |
| Random Forest (Ensemble) | Delivers strong accuracy and AUC by combining multiple trees, improving robustness and reducing overfitting.                                                |
| XGBoost (Ensemble)       | Best-performing model with highest AUC, F1 score, and MCC, indicating superior ability to capture complex patterns and handle imbalance effectively.        |


