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

| ML Model Name                | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ---------------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression          | 0.9164   | 0.9424 | 0.7100    | 0.4353 | 0.5397 | 0.5147 |
| Decision Tree                | 0.8945   | 0.7411 | 0.5311    | 0.5431 | 0.5370 | 0.4775 |
| kNN                          | 0.9022   | 0.8322 | 0.6178    | 0.3448 | 0.4426 | 0.4138 |
| Naive Bayes                  | 0.7527   | 0.8493 | 0.2902    | 0.8265 | 0.4296 | 0.3860 |
| Random Forest (Ensemble)     | 0.9175   | 0.9484 | 0.6824    | 0.5000 | 0.5771 | 0.5405 |
| XGBoost (Ensemble)           | 0.9175   | 0.9496 | 0.6590    | 0.5539 | 0.6019 | 0.5588 |


d. Observation of each model

| ML Model Name            | Observation about Model Performance                                                                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Achieves high accuracy and AUC with good precision, but has moderate recall, indicating that some positive cases are missed.                            |
| Decision Tree            | Models non-linear relationships effectively with balanced precision and recall, but shows lower generalization performance compared to ensemble models. |
| kNN                      | Provides reasonable accuracy, but lower recall and F1 score indicate sensitivity to class imbalance and the choice of k value.                          |
| Naive Bayes              | Exhibits very high recall with low precision, meaning it detects most positive cases but produces more false positives.                                 |
| Random Forest (Ensemble) | Achieves high accuracy and AUC by combining multiple trees, reducing overfitting and improving robustness.                                              |
| XGBoost (Ensemble)       | Best-performing model with the highest AUC, F1 score, and MCC, indicating excellent handling of complex patterns and class imbalance.                   |


