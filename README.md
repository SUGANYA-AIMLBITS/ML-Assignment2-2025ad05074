# M.Tech (AIML/DSE) - Machine Learning Assignment 2

## a. Problem Statement
The objective of this assignment is to implement, evaluate, and deploy an end-to-end multi-model classification pipeline. This involves training five distinct machine learning classifiers on a benchmark dataset, computing comprehensive evaluation metrics, and deploying an interactive Streamlit application to demonstrate real-time model inference and performance comparison.

## b. Dataset Description
* **Dataset Used:** Breast Cancer Wisconsin (Diagnostic) Dataset (sourced via scikit-learn / UCI repository).
* **Problem Type:** Binary Classification (Malignant vs. Benign).
* **Number of Instances:** 569 (Satisfies the $\ge$ 500 requirement).
* **Number of Features:** 30 numeric features computed from digitized images (Satisfies the $\ge$ 12 requirement).

## c. GitHub Repository Link
* **Repository URL:** 

## d. Models Used & Performance Comparison

### Evaluation Metrics Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9737 | 0.9951 | 0.9722 | 0.9859 | 0.9790 | 0.9429 |
| **Decision Tree** | 0.9298 | 0.9218 | 0.9420 | 0.9437 | 0.9429 | 0.8443 |
| **kNN** | 0.9561 | 0.9880 | 0.9583 | 0.9718 | 0.9650 | 0.9037 |
| **Naive Bayes** | 0.9386 | 0.9802 | 0.9333 | 0.9718 | 0.9522 | 0.8601 |
| **Random Forest (Ensemble)** | 0.9649 | 0.9930 | 0.9589 | 0.9859 | 0.9722 | 0.9231 |

### Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed exceptionally well on the scaled features, yielding high accuracy and the highest MCC score, demonstrating linear separability in the dataset. |
| **Decision Tree** | Exhibited lower overall accuracy and MCC compared to ensemble and linear models, likely due to variance and minor overfitting on training splits. |
| **kNN** | Delivered robust results after feature scaling, capturing local neighborhood structures effectively with strong precision and recall. |
| **Naive Bayes** | Showed high recall but slightly lower precision, indicating a higher rate of false positives under the Gaussian independence assumption. |
| **Random Forest (Ensemble)** | Achieved outstanding performance by combining multiple trees, yielding high recall and robust generalization across unseen test samples. |
| **Overall Winner for Dataset?** | **Logistic Regression** (closely followed by Random Forest), balancing superior accuracy, high AUC, and a top Matthews Correlation Coefficient. |
