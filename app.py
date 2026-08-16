import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Assignment 2 - Classifier Demo", layout="wide")

st.title("🔬 M.Tech ML Assignment 2: Classification Models Demo")
st.markdown("**Student Deployment App** built for BITS Virtual Lab evaluation.")

# Sidebar for controls
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])

@st.cache_resource
def load_artifacts():
    scaler = joblib.load('model/scaler.pkl')
    models = {
        "Logistic Regression": joblib.load('model/logistic_regression.pkl'),
        "Decision Tree": joblib.load('model/decision_tree.pkl'),
        "kNN": joblib.load('model/knn.pkl'),
        "Naive Bayes": joblib.load('model/naive_bayes.pkl'),
        "Random Forest": joblib.load('model/random_forest.pkl')
    }
    return scaler, models

try:
    scaler, models = load_artifacts()
except Exception as e:
    st.error("Models not found! Please run `train_models.py` first to generate saved models.")
    st.stop()

# Load default test data if none uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    if os.path.exists('test_data.csv'):
        df = pd.read_csv('test_data.csv')
    else:
        st.warning("Please upload a test CSV file or ensure `test_data.csv` is in the directory.")
        st.stop()

st.subheader("📊 Test Dataset Preview")
st.dataframe(df.head())

if 'target' in df.columns:
    X_input = df.drop(columns=['target'])
    y_true = df['target']
else:
    st.error("Uploaded CSV must contain a 'target' column for evaluation.")
    st.stop()

# Model selection dropdown
selected_model_name = st.sidebar.selectbox("Select Classification Model", list(models.keys()))
model = models[selected_model_name]

# Prediction logic
if selected_model_name in ["Logistic Regression", "kNN"]:
    X_scaled = scaler.transform(X_input)
    y_pred = model.predict(X_scaled)
else:
    y_pred = model.predict(X_input)

# Display results
st.markdown(f"### Evaluation Metrics for: **{selected_model_name}**")

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

y_prob = model.predict_proba(scaler.transform(X_input) if selected_model_name in ["Logistic Regression", "kNN"] else X_input)[:, 1]

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
col2.metric("AUC Score", f"{roc_auc_score(y_true, y_prob):.4f}")
col3.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")

col4, col5, col6 = st.columns(3)
col4.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")
col5.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")
col6.metric("MCC Score", f"{matthews_corrcoef(y_true, y_pred):.4f}")

st.markdown("---")
st.subheader("📉 Confusion Matrix")
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
st.pyplot(fig)