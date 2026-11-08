import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Set page layout and aesthetics
st.set_page_config(
    page_title="Wisconsin Breast Cancer Diagnostic Classifier",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Breast Cancer Diagnostic Classifier Benchmark")
st.markdown(
    "Interactive evaluation interface for supervised machine learning models pre-trained "
    "on the **Breast Cancer Wisconsin (Diagnostic)** dataset."
)
st.markdown("---")

# 1. Sidebar: Model Selection
st.sidebar.header("⚙️ Model Configuration")

model_mapping = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib"
}

selected_model_name = st.sidebar.selectbox(
    "Choose a Trained Classifier:",
    list(model_mapping.keys())
)

# Cache artifacts to load them only once into memory
@st.cache_resource
def load_model_artifacts(model_path):
    clf = joblib.load(model_path)
    scaler = joblib.load("model/scaler.joblib")
    features = joblib.load("model/feature_columns.joblib")
    return clf, scaler, features

current_classifier, standard_scaler, required_features = load_model_artifacts(
    model_mapping[selected_model_name]
)

# 2. Sidebar: Test Dataset Upload
st.sidebar.header("📁 Data Source")
user_csv_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

TARGET_COLUMN = "diagnosis"

# Load uploaded data or fallback to local test_data.csv
if user_csv_file is not None:
    evaluation_df = pd.read_csv(user_csv_file)
    st.sidebar.success("Custom test CSV loaded successfully!")
else:
    if os.path.exists("test_data.csv"):
        evaluation_df = pd.read_csv("test_data.csv")
        st.info("ℹ️ Using default repository `test_data.csv`. Upload your own CSV in the sidebar to test custom data.")
    else:
        st.error("Error: `test_data.csv` was not found locally. Please upload a test dataset.")
        st.stop()

# 3. Input Validation
if TARGET_COLUMN not in evaluation_df.columns:
    st.error(f"Uploaded file must contain the target column: `{TARGET_COLUMN}`")
    st.stop()

missing_columns = [col for col in required_features if col not in evaluation_df.columns]
if missing_columns:
    st.error(f"The uploaded dataset is missing required features: {missing_columns}")
    st.stop()

# 4. Preprocessing & Predictions (Zero Retraining)
X_test_raw = evaluation_df[required_features]
y_test_true = evaluation_df[TARGET_COLUMN]

# Scale features using the fitted training scaler
X_test_transformed = standard_scaler.transform(X_test_raw)

# Generate predictions
predicted_labels = current_classifier.predict(X_test_transformed)

# Obtain prediction probabilities for AUC calculation
if hasattr(current_classifier, "predict_proba"):
    prediction_probabilities = current_classifier.predict_proba(X_test_transformed)[:, 1]
else:
    prediction_probabilities = predicted_labels

# 5. Compute Metrics
acc = accuracy_score(y_test_true, predicted_labels)
auc = roc_auc_score(y_test_true, prediction_probabilities)
prec = precision_score(y_test_true, predicted_labels)
rec = recall_score(y_test_true, predicted_labels)
f1 = f1_score(y_test_true, predicted_labels)
mcc = matthews_corrcoef(y_test_true, predicted_labels)

# 6. Metric Cards Display
st.subheader(f"📊 Evaluation Metrics: {selected_model_name}")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{acc:.4f}")
col2.metric("AUC", f"{auc:.4f}")
col3.metric("Precision", f"{prec:.4f}")
col4.metric("Recall", f"{rec:.4f}")
col5.metric("F1 Score", f"{f1:.4f}")
col6.metric("MCC", f"{mcc:.4f}")

st.markdown("---")

# 7. Diagnostic Visualizations
diag_col1, diag_col2 = st.columns([1, 1])

with diag_col1:
    st.subheader("Confusion Matrix")
    conf_mat = confusion_matrix(y_test_true, predicted_labels)
    
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax_cm,
        xticklabels=["Benign (0)", "Malignant (1)"],
        yticklabels=["Benign (0)", "Malignant (1)"]
    )
    ax_cm.set_xlabel("Predicted Label", fontsize=9, fontweight="bold")
    ax_cm.set_ylabel("Actual Label", fontsize=9, fontweight="bold")
    st.pyplot(fig_cm)

with diag_col2:
    st.subheader("Classification Report")
    report_dict = classification_report(
        y_test_true, 
        predicted_labels, 
        target_names=["Benign (0)", "Malignant (1)"], 
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.format(precision=4), use_container_width=True)

# 8. Data Inspection Accordion
with st.expander("🔍 Inspect Uploaded Test Data"):
    st.dataframe(evaluation_df.head(20), use_container_width=True)