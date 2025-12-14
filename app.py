import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import gdown
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")
st.markdown("""
This app compares **Logistic Regression** and **Random Forest** models  
for detecting **credit card fraud**.

The dataset is downloaded **securely at runtime** to keep the repository lightweight.
""")

st.divider()

# --------------------------------------------------
# Load models
# --------------------------------------------------
@st.cache_resource
def load_models():
    lr_model = joblib.load("lr_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return lr_model, rf_model, scaler

lr_model, rf_model, scaler = load_models()

# --------------------------------------------------
# Download & load dataset from Google Drive
# --------------------------------------------------
@st.cache_data
def load_dataset():
    FILE_ID = "1Lrg8l73vJcVFSkdbAJsgQceJlM1wW1Xn"
    CSV_PATH = "creditcard_runtime.csv"

    if not os.path.exists(CSV_PATH):
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            CSV_PATH,
            quiet=False
        )

    df = pd.read_csv(CSV_PATH)

    expected_columns = [
        'Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13',
        'V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25',
        'V26','V27','V28','Amount','Class'
    ]

    if df.shape[1] != 31:
        st.error(f"Invalid dataset format: found {df.shape[1]} columns, expected 31.")
        st.stop()

    df.columns = expected_columns
    return df

df = load_dataset()
fraud_samples = df[df["Class"] == 1].reset_index(drop=True)

st.success(f"Dataset loaded successfully: {df.shape}")

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("🧾 Transaction Input")

model_choice = st.sidebar.radio(
    "Choose Model",
    ["Logistic Regression", "Random Forest"]
)

use_example = st.sidebar.checkbox("Use Example Fraud Transaction")

if use_example:
    idx = st.sidebar.selectbox(
        "Select Fraud Example",
        fraud_samples.index
    )
    row = fraud_samples.loc[idx]
    time = row["Time"]
    V_values = [row[f"V{i}"] for i in range(1, 29)]
    amount = row["Amount"]
else:
    time = st.sidebar.number_input(
        "Seconds Since First Transaction",
        min_value=0.0,
        value=100000.0
    )

    st.sidebar.subheader("🔐 Transaction Pattern Components (V1–V28)")
    st.sidebar.caption(
        "These sliders represent anonymized PCA-based transaction features."
    )

    V_values = [
        st.sidebar.slider(f"V{i}", -10.0, 10.0, 0.0)
        for i in range(1, 29)
    ]

    amount = st.sidebar.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0
    )

# --------------------------------------------------
# Prepare input
# --------------------------------------------------
amount_scaled = scaler.transform([[amount]])[0][0]
X = np.array([time] + V_values + [amount_scaled]).reshape(1, -1)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Fraud"):
    lr_pred = lr_model.predict(X)[0]
    lr_prob = lr_model.predict_proba(X)[0][1]

    rf_pred = rf_model.predict(X)[0]
    rf_prob = rf_model.predict_proba(X)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Logistic Regression")
        st.metric(
            "Prediction",
            "Fraud 🚨" if lr_pred == 1 else "Legitimate ✅"
        )
        st.metric("Fraud Probability", f"{lr_prob:.2%}")

    with col2:
        st.subheader("Random Forest")
        st.metric(
            "Prediction",
            "Fraud 🚨" if rf_pred == 1 else "Legitimate ✅"
        )
        st.metric("Fraud Probability", f"{rf_prob:.2%}")

    st.divider()

    # --------------------------------------------------
    # Visualizations
    # --------------------------------------------------
    st.subheader("📊 Model Insight")

    fig, ax = plt.subplots()
    ax.bar(
        ["Logistic Regression", "Random Forest"],
        [lr_prob, rf_prob]
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraud Probability")
    ax.set_title("Model Probability Comparison")

    st.pyplot(fig)

    st.info(
        "Logistic Regression reacts strongly to extreme values, "
        "while Random Forest is more conservative and relies on learned patterns."
    )
