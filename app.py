import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# Load models
# ------------------------
lr = joblib.load('lr_model.pkl')
rf = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# ------------------------
# Load dataset from Google Drive
# ------------------------
csv_url = "https://drive.usercontent.google.com/download?id=1b-LDt5p10Q-GD1aRl0rfYX6vcleQqoWg&export=download&authuser=0"
df = pd.read_csv(csv_url)

expected_columns = ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13',
                    'V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28',
                    'Amount','Class']
df.columns = expected_columns

fraud_samples = df[df['Class'] == 1].reset_index(drop=True)

# ------------------------
# Page config
# ------------------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection System")
st.markdown("""
This interactive app predicts whether a credit card transaction is **fraudulent** or **legitimate**.
You can compare **Logistic Regression** and **Random Forest** models.
Use example fraud transactions or simulate your own.
""")

st.divider()

# ------------------------
# Model selection
# ------------------------
model_name = st.selectbox("Select Model for Prediction", ["Logistic Regression", "Random Forest"])
model = lr if model_name == "Logistic Regression" else rf

# ------------------------
# Sidebar inputs
# ------------------------
st.sidebar.header("🧾 Transaction Input")
st.sidebar.markdown("Adjust values or select an example fraud transaction.")

use_example = st.sidebar.checkbox("Use Example Fraud Transaction")
if use_example:
    idx = st.sidebar.selectbox("Select Example Fraud Transaction", fraud_samples.index)
    row = fraud_samples.loc[idx]
    time = row['Time']
    V_features = row[['V'+str(i) for i in range(1,29)]].tolist()
    amount = row['Amount']
else:
    time = st.sidebar.number_input("Seconds since first transaction", min_value=0.0, value=100000.0)
    
    st.sidebar.subheader("🔐 Anonymized Transaction Patterns (V1-V28)")
    st.sidebar.caption("These sliders represent PCA components of transaction patterns.")
    V_features = []
    for i in range(1,29):
        V_features.append(st.sidebar.slider(f"Pattern V{i}", -10.0, 10.0, 0.0))
    
    amount = st.sidebar.number_input("Amount (in currency units)", min_value=0.0, value=100.0)

# ------------------------
# Scale features
# ------------------------
amount_scaled = scaler.transform([[amount]])[0][0]
X = np.array([time] + V_features + [amount_scaled]).reshape(1, -1)

# ------------------------
# Prediction
# ------------------------
if st.button("Predict"):
    lr_pred = lr.predict(X)[0]
    lr_prob = lr.predict_proba(X)[0][1]
    rf_pred = rf.predict(X)[0]
    rf_prob = rf.predict_proba(X)[0][1]

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Logistic Regression")
        st.metric("Prediction", "Fraud 🚨" if lr_pred==1 else "Legitimate ✅")
        st.metric("Fraud Probability", f"{lr_prob:.2%}")
    
    with col2:
        st.subheader("Random Forest")
        st.metric("Prediction", "Fraud 🚨" if rf_pred==1 else "Legitimate ✅")
        st.metric("Fraud Probability", f"{rf_prob:.2%}")

    # ------------------------
    # Visualizations
    # ------------------------
    st.subheader("Transaction Insights")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**Transaction Amount Distribution**")
        fig, ax = plt.subplots()
        sns.histplot(df['Amount'], bins=50, kde=True, color='skyblue')
        ax.axvline(amount, color='red', linestyle='--', label='Current Transaction')
        ax.set_xlabel("Amount")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)
    
    with col4:
        st.markdown("**Fraud Probability Comparison**")
        fig2, ax2 = plt.subplots()
        models = ['Logistic Regression', 'Random Forest']
        probs = [lr_prob, rf_prob]
        sns.barplot(x=models, y=probs, palette='Reds')
        ax2.set_ylim(0,1)
        ax2.set_ylabel("Fraud Probability")
        st.pyplot(fig2)
    
    st.info("Red line in histogram shows your transaction amount. Bar chart shows predicted fraud probabilities from both models.")
