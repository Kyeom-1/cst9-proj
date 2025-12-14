import streamlit as st
import numpy as np
import pandas as pd
import joblib

# --- Load trained models and scaler ---
# Ensure these files ('lr_model.pkl', 'rf_model.pkl', 'scaler.pkl') exist in the application's root directory
try:
    lr = joblib.load('lr_model.pkl')
    rf = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model files (lr_model.pkl, rf_model.pkl, scaler.pkl) not found. Please ensure they are in the correct directory.")
    st.stop()

# --- Load dataset from Google Drive ---
# This dataset is used *only* for the example transactions (fraud_samples)
try:
    csv_url = "https://drive.google.com/uc?export=download&id=1b-LDt5p10Q-GD1aRl0rfYX6vcleQqoWg"
    # Read only the necessary columns to save memory/time
    df = pd.read_csv(csv_url, usecols=['Time'] + ['V'+str(i) for i in range(1,29)] + ['Amount', 'Class'])
    fraud_samples = df[df['Class'] == 1].reset_index(drop=True)
    if fraud_samples.empty:
        st.warning("No fraud samples found in the loaded dataset.")
except Exception as e:
    st.error(f"Could not load example dataset: {e}")
    # Create an empty DataFrame to prevent errors if the load fails
    fraud_samples = pd.DataFrame()


# --- Page configuration ---
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# --- App Header ---
st.title("💳 Credit Card Fraud Detection System")
st.markdown(
    "This interactive app predicts whether a credit card transaction is **fraudulent** or **legitimate**.\n"
    "You can compare two machine learning models: **Logistic Regression** and **Random Forest**.\n"
    "Choose to analyze an **example fraud transaction** or **simulate your own**."
)
st.divider()

# --- Model Selection ---
model_name = st.selectbox(
    "Select Model for Prediction",
    ["Logistic Regression", "Random Forest"],
    help="Choose which trained model will be used for the prediction comparison"
)
# Note: Both models are predicted simultaneously for comparison, but this variable is kept for potential future use.
# model = lr if model_name == "Logistic Regression" else rf

# --- Sidebar Inputs ---
st.sidebar.header("🧾 Transaction Input")
st.sidebar.markdown("Adjust values to simulate a transaction or select an example fraud transaction.")

# Option to select example fraud transaction
use_example = st.sidebar.checkbox("Use Example Fraud Transaction", value=True)

# Initialize inputs
time = 0.0
V_features = [0.0] * 28
amount = 0.0

if use_example and not fraud_samples.empty:
    idx = st.sidebar.selectbox("Select Example Fraud Transaction (Fraud Class=1)", fraud_samples.index)
    row = fraud_samples.loc[idx]
    
    # Extract values from the example
    time = row['Time']
    # The 'V' columns are already selected in the fraud_samples DataFrame
    V_features = row[['V'+str(i) for i in range(1,29)]].tolist()
    amount = row['Amount']
    
    # Display the selected example values in the main page for review
    st.subheader("Selected Example Transaction Details")
    st.markdown(f"* **Time:** `{time}` seconds since the first transaction.")
    st.markdown(f"* **Amount:** `{amount:.2f}` currency units.")
    # Show V features in an expander for detail
    with st.expander("Show Anonymized Pattern Features (V1-V28)"):
        st.dataframe(pd.DataFrame([V_features], columns=[f'V{i}' for i in range(1, 29)]), hide_index=True)

else:
    # --- Custom Simulation Inputs ---
    st.sidebar.caption("Simulate a custom transaction.")
    
    # Time feature
    time = st.sidebar.number_input(
        "Seconds since first transaction", 
        min_value=0.0, 
        value=100000.0, 
        step=1000.0
    )
    
    # PCA V1-V28 features
    st.sidebar.subheader("🔐 Anonymized Transaction Patterns (V1-V28)")
    st.sidebar.caption("Adjust the principal component values. Fraudulent transactions often have extreme values in these components.")
    
    # Create sliders for V1-V28
    V_features = []
    # Set a common, wide range typical of PCA features
    v_range = (-20.0, 20.0) 
    
    # Use columns to make the sidebar less vertical
    v_cols = st.sidebar.columns(4) 
    for i in range(1, 29):
        # Determine which column to place the slider in
        col_idx = (i - 1) % 4
        with v_cols[col_idx]:
            V_features.append(st.slider(f"V{i}", v_range[0], v_range[1], 0.0, key=f"v_slider_{i}", help=f"Principal Component {i}"))

    # Transaction amount
    amount = st.sidebar.number_input(
        "Amount (in currency units)", 
        min_value=0.0, 
        value=100.0, 
        step=10.0
    )

# --- Data Preparation for Model Input ---
# The Amount feature must be scaled using the pre-fitted scaler regardless of whether it's an example or custom input.
# The scaler expects a 2D array, so we pass [[amount]].
amount_scaled = scaler.transform(np.array(amount).reshape(-1, 1))[0][0]

# Construct the final feature vector (Time, V1-V28, Scaled Amount)
# The order must match the training data: [Time, V1, V2, ..., V28, Scaled Amount]
inputs = [time] + V_features + [amount_scaled]
X = np.array(inputs).reshape(1, -1) # Reshape for single sample prediction

# --- Prediction Button and Results ---
if st.button("Predict Transaction Class"):
    
    # Predict with both models
    lr_pred = lr.predict(X)[0]
    # Probability of class 1 (Fraud)
    lr_prob = lr.predict_proba(X)[0][1] 
    
    rf_pred = rf.predict(X)[0]
    # Probability of class 1 (Fraud)
    rf_prob = rf.predict_proba(X)[0][1] 

    st.header("Prediction Results Comparison")
    col1, col2 = st.columns(2)

    # Result for Logistic Regression
    with col1:
        st.subheader("Logistic Regression")
        st.markdown("A **linear** model highly sensitive to the magnitude of features.")
        lr_status = "Fraud 🚨" if lr_pred == 1 else "Legitimate ✅"
        st.metric("Prediction", lr_status)
        st.metric("Fraud Probability", f"{lr_prob:.2%}")
        # Optional: Add explanation based on the primary features
        if lr_pred == 1 and amount > 500: # Heuristic for explanation
             st.caption("🚨 **Note:** High transaction amount is a significant linear factor for this model.")

    # Result for Random Forest
    with col2:
        st.subheader("Random Forest")
        st.markdown("A **non-linear** ensemble model considering complex feature interactions.")
        rf_status = "Fraud 🚨" if rf_pred == 1 else "Legitimate ✅"
        st.metric("Prediction", rf_status)
        st.metric("Fraud Probability", f"{rf_prob:.2%}")
        if rf_pred == 1 and np.abs(np.array(V_features)).max() > 5: # Heuristic for explanation
             st.caption("🚨 **Note:** Extreme values in the anonymized patterns (V-features) often trigger this model.")

    st.info(
        "**Key takeaway:** Logistic Regression is often more sensitive to the magnitude of the scaled **Amount**, "
        "while Random Forest excels at capturing **non-linear interactions** among the anonymized V1–V28 patterns and **Time**."
    )

    # Triggering a diagram to explain the models used
    

# --- Footer or additional context ---
st.divider()
st.caption(f"Model used for comparison: **{model_name}** is highlighted but results from both are shown.")