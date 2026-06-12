import streamlit as st
import pandas as pd
import joblib
import os

# Set page layout to wide to match your UI design
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

st.title("📞 Telco Customer Churn Prediction App")
st.markdown("**End-to-End Data Analyst Project**")

# 1. Safe Model Loading
MODEL_PATH = os.path.join("models", "churn_model.pkl")


@st.cache_resource
def load_model():
    try:
        # Load the trained model artifact
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"❌ Model file not found! Please ensure your model is saved at: {MODEL_PATH}")
        return None


model = load_model()

# Stop execution gracefully if the model isn't loaded yet
if model is None:
    st.stop()

# 2. Sidebar UI Input Fields
st.sidebar.header("Enter Customer Details")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 18.25, 118.75, 70.0)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 9000.0, 800.0)

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# 3. Create Feature Dictionary Matching One-Hot Encoded Notebook Outputs
input_data = {
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Gender_Male': [1 if gender == "Male" else 0],
    'SeniorCitizen': [1 if senior == "Yes" else 0],
    'Partner_Yes': [1 if partner == "Yes" else 0],
    'Dependents_Yes': [1 if dependents == "Yes" else 0],
    'MultipleLines_No phone service': [1 if multiple_lines == "No phone service" else 0],
    'MultipleLines_Yes': [1 if multiple_lines == "Yes" else 0],
    'InternetService_Fiber optic': [1 if internet == "Fiber optic" else 0],
    'InternetService_No': [1 if internet == "No" else 0],
    'OnlineBackup_No internet service': [1 if online_backup == "No internet service" else 0],
    'OnlineBackup_Yes': [1 if online_backup == "Yes" else 0],
    'DeviceProtection_No internet service': [1 if device_protection == "No internet service" else 0],
    'DeviceProtection_Yes': [1 if device_protection == "Yes" else 0],
    'TechSupport_No internet service': [1 if tech_support == "No internet service" else 0],
    'TechSupport_Yes': [1 if tech_support == "Yes" else 0],
    'Contract_One year': [1 if contract == "One year" else 0],
    'Contract_Two year': [1 if contract == "Two year" else 0],
    'PaperlessBilling_Yes': [1 if paperless == "Yes" else 0],
    'PaymentMethod_Credit card (automatic)': [1 if payment_method == "Credit card (automatic)" else 0],
    'PaymentMethod_Electronic check': [1 if payment_method == "Electronic check" else 0],
    'PaymentMethod_Mailed check': [1 if payment_method == "Mailed check" else 0]
}

# Convert input dictionary into a pandas DataFrame
input_df = pd.DataFrame(input_data)

# 4. CRITICAL: Enforce exact feature order expected by your scikit-learn model
if hasattr(model, "feature_names_in_"):
    # Reindex the dataframe columns dynamically to match training data structure
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# 5. Prediction Execution
if st.sidebar.button("🚀 Predict Churn", type="primary"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error(f"🚨 **High Risk**: This customer is likely to Churn! (Probability: {probability:.1%})")
        else:
            st.success(f"✅ **Low Risk**: This customer is stable and likely to Stay. (Probability: {probability:.1%})")

    with col2:
        st.metric(label="Calculated Churn Probability", value=f"{probability:.1%}")

# 6. Static Project Performance Meta (Footer)
st.markdown("---")
st.subheader("Model Performance Details")
st.write("📈 **Model Used**: Random Forest Classifier • **Target Metric**: ROC-AUC ~0.85")
st.info(
    "💡 *Tip: If predictions feel off, double check that your input field names match your original notebook feature transforms.*")