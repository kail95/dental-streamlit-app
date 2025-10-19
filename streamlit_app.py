import streamlit as st
import pandas as pd
import joblib
import numpy as np


# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Appointment Cancellation Prediction",
    page_icon="👨‍⚕️",
    layout="wide"
)

# --- 1. Model and Artifact Loading ---
@st.cache_resource
def load_model_artifacts():
    """Loads the scikit-learn model, scaler, and column list."""
    try:
        model = joblib.load('cancellation_model.joblib')
        scaler = joblib.load('scaler.joblib')
        model_columns = joblib.load('model_columns.joblib')
        return model, scaler, model_columns
    except (FileNotFoundError, IOError):
        return None, None, None

model, scaler, model_columns = load_model_artifacts()

if model is None:
    st.error(
        "🚨 Model files not found! "
        "Please ensure the .joblib files are in the same directory."
    )
    st.stop()

# --- 2. Helper Functions ---
def get_risk_details(probability):
    """Categorizes probability into risk levels and provides recommendations."""
    if probability < 0.30:
        return "✅ Low Risk", "Standard procedure is likely sufficient. This patient has a strong history of attendance."
    elif probability < 0.60:
        return "⚠️ Medium Risk", "A standard confirmation reminder (SMS or call) is recommended. Monitor for any communication from the patient."
    else:
        return "🚨 High Risk", "Consider a proactive confirmation call or a more flexible scheduling option. Prioritize confirming this appointment closer to the date."

# --- 3. Application UI ---
st.title("Patient Appointment Cancellation Predictor")
st.markdown("This app uses a **Scikit-learn Random Forest** model to predict the probability of a patient cancelling their dental appointment.")
st.markdown("---")


# --- 4. Demonstration Section ---
st.header("🔬 Model Demonstration on Sample Patients")
st.markdown("Here are predictions for three sample patients representing different risk profiles.")

sample_data = {
    "Patient Profile": [
        "Low Risk (Reliable Patient)",
        "Medium Risk (Habitual Rescheduler)",
        "High Risk (History of Cancellations)"
    ],
    'total_appointments_last': [10, 15, 4],
    'total_cancelled_appointment_last': [0, 2, 3],
    'total_shifted_appointments_last': [1, 8, 0],
    'tot_on_time_arrivals_last': [9, 5, 1],
    'total_late_arrivals_last': [0, 0, 0],
    'lead_time_days': [7, 30, 60],
    'age_atlast_treatment': [45, 28, 22],
    'town_distance_last': [5.0, 15.0, 25.0]
}
sample_df = pd.DataFrame(sample_data)

try:
    sample_features = sample_df[model_columns]
    sample_scaled = scaler.transform(sample_features)
    sample_predictions_proba = model.predict_proba(sample_scaled)
    cancellation_probabilities = sample_predictions_proba[:, 1]
    
    sample_df["Predicted Probability"] = [f"{p:.0%}" for p in cancellation_probabilities]
    risk_categories, _ = zip(*[get_risk_details(p) for p in cancellation_probabilities])
    sample_df["Risk Category"] = risk_categories
    st.dataframe(sample_df)
except Exception as e:
    st.error(f"An error occurred during sample prediction: {e}")

st.markdown("---")


# --- 5. Interactive Prediction Section ---
st.header("🕹️ Interactive Predictor")
st.markdown("Enter the patient and appointment details below to get a prediction.")

with st.form("prediction_form"):
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient History")
        total_appointments = st.number_input('Total Past Appointments', min_value=0, value=5, step=1)
        past_cancellations = st.number_input('Total Past Cancellations', min_value=0, value=0, step=1)
        past_reschedules = st.number_input('Total Past Reschedules', min_value=0, value=1, step=1)
        on_time_arrivals = st.number_input('Total On-Time Arrivals', min_value=0, value=4, step=1)
        late_arrivals = st.number_input('Total Late Arrivals', min_value=0, value=0, step=1)

    with col2:
        st.subheader("Appointment & Demographics")
        lead_time = st.slider('Lead Time (Days)', 0, 365, 14, help="How many days in advance was the appointment booked?")
        age = st.slider('Patient Age', 5, 100, 35)
        distance = st.slider('Distance from Clinic (km)', 0.0, 100.0, 10.0, 0.5)

    submitted = st.form_submit_button("Predict Cancellation Risk", type="primary")

if submitted:
    data = {
        'total_cancelled_appointment_last': past_cancellations,
        'total_shifted_appointments_last': past_reschedules,
        'tot_on_time_arrivals_last': on_time_arrivals,
        'total_late_arrivals_last': late_arrivals,
        'total_appointments_last': total_appointments,
        'lead_time_days': lead_time,
        'age_atlast_treatment': age,
        'town_distance_last': distance
    }
    
    input_df = pd.DataFrame(data, index=[0])
    input_df_ordered = input_df[model_columns]

    st.subheader("Patient Input Summary")
    st.write(input_df_ordered)

    try:
        input_scaled = scaler.transform(input_df_ordered)
        prediction_proba = model.predict_proba(input_scaled)
        cancellation_prob = prediction_proba[0][1]

        risk_category, recommended_action = get_risk_details(cancellation_prob)

        st.subheader("🔮 Prediction Result")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Cancellation Probability", value=f"{cancellation_prob:.0%}")
        with col2:
            st.markdown(f"**{risk_category}**")
        
        st.progress(cancellation_prob)
        st.info(f"**Recommended Action:** {recommended_action}", icon="💡")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.markdown("*Disclaimer: This prediction is based on historical data and is not a guarantee of future behavior.*")