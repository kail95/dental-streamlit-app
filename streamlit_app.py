import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page config
st.set_page_config(
    page_title="Patient Prediction Dashboard",
    page_icon="🏥",
    layout="wide"
)

# --- 1. Load All Models and Artifacts ---
# We use try/except blocks for robust error handling

try:
    # Load cancellation model
    model_cancel = joblib.load('cancellation_model.joblib') 
    features_cancel = joblib.load('model_columns.joblib')
    st.session_state['cancel_model_loaded'] = True
except FileNotFoundError:
    st.error("Error: 'cancellation_model.joblib' or 'model_columns.joblib' not found.")
    st.info("Please run the 'regenerate_all_models.py' script first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading cancellation model: {e}")
    st.info("Please ensure your scikit-learn version matches the one used for training.")
    st.stop()

# Load the regression model and its features
try:
    model_arrival = joblib.load('arrival_time_model.joblib')
    features_arrival = joblib.load('arrival_time_model_columns.joblib')
    st.session_state['arrival_model_loaded'] = True
except FileNotFoundError:
    st.error("Error: 'arrival_time_model.joblib' or 'arrival_time_model_columns.joblib' not found.")
    st.info("Please run the 'regenerate_all_models.py' script first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading arrival time model: {e}")
    st.info("This is often a scikit-learn version mismatch. Please re-train your models.")
    st.stop()


# --- 2. Main Page Title ---
st.title("🏥 Patient Appointment Prediction Dashboard")
st.write("This app predicts both **appointment cancellation** and **arrival punctuality** based on patient history.")
st.markdown("---")


# --- 3. User Input Form (on Main Page) ---

def get_user_inputs():
    """
    Collects all user inputs from the main page form for BOTH models.
    """
    st.subheader("👤 Patient History Inputs")
    st.write("Fill in the patient's details below and click 'Run Predictions'.")

    # Use columns for a cleaner layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Appointment & Scheduling**")
        total_appointments_last = st.number_input("Total Past Appointments", min_value=0, value=10)
        lead_time_days = st.number_input("Lead Days to Appointment", min_value=0, value=7)
        town_distance_last = st.number_input("Town Distance (km)", min_value=0.0, value=10.0, format="%.2f")

    with col2:
        st.write("**Patient History**")
        total_cancelled_appointment_last = st.number_input("Total Past Cancellations", min_value=0, value=1)
        total_shifted_appointments_last = st.number_input("Total Past Reschedules", min_value=0, value=0)
        past_app_confirmed_count = st.number_input("Total Past Confirmed Appointments", min_value=0, value=9)
        

    with col3:
        st.write("**Patient Demographics & Behavior**")
        age_atlast_treatment = st.number_input("Patient Age", min_value=0, max_value=120, value=35)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
    st.markdown("---")
    st.write("**Detailed Arrival & Waiting History**")
    
    col4, col5, col6, col7 = st.columns(4)

    with col4:
        tot_on_time_arrivals_last = st.number_input("Total Past On-Time Arrivals", min_value=0, value=8)
    with col5:
        total_late_arrivals_last = st.number_input("Total Past Late Arrivals", min_value=0, value=1)
    with col6:
        past_avg_timediff_early = st.number_input("Past Avg. Mins Early", min_value=0.0, value=2.0, format="%.2f")
    with col7:
        past_avg_timediff_late = st.number_input("Past Avg. Mins Late", min_value=0.0, value=5.0, format="%.2f")

    past_avg_waiting_time = st.number_input("Past Avg. Waiting Time (Mins)", min_value=0.0, value=10.0, format="%.2f")


    # Create a single dictionary to hold all inputs
    data = {
        # --- Features for Cancellation Model ---
        'total_cancelled_appointment_last': total_cancelled_appointment_last,
        'total_shifted_appointments_last': total_shifted_appointments_last,
        'tot_on_time_arrivals_last': tot_on_time_arrivals_last,
        'total_late_arrivals_last': total_late_arrivals_last,
        'total_appointments_last': total_appointments_last,
        'lead_time_days': lead_time_days,
        'age_atlast_treatment': age_atlast_treatment,
        'town_distance_last': town_distance_last,

        # --- Features for Arrival Time Model ---
        # Shared features (using the same input variable)
        'past_appointment_count': total_appointments_last,
        'past_cancel_count': total_cancelled_appointment_last,
        'past_shift_count': total_shifted_appointments_last,
        'ageat_app': age_atlast_treatment,
        
        # New features
        'gender': gender,
        'past_app_confirmed_count': past_app_confirmed_count,
        'past_avg_timediff_late': past_avg_timediff_late,
        'past_avg_timediff_early': past_avg_timediff_early,
        'past_avg_waiting_time': past_avg_waiting_time
    }
    
    st.markdown("---")
    # The button is now part of the main page form
    predict_button = st.button("Run Predictions", type="primary")

    return data, predict_button

# Collect inputs and button state from the form function
inputs, predict_button = get_user_inputs()


# --- 4. Main Page Prediction Logic ---

if predict_button:
    
    st.header("📈 Prediction Results")
    pred_col1, pred_col2 = st.columns(2)

    # --- Prediction 1: Cancellation (Existing Model) ---
    with pred_col1:
        st.subheader("1. Appointment Cancellation")
        
        try:
            # Create DataFrame for cancellation model
            input_df_cancel = pd.DataFrame([inputs], columns=features_cancel)
            
            # Make prediction
            prediction_cancel = model_cancel.predict(input_df_cancel)[0]
            prediction_proba_cancel = model_cancel.predict_proba(input_df_cancel)[0]
            
            # Display result
            if prediction_cancel == 1:
                st.error(f"**Prediction: Patient will CANCEL** \n\n(Probability: {prediction_proba_cancel[1]:.2%})")
            else:
                st.success(f"**Prediction: Patient will ATTEND** \n\n(Probability: {prediction_proba_cancel[0]:.2%})")
        except Exception as e:
            st.error(f"An error occurred during cancellation prediction: {e}")

    # --- Prediction 2: Arrival Time (New Model) ---
    with pred_col2:
        st.subheader("2. Patient Arrival Time")

        # Re-create the engineered features from your training script
        epsilon = 1  # To avoid division by zero
        inputs['past_cancel_rate'] = inputs['past_cancel_count'] / (inputs['past_appointment_count'] + epsilon)
        inputs['past_shift_rate'] = inputs['past_shift_count'] / (inputs['past_appointment_count'] + epsilon)
        inputs['past_app_confirmed_rate'] = inputs['past_app_confirmed_count'] / (inputs['past_appointment_count'] + epsilon)

        try:
            input_df_arrival = pd.DataFrame([inputs], columns=features_arrival)
            
            # Make prediction. The pipeline handles all scaling and encoding.
            prediction_arrival = model_arrival.predict(input_df_arrival)[0]
            prediction_mins = round(prediction_arrival, 2)

            # Display result in a user-friendly way
            if prediction_mins > 1:
                st.warning(f"**Prediction: Patient will be {abs(prediction_mins)} minutes LATE**")
            elif prediction_mins < -1:
                st.info(f"**Prediction: Patient will be {abs(prediction_mins)} minutes EARLY**")
            else:
                st.success(f"**Prediction: Patient will be ON TIME** (within 1 minute)")

            st.caption(f"Raw model output: {prediction_mins} minutes. (Positive = Late, Negative = Early)")

        except Exception as e:
            st.error(f"An error occurred during arrival time prediction: {e}")
            st.error("Please check if the input data matches the model's requirements.")

else:
    st.info("Please fill in the patient data above and click 'Run Predictions'.")

# --- 5. (Optional) Show Raw Input Data ---
st.markdown("---")
with st.expander("Show Raw Input Data"):
    st.write("The following data was collected from the form and used for prediction:")
    
    # Display the combined 'inputs' dictionary
    st.json(inputs)
