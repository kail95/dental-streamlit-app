import streamlit as st

st.title("ℹ️ 👨‍⚕️ Welcome to the Patient Appointment Predictor")

st.markdown("""
This application demonstrates a machine learning model for predicting patient appointment cancellations.

### Purpose
The application was developed as a part of the Master of Business Administration (MBA) degree project to showcase the suitability of machine learning models in healthcare settings, specifically for predicting patient appointment cancellations. 
            
#### MBA Project:
            
    - Title: Factors Influencing Appointment Compliance and Punctuality Among Dental Surgery Patients in the Private Sector
    - Student Name: Dr. Sajeewa Kandegedara
    - Supervisors: Prof. Sarath S. Kodithuwakku
    - Institution: Postgraduate Institute of Agriculture (PGIA), University of Peradeniya, Sri Lanka

            
### Key Features
- **Interactive Prediction:** Use the form on the 'Predictor' page to get real-time risk scores.
- **Model Demonstration:** See pre-calculated examples for different patient profiles.
- **Data-Driven:** Built using a Scikit-learn Random Forest model trained on historical appointment data.
""")