import streamlit as st
import pandas as pd

# --- 2. Main Page Content ---
st.title("Machine Learning Model for Patient Appointment Cancellation Prediction (Random Forest Model)")
st.markdown("This page provides insights into the Random Forest model used for cancellation prediction, including its performance and key drivers.")
st.markdown("---")

# --- 1. Model Overview ---
st.header("1. Model Overview")
st.markdown("""
The model employed for predicting appointment cancellations is a **Random Forest Classifier**.
Random Forests are powerful ensemble learning methods that work by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

**Key characteristics of this model:**
-   **Ensemble Method:** Combines multiple decision trees to improve accuracy and control overfitting.
-   **Robustness:** Less prone to overfitting compared to single decision trees.
-   **Feature Importance:** Naturally provides insights into which features are most influential.
-   **Tuned for Performance:** Hyperparameters were carefully tuned using `GridSearchCV` and cross-validation to optimize for high **AUC (Area Under the Receiver Operating Characteristic Curve)**, which is crucial for imbalanced datasets like cancellation prediction.
""")

# --- 2. Model Performance ---
st.header("2. Model Performance on Test Data")
st.markdown("The model was evaluated on a held-out test set (2,722 appointments) that it had not seen during training.")

# --- 2a. Key Metrics Table ---
st.subheader("Key Performance Metrics")

# Use values from your provided output for the Random Forest model
metrics_data = {
    "Metric": [
        "Overall Accuracy",
        "AUC (Area Under Curve)",
        "Cancelled' Precision", # Precision for Class 1
        "Cancelled' Recall"     # Recall for Class 1
    ],
    "Score": [
        "94.78%",  # From 'Overall Accuracy: 0.9478'
        "~0.975",  # Average of 95% CI (0.9738, 0.9770)
        "0.64",    # From Classification Report for class 1
        "0.88"     # From Classification Report for class 1
    ],
    "Interpretation": [
        "Percentage of all predictions (both 'Attended' and 'Cancelled') that were correct.",
        "The model's ability to distinguish between the two classes (0.5 is random, 1.0 is perfect).",
        "When the model predicts 'Cancelled', it is correct 64% of the time.",
        "The model successfully identifies 88% of all patients who will actually cancel."
    ]
}
metrics_df = pd.DataFrame(metrics_data)
st.dataframe(metrics_df, hide_index=True)

st.info("""
**What do these metrics mean for a dental clinic?**
-   The **'Cancelled' Recall (88%)** is excellent. This means the model successfully identifies almost 9 out of every 10 patients who will actually cancel their appointment. This allows for highly effective proactive interventions.
-   The **'Cancelled' Precision (64%)** indicates that when the model flags a patient as 'high risk' (likely to cancel), there's a 64% chance they actually will. This suggests that while it catches many cancellations, there's still room for refinement to reduce false alarms. The balance between recall and precision is often a strategic decision for the clinic.
""")

# --- 2b. Confusion Matrix ---

col1, col2 = st.columns(2)

with col1:
    st.subheader("Confusion Matrix")
    st.markdown("This matrix visualizes the number of correct and incorrect predictions made by the model.")
    try:
        st.image("confusion_matrix_sklearn.png",
                 caption="Confusion Matrix for Random Forest Model (Actual vs. Predicted)",
             use_container_width=True)
    except FileNotFoundError:
        st.error("Please ensure 'confusion_matrix_sklearn.png' is in the main project folder.")

st.markdown("""
**Interpreting the Confusion Matrix:**
-   **2373 (Top-Left):** **True Negatives.** Patients who actually attended AND were predicted to attend. (Correctly identified as 'low risk').
-   **207 (Bottom-Right):** **True Positives.** Patients who actually cancelled AND were predicted to cancel. (Correctly identified as 'high risk').
-   **28 (Bottom-Left):** **False Negatives.** Patients who actually cancelled BUT were predicted to attend. These are missed cancellations, where proactive intervention opportunities were lost.
-   **114 (Top-Right):** **False Positives.** Patients who actually attended BUT were predicted to cancel. These are 'false alarms', where resources might be spent on unnecessary interventions.
""")

st.markdown("---")

# --- 3. Model Interpretation with SHAP ---
st.header("3. Understanding Feature Importance (SHAP Values)")
st.markdown("""
SHAP (SHapley Additive exPlanations) values help us understand how each feature contributes to the model's prediction for a specific outcome. The plot below shows the **global importance** of each feature – meaning, on average, how much each feature impacts the model's prediction of a *cancellation*.
""")

col3, col4 = st.columns(2)

with col3:
    st.markdown("**SHAP Summary Plot**")
    try:
        st.image("dental_data_shap_analysis.png",
                caption="SHAP Global Feature Importance for Predicting Cancellation",
                use_container_width=True)
    except FileNotFoundError:
        st.error("Please ensure 'shap_summary_bar.png' is in the main project folder.")


st.markdown("""
**Key Takeaways from the SHAP Plot:**
1.  **`total_cancelled_appointment_last` (Most Important):** This is by far the strongest predictor. Patients with a history of past cancellations are significantly more likely to cancel future appointments. This confirms intuition and is a robust signal.
2.  **`total_shifted_appointments_last`:** Patients who frequently reschedule (shift appointments) also have a notable impact on the cancellation prediction, suggesting a tendency towards less firm commitments.
3.  **`lead_time_days`:** The length of time between booking and the appointment date is also important. Longer lead times might increase the likelihood of cancellation, possibly due to changing plans or forgetfulness.
4.  **`tot_on_time_arrivals_last` & `total_appointments_last`:** These features indicate overall engagement and reliability. More past appointments and on-time arrivals generally reduce the cancellation risk.
5.  **Less Influential:** `total_late_arrivals_last`, `age_atlast_treatment`, and `town_distance_last` have a comparatively smaller average impact on the model's cancellation predictions.

Understanding these feature importances allows the clinic to prioritize interventions. For example, focusing on patients with a high number of past cancellations or shifted appointments would be most impactful.
""")
