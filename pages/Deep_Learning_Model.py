import streamlit as st
import pandas as pd

# --- 2. Main Page Content ---
st.title("Keras based Deep Learning Model for Patient Appointment Cancellation Prediction")
st.markdown("This page details the performance and architecture of the model used in the predictor.")
st.markdown("---")

# --- 1. Model Architecture ---
st.header("1. Model Architecture")
st.markdown("""
The model is a **Keras/TensorFlow Neural Network** (specifically, a Multi-Layer Perceptron) designed for binary classification. It was trained on historical patient data to predict a single outcome: **Will the patient cancel?**

Its architecture consists of:
- **Input Layer:** Takes in the 8 patient features.
- **Hidden Layers:** Two hidden layers with 'ReLU' activation to learn complex patterns.
- **Dropout:** Dropout layers are used during training to prevent overfitting.
- **Output Layer:** A single 'Sigmoid' neuron that outputs a probability between 0 (Will Attend) and 1 (Will Cancel).

The model was trained using an **Adam optimizer** and **binary cross-entropy** loss function, which are standard for this type of problem.
""")

# --- 2. Model Performance ---
st.header("2. Model Performance")
st.markdown("The model was evaluated on a test set (2,722 appointments) that it had never seen before.")

# --- 2a. Key Metrics Table ---
st.subheader("Key Performance Metrics")

# Create a DataFrame for the metrics table
metrics_data = {
    "Metric": [
        "Overall Accuracy",
        "AUC (Area Under Curve)",
        "Cancelled' Precision",
        "Cancelled' Recall"
    ],
    "Score": [
        "96.7%",
        "~0.978",
        "0.86",
        "0.73"
    ],
    "Interpretation": [
        "Percentage of all predictions (both 'Attended' and 'Cancelled') that were correct.",
        "The model's ability to distinguish between the two classes. (A score of 1.0 is perfect).",
        "When the model predicts 'Cancelled', it is correct 86% of the time.",
        "The model successfully identifies 73% of all patients who will actually cancel."
    ]
}
metrics_df = pd.DataFrame(metrics_data)
st.dataframe(metrics_df, hide_index=True)

st.info("""
**What do these metrics mean for the clinic?**
-   The **'Cancelled' Recall (73%)** is the most important number. It means the model finds almost 3 out of every 4 high-risk patients, allowing staff to intervene.
-   The **'Cancelled' Precision (86%)** is also key. It means the staff won't waste much time, as most of the patients flagged by the model are indeed high-risk.
""")

# --- 2b. Visual Reports ---
st.subheader("Performance Visuals")

# Use columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Confusion Matrix**")
    try:
        st.image("confusion_matrix_dl.png",
                 caption="Actual vs. Predicted outcomes.")
    except FileNotFoundError:
        st.error("Please run the training script to generate 'confusion_matrix.png'.")

st.markdown("""
**How to read these charts:**
-   **Classification Report:** Shows the high performance for `Class 0` (Attended) and the good (but lower) performance for `Class 1` (Cancelled).
-   **Confusion Matrix:** This is the source of the report.
    -   **2459 (Top-Left):** Correctly predicted 'Attended'.
    -   **172 (Bottom-Right):** Correctly predicted 'Cancelled'.
    -   **63 (Bottom-Left):** **False Negatives.** These are the most important errors. The model said 'Attended', but the patient *Cancelled*.
    -   **28 (Top-Right):** **False Positives.** The model said 'Cancelled', but the patient *Attended*.
""")

st.markdown("---")

# --- 3. Model Stability (Training History) ---
st.header("3. Training History")
st.markdown("""
These charts show the model's performance as it learned over 38 "epochs" (rounds of training). The key is to see if the **Validation** line (orange) closely follows the **Train** line (blue).
""")

col3, col4 = st.columns(2)
with col3:
    st.markdown("**Model Accuracy**")
    try:
        st.image("model_accuracy_dl.png")
    except FileNotFoundError:
        st.error("Please run the training script to generate 'model_accuracy.png'.")

with col4:
    st.markdown("**Model Loss**")
    try:
        st.image("model_loss_dl.png")
    except FileNotFoundError:
        st.error("Please run the training script to generate 'model_loss.png'.")

st.success("""
**Conclusion:**
The validation lines are stable and track the training lines well. The validation loss (orange, right chart) is low and doesn't fly upwards.

This indicates that the model is **stable, not overfitting,** and can be trusted to generalize well to new patient data it hasn't seen before.
""")