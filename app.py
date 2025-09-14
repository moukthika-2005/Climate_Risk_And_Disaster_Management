import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
# Page setup
st.set_page_config(page_title="Earthquake Severity Prediction", layout="wide")

# Custom background colors
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #74ebd5 0%, #ACB6E5 100%);
    background-attachment: fixed;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.sub-title {
    font-size:24px !important;
    font-weight: 600;
    color: #1E90FF;
}
.normal-text {
    font-size:18px !important;
}
.centered-chart {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 10%;   /* Adjust width (percentage of page) */
        margin: auto 130px; 
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)
# Load trained model + feature columns
model = joblib.load("earthquake_severity_model.pkl")          # trained ML model
feature_columns = joblib.load("feature_columns.pkl") # list of columns used during training
# Prediction Function
def predict_earthquake(new_data):
    # Convert categoricals to dummies
    new_data = pd.get_dummies(new_data, drop_first=True)
    # Reindex to match training columns
    new_data = new_data.reindex(columns=feature_columns, fill_value=0)
    # Predict probabilities
    probs = model.predict_proba(new_data)[0]
    classes = model.classes_
    prediction = classes[np.argmax(probs)]
    return prediction, probs, classes
# Plot Function
def plot_prediction(classes, probs, prediction):
    colors = {"Low": "green", "Moderate": "orange", "Severe": "red"}
    fig, ax = plt.subplots(figsize=(2.5,2))
    ax.bar(classes, probs, color=[colors[c] for c in classes])
    ax.set_title(f"Prediction: {prediction}", fontsize=10)
    ax.set_ylabel("Probability", fontsize=10)
    ax.set_ylim(0,1)
    return fig
# Streamlit UI
st.title("🌍 Earthquake Severity Prediction")
st.write("Enter earthquake details to predict severity (Low / Moderate / Severe).")
# Example input fields
magnitude = st.number_input("Magnitude", min_value=0.0, max_value=10.0, step=0.1)
depth = st.number_input("Depth (km)", min_value=0.0, max_value=700.0, step=0.1)
latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, step=0.01)
longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, step=0.01)
location = st.text_input("Location")  # categorical feature
if st.button("Predict Severity"):
    # Build DataFrame
    input_df = pd.DataFrame([{
        "magnitude": magnitude,
        "depth": depth,
        "latitude": latitude,
        "longitude": longitude,
        "location": location
    }])
    # Predict
    prediction, probs, classes = predict_earthquake(input_df)
    # Show result
    st.subheader(f"✅ Predicted Severity: {prediction}")
    # Show probability chart
    fig = plot_prediction(classes, probs, prediction)
    st.markdown('<div class="centered-chart">', unsafe_allow_html=True)
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)
