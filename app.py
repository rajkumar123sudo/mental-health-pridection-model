import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Mental Health Score Predictor", layout="centered")

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_data():
    model_path = BASE_DIR / "mental_health_model.pkl"
    return joblib.load(model_path)

data = load_data()

model = data['model']
top_countries = data['top_countries']

st.title("🧠 Student Mental Health Score Predictor")
st.write("Apni daily routine aur usage details enter karein prediction dekhne ke liye.")

# 2. User Input Form
with st.form("prediction_form"):
    st.subheader("Demographics & Basic Info")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=80, value=20)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        academic_level = st.selectbox("Academic Level", ["High School", "Undergraduate", "Graduate", "Postgraduate"])
    with col2:
        country = st.selectbox("Country", top_countries + ["Other"])
        purpose_of_use = st.selectbox("Purpose Of Use", ["Academic", "Entertainment", "Socializing", "Work"])
        most_used_platform = st.selectbox("Most Used Platform", ["Instagram", "YouTube", "TikTok", "Facebook", "LinkedIn", "Other"])

    st.divider()
    st.subheader("Daily Routine & Habits")
    col3, col4 = st.columns(2)
    with col3:
        study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, value=4.0, step=0.5)
        sleep_hours = st.number_input("Sleep Hours Per Night", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        physical_activity = st.number_input("Physical Activity Hours", min_value=0.0, max_value=24.0, value=1.0, step=0.5)
    with col4:
        avg_usage = st.number_input("Avg Daily Screen Usage (Hours)", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
        daily_unlocks = st.number_input("Daily Phone Unlocks", min_value=0, max_value=500, value=80)
        stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High", "Very High"])

    submit_btn = st.form_submit_button("Predict Score")

# 3. Prediction Logic
if submit_btn:
    # Country grouping logic
    grouped_country = country if country in top_countries else "Other"

    # Input DataFrame banana (Exact training feature columns ke according)
    input_data = pd.DataFrame([{
        'Study_Hours': study_hours,
        'Age': age,
        'Avg_Daily_Usage_Hours': avg_usage,
        'Daily_Unlocks': daily_unlocks,
        'Physical_Activity_Hours': physical_activity,
        'Sleep_Hours_Per_Night': sleep_hours,
        'Stress_Level': stress_level,
        'Gender': gender,
        'Academic_Level': academic_level,
        'Most_Used_Platform': most_used_platform,
        'grouped_country': grouped_country,
        'Purpose_Of_Use': purpose_of_use
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Mental Health Score:    **{prediction:.2f}**")
    