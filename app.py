import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title='Athletic Performance Prediction', layout='centered')
st.title("🏃 Athletic Performance Level Predictor")

st.write("""Enter athlete features to predict performance level: Beginner / Intermediate / Advanced / Elite.""")

# Load artifacts
try:
    model = joblib.load('athlete_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
except Exception as e:
    st.error('Model files not found. Run training script/notebook to create artifacts.')
    st.stop()

# Input widgets
age = st.slider('Age', 5, 60, 18)
gender = st.selectbox('Gender', ['Male','Female'])
height = st.number_input('Height (cm)', 130.0, 220.0, 175.0)
weight = st.number_input('Weight (kg)', 40.0, 150.0, 70.0)
training_hours = st.slider('Training hours per week', 0.0, 40.0, 8.0)
resting_hr = st.slider('Resting heart rate (bpm)', 40, 120, 60)
vo2max = st.slider('VO2 Max', 20.0, 80.0, 45.0)
reaction = st.slider('Reaction time (ms)', 120.0, 700.0, 250.0)
sleep_hours = st.slider('Sleep hours per night', 4.0, 10.0, 7.0)
flexibility = st.slider('Flexibility score (0-100)', 0.0, 100.0, 30.0)

bmi = round(weight / ((height/100)**2),1)
gender_val = 1 if gender=='Male' else 0

input_df = pd.DataFrame([{
    'age': age,
    'gender': gender_val,
    'height_cm': height,
    'weight_kg': weight,
    'bmi': bmi,
    'training_hours_per_week': training_hours,
    'resting_heart_rate': resting_hr,
    'vo2max': vo2max,
    'reaction_time_ms': reaction,
    'sleep_hours': sleep_hours,
    'flexibility_score': flexibility
}])

X_scaled = scaler.transform(input_df)
pred = model.predict(X_scaled)[0]
pred_label = le.inverse_transform([pred])[0]
prob = model.predict_proba(X_scaled).max()

st.markdown(f"### Predicted performance level: **{pred_label}**")
st.markdown(f"**Confidence:** {prob*100:.2f}%")

st.write('---')
st.subheader('Input values')
st.write(input_df.T)
