import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Load model
with open('final_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load ordinal mapping
with open('encoders/ordinal_mappings.pkl', 'rb') as f:
    ordinal_mappings = pickle.load(f)

# Load label encoders
with open('encoders/CalorieMonitoring_encoder.pkl', 'rb') as f:
    calorie_encoder = pickle.load(f)
with open('encoders/FamilyHistoryOverweight_encoder.pkl', 'rb') as f:
    family_encoder = pickle.load(f)
with open('encoders/Gender_encoder.pkl', 'rb') as f:
    gender_encoder = pickle.load(f)
with open('encoders/HighCalorieFood_encoder.pkl', 'rb') as f:
    highcal_encoder = pickle.load(f)
with open('encoders/Transportation_encoder.pkl', 'rb') as f:
    transport_encoder = pickle.load(f)

# Kolom numeric yang di-scale
scaler_columns = list(scaler.feature_names_in_)

# Streamlit app
st.title("🎯 Obesity Level Prediction")

with st.form("user_input_form"):
    st.subheader("Masukkan Data Diri:")

    age = st.number_input("Age", min_value=1, max_value=100)
    gender = st.selectbox("Gender", gender_encoder.classes_)
    height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0)
    height = height_cm / 100  # convert ke meter sebelum masuk ke model
    weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0)

    alcohol = st.selectbox("Alcohol Consumption", list(ordinal_mappings['AlcoholConsumption'].keys()))
    high_calorie = st.selectbox("High Calorie Food?", highcal_encoder.classes_)
    vegetable = st.selectbox("Vegetable Consumption", [1, 2, 3])
    meal_freq = st.selectbox("Meal Frequency", [1, 2, 3, 4])
    calorie_monitoring = st.selectbox("Calorie Monitoring?", calorie_encoder.classes_)
    water_intake = st.selectbox("Water Intake (cups/day)", [1, 2, 3, 4, 5, 6, 7, 8])
    family_history = st.selectbox("Family History Overweight?", family_encoder.classes_)
    physical_activity = st.selectbox("Physical Activity (times/week)", [0, 1, 2, 3, 4])
    tech_use = st.selectbox("Technology Use (hours/day category)", [0, 1, 2, 3, 4])
    snack = st.selectbox("Snack Consumption", list(ordinal_mappings['SnackConsumption'].keys()))
    transportation = st.selectbox("Transportation", transport_encoder.classes_)

    submitted = st.form_submit_button("Prediksi Obesity Level")

if submitted:
    # Buat dataframe input
    input_df = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Height': height,  # meter
        'Weight': weight,
        'AlcoholConsumption': alcohol,
        'HighCalorieFood': high_calorie,
        'VegetableConsumption': vegetable,
        'MealFrequency': meal_freq,
        'CalorieMonitoring': calorie_monitoring,
        'WaterIntake': water_intake,
        'FamilyHistoryOverweight': family_history,
        'PhysicalActivity': physical_activity,
        'TechnologyUse': tech_use,
        'SnackConsumption': snack,
        'Transportation': transportation
    }])

    # Mapping ordinal
    for col in ['AlcoholConsumption', 'SnackConsumption']:
        input_df[col] = input_df[col].map(ordinal_mappings[col])

    # Encoding label encoder
    input_df['Gender'] = gender_encoder.transform(input_df['Gender'])
    input_df['HighCalorieFood'] = highcal_encoder.transform(input_df['HighCalorieFood'])
    input_df['CalorieMonitoring'] = calorie_encoder.transform(input_df['CalorieMonitoring'])
    input_df['FamilyHistoryOverweight'] = family_encoder.transform(input_df['FamilyHistoryOverweight'])
    input_df['Transportation'] = transport_encoder.transform(input_df['Transportation'])

    # Scaling numeric features
    scaled_numeric = scaler.transform(input_df[scaler_columns])

    # Gabungkan scaled + 2 kolom ordinal
    final_features = np.hstack([
        scaled_numeric,
        input_df[['AlcoholConsumption', 'SnackConsumption']].values
    ])

    # Prediksi
    pred = model.predict(final_features)[0]

    # Load LabelEncoder object untuk hasil prediksi
    with open('encoders/ObesityLevel_encoder.pkl', 'rb') as f:
        obesity_encoder = pickle.load(f)

    # Inverse transform hasil prediksi
    pred_label = obesity_encoder.inverse_transform([pred])[0]

    # Tampilkan hasil
    st.success(f"Prediksi Obesity Level Anda: **{pred_label}**")