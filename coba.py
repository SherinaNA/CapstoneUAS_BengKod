import numpy as np
import pandas as pd
import pickle

# Load scaler & model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('final_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoders/ordinal_mappings.pkl', 'rb') as f:
    ordinal_mappings = pickle.load(f)

# Misal ini input predict
input_df = pd.DataFrame([{
    'Age': 25,
    'Gender': 1,
    'Height': 170,
    'Weight': 65,
    'AlcoholConsumption': 'Sometimes',
    'HighCalorieFood': 1,
    'VegetableConsumption': 2,
    'MealFrequency': 3,
    'CalorieMonitoring': 0,
    'WaterIntake': 8,
    'FamilyHistoryOverweight': 0,
    'PhysicalActivity': 2,
    'TechnologyUse': 4,
    'SnackConsumption': 'No',
    'Transportation': 1
}])

# Mapping ordinal dulu
for col in ['AlcoholConsumption', 'SnackConsumption']:
    input_df[col] = input_df[col].map(ordinal_mappings[col])

# Scaling 13 kolom numeric
scaler_cols = list(scaler.feature_names_in_)
scaled_numeric = scaler.transform(input_df[scaler_cols])

# Gabungkan hasil scaling dengan 2 kolom ordinal
final_features = np.hstack([
    scaled_numeric,
    input_df[['AlcoholConsumption', 'SnackConsumption']].values
])

# Prediksi
pred = model.predict(final_features)[0]
print(f'Prediksi: {pred}')