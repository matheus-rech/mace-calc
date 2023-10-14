import streamlit as st
import joblib
import json
import numpy as np
import pickle

model = joblib.load('model.joblib')

st.title("MACE Prediction After LT")

numeric_cols = [
    'Weight', 'Height', 'BodyMassIndex', 'Hematocrit', 'Leukocytes', 'Platelets',
    'TotalBilirubin', 'DirectBilirubin', 'Creatinine', 'Urea', 'ProthrombinTimeActivity',
    'InternationalNormalizedRatio', 'Sodium', 'Potassium', 'Albumin', 'AST', 'ALT', 'GGT',
    'AlkalinePhosphatase', 'LeftAtriumSize', 'DistalVolumeOfLeftVentricle',
    'SystolicVolumeOfLeftVentricle'
]

categorical_cols = [
    ("Race", ["White", "Black", "Mixed/Other"]),
    ("Sex", ["Female", "Male"]),
    ("PreviousVaricealBandLigation", ["No", "Yes"]),
    ("PortalHypertensiveGastropathy", ["Mild", "Severe", "None"]),
    ("Ascites", ["No", "Yes"]),
    ("SpontaneousBacterialPeritonitis", ["No", "Yes"]),
    ("HepatopulmonarySyndrome", ["No", "Yes"]),
    ("BetaBlockerUse", ["No", "Yes"]),
    ("PortalVeinThrombosis", ["No", "Yes"]),
    ("HepaticEncephalopathy", ["No", "Yes"]),
    ("HepatorenalSyndrome", ["No", "Yes"]),
    ("AntibioticTherapyFor24h", ["No", "Yes"]),
    ("HospitalizedFor48h", ["No", "Yes"]),
    ("PreTransplantHemodialysis", ["No", "Yes"]),
    ("HepatocellularCarcinoma", ["No", "Yes"]),
    ("MitralInsufficiency", ["Mild", "Moderate", "Severe", "Absent"]),
    ("TricuspidInsufficiency", ["Mild", "Moderate", "Severe", "Absent"]),
    ("BloodGroup", ["A", "B", "AB", "O"]),
    ("CongestiveHeartFailure", ["No", "Yes"]),
    ("Angioplasty", ["No", "Yes"]),
    ("Dyslipidemia", ["No", "Yes"]),
    ("Hypertension", ["No", "Yes"]),
    ("AcuteMyocardialInfarction", ["No", "Yes"]),
    ("Stroke", ["Ischemic", "Hemorrhagic", "No"]),
    ("DiabetesMellitus", ["No", "Yes"]),
    ("DynamicAlteration", ["No", "Yes"]),
    ("NonInvasiveMethod", ["No", "Yes"]),
    ("ValveReplacement", ["No", "Yes"])
]


# Create input fields for numeric variables
numeric_values = {}
for numeric_var in numeric_cols:
    numeric_values[numeric_var] = st.sidebar.number_input(numeric_var, value=0.0)

# Create select boxes for categorical variables with numeric values
categorical_values = {}
for categorical_var, classes in categorical_cols:
    options = {cls: i for i, cls in enumerate(classes)}
    selected_option = st.sidebar.selectbox(categorical_var, list(options.keys()))
    categorical_values[categorical_var] = options[selected_option]

# Combine numeric and categorical values into a single dictionary
features = {**numeric_values, **categorical_values}

# Convert features dictionary to a 2D array for model prediction
features_array = np.array(list(features.values())).reshape(1, -1)

# Define the predict function
def predict(features_array):
    # Make predictions using the model
    prediction = model.predict(features_array)
    return prediction

if st.button('Predict'):
    # Get the prediction
    prediction = predict(features_array)
    st.write(f"Prediction: {prediction[0]}")