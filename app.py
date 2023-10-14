import streamlit as st
import joblib
import json
import numpy as np
import pickle

model = joblib.load('model.joblib')

st.title("MACE Prediction After LT")


# Ranges for sliders derived from the dataset
ranges = {
    'Weight': (30, 140, 'Weight (kg):'),
    'Height': (130, 200, 'Height (cm):'),
    'BodyMassIndex': (15, 50, 'Body Mass Index (BMI):'),
    'Hematocrit': (4, 60, 'Hematocrit:'),
    'Leukocytes': (1080, 65000, 'Leukocyte Count:'),
    'Platelets': (4000, 666000, 'Platelets:'),
    'TotalBilirubin': (0, 50, 'Total Bilirubin:'),
    'DirectBilirubin': (0, 30, 'Direct Bilirubin Level:'),
    'Creatinine': (0, 30, 'Creatinine Level:'),
    'Urea': (1, 500, 'Urea Level:'),
    'ProthrombinTimeActivity': (13, 100, 'Prothrombin Time Activity:'),
    'InternationalNormalizedRatio': (0, 10, 'International Normalized Ratio:'),
    'Sodium': (100, 160, 'Sodium:'),
    'Potassium': (1, 10, 'Potassium:'),
    'Albumin': (1, 10, 'Albumin:'),
    'AST': (1, 510, 'Aspartate Aminotransferase (AST):'),
    'ALT': (1, 700, 'Alanine Aminotransferase (ALT):'),
    'GGT': (1, 1900, 'Gamma-Glutamyl Transferase (GGT):'),
    'AlkalinePhosphatase': (10, 1300, 'Alkaline Phosphatase:'),
    'LeftAtriumSize': (20, 90, 'Left Atrium Size:'),
    'DistalVolumeOfLeftVentricle': (10, 100, 'Distal Volume Of Left Ventricle:'),
    'SystolicVolumeOfLeftVentricle': (1, 90, 'Systolic Volume Of Left Ventricle:'),
}


# Get input values from user
numeric_values = {}
for var, (min_val, max_val) in ranges.items():
    numeric_values[var] = st.sidebar.slider(var, min_value=min_val, max_value=max_val, value=min_val)

categorical_cols = {
    ("Race*", ["White", "Mixed/Other", "Black"]),
    ("Sex*", ["Male", "Female"]),
    ("Previous esophageal variceal ligation*", ["No", "Yes"]),
    ("Portal Hypertensive Gastropathy*", ["Mild", "Absent", "Intense"]),
    ("Previous Ascites*", ["Yes", "No"]),
    ("Previous Spontaneous Bacterial Peritonitis*", ["No", "Yes"]),
    ("Previous Hepatopulmonary Syndrome*", ["No", "Yes"]),
    ("Previous use of non-selective beta-blockers*", ["No", "Yes"]),
    ("Portal Vein Thrombosis*", ["No", "Yes"]),
    ("Hepatic encephalopathy*", ["No", "Yes"]),
    ("Previous Hepatorenal Syndrome*", ["No", "Yes"]),
    ("Antibiotic Therapy More Than 24h", ["No", "Yes"]),
    ("Hospitalized For More than 48h", ["No", "Yes"]),
    ("PreTransplant Hemodialysis", ["No", "Yes"]),
    ("Hepatocellular Carcinoma*", ["No", "Yes"]),
    ("Blood Group", ["O", "A", "B", "AB"]),
    ("Congestive Heart Failure*", ["No", "Yes"]),
    ("Angioplasty", ["No", "Yes"]),
    ("Dyslipidemia*", ["No", "Yes"]),
    ("Hypertension", ["No", "Yes"]),
    ("Acute Myocardial Infarction", ["No", "Yes"]),
    ("Stroke", ["Other", "Hemorrhagic", "Ischemic"]),
    ("Diabetes Mellitus*", ["No", "Yes"]),
    ("Valve Replacement", ["Other", "Biological", "Metallic"]),
    ("Mitral Insufficiency", ["Other", "Yes"]),
    ("Tricuspid Insufficiency", ["Yes"]),
    ("Non-invasive Diagnostic Method", ["Yes", "No"]),
    ("Dynamic Alteration", ["No", "Yes"])
}




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
    prediction = model.predict(features_array)
    return prediction

# Make prediction and display result
if st.button('Predict'):
    prediction = predict(features_array)
    if prediction[0] == 1:
        st.write("High risk of MACE")
    elif prediction[0] == 0:
        st.write("Low risk of MACE")
    else:
        st.write("Unexpected prediction value: please check the model.")

# Add disclaimer
disclaimer_text = """
#### Disclaimer

*This tool is designed for general educational purposes only and is not intended in any way to substitute for professional medical advice, consultation, diagnosis, or treatment. Any analysis, report, or information contained in or produced by this tool is intended to serve as a supplement to, and not a substitute for the knowledge, expertise, skill and judgment of health care professionals. In no event shall this tool under this Agreement, be considered to be in any form, medical care, treatment, or therapy for patients or users of this tool.*

*This tool's services are provided 'as is'. These services provide no warranties, express or implied and shall not be liable for any direct, consequential, lost profits, or other damages incurred by the user of this information tool.*

*The default values included in the web application are placeholders. Clinicians can modify the inputs as per the clinical characteristics of individual patients to examine the impact on survival prediction in real-time. Clinicians should be aware that the algorithms require complete information for the factors included in the interface.*
"""

st.markdown(disclaimer_text, unsafe_allow_html=True)