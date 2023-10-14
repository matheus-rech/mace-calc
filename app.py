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