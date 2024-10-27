import streamlit as st
import numpy as np
import pickle  # if your model is saved as a pickle file
import warnings
import sklearn
warnings.filterwarnings('ignore')

# Load the trained model (adjust the file path if needed)
with open('gb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Page styling
st.set_page_config(page_title="CKD Prediction", layout="centered")
st.markdown("""
    <style>
        .big-font { font-size:30px !important; }
        .stButton>button { font-size: 20px; padding: 0.5rem 1rem; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">Chronic Kidney Disease Prediction</p>', unsafe_allow_html=True)
st.write("Please enter the following details to predict the likelihood of Chronic Kidney Disease.")

# Organize inputs into columns
col1, col2 = st.columns(2)

# Column 1 inputs
with col1:
    age = st.slider("Age", min_value=0, max_value=100, value=50)
    blood_pressure = st.slider("Blood Pressure (in mmHg)", min_value=0, max_value=200, value=80)
    
    haemoglobin = st.slider("Haemoglobin", min_value=0, max_value=20, value=15)
    albumin = st.slider("Albumin", min_value=0, max_value=5, value=0)
    sugar = st.slider("Sugar", min_value=0, max_value=5, value=0)
    blood_glucose_random = st.slider("Blood Glucose Random", min_value=0, max_value=500, value=120)
    blood_urea = st.slider("Blood Urea", min_value=0, max_value=200, value=40)

# Column 2 inputs
with col2:
    serum_creatinine = st.slider("Serum Creatinine", min_value=0, max_value=15, value=1)
    sodium = st.slider("Sodium", min_value=100, max_value=150, value=135)
    potassium = st.slider("Potassium", min_value=0, max_value=10, value=4)
    specific_gravity = st.selectbox("Specific Gravity", options=[1.005, 1.010, 1.015, 1.020, 1.025])
    packed_cell_volume = st.number_input("Packed Cell Volume", min_value=0, max_value=50, value=30)
    white_blood_cell_count = st.number_input("White Blood Cell Count", min_value=0, max_value=20000, value=8000)
    red_blood_cell_count = st.number_input("Red Blood Cell Count", min_value=0.0, max_value=10.0, value=5.0)

# Organize categorical options in columns
st.write("### Categorical Information")
col3, col4, col5 = st.columns(3)

with col3:
    red_blood_cells = st.selectbox("Red Blood Cells", options=["normal", "abnormal"])
    pus_cell = st.selectbox("Pus Cell", options=["normal", "abnormal"])
    pus_cell_clumps = st.selectbox("Pus Cell Clumps", options=["present", "notpresent"])

with col4:
    bacteria = st.selectbox("Bacteria", options=["present", "notpresent"])
    hypertension = st.selectbox("Hypertension", options=["yes", "no"])
    diabetes_mellitus = st.selectbox("Diabetes Mellitus", options=["yes", "no"])

with col5:
    coronary_artery_disease = st.selectbox("Coronary Artery Disease", options=["yes", "no"])
    appetite = st.selectbox("Appetite", options=["good", "poor"])
    peda_edema = st.selectbox("Pedal Edema", options=["yes", "no"])
    aanemia = st.selectbox("Anemia", options=["yes", "no"])

# Convert categorical fields to numerical
red_blood_cells = 1 if red_blood_cells == 'normal' else 0
pus_cell = 1 if pus_cell == 'normal' else 0
pus_cell_clumps = 1 if pus_cell_clumps == 'present' else 0
bacteria = 1 if bacteria == 'present' else 0
hypertension = 1 if hypertension == 'yes' else 0
diabetes_mellitus = 1 if diabetes_mellitus == 'yes' else 0
coronary_artery_disease = 1 if coronary_artery_disease == 'yes' else 0
appetite = 1 if appetite == 'good' else 0
peda_edema = 1 if peda_edema == 'yes' else 0
aanemia = 1 if aanemia == 'yes' else 0

# Multi-category input conversion for specific_gravity
specific_gravity_mapping = {
    1.005: 0,
    1.010: 1,
    1.015: 2,
    1.020: 3,
    1.025: 4
}
specific_gravity = specific_gravity_mapping[specific_gravity]

# Construct the input data array
input_data = np.array([[age, blood_pressure, specific_gravity, albumin, sugar, red_blood_cells, pus_cell,
                        pus_cell_clumps, bacteria, blood_glucose_random, blood_urea, serum_creatinine,
                        sodium, potassium, haemoglobin, packed_cell_volume, white_blood_cell_count,
                        red_blood_cell_count, hypertension, diabetes_mellitus, coronary_artery_disease,
                        appetite, peda_edema, aanemia]])

# Prediction button
if st.button("Predict"):
    # Make the prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)  # Returns probability for both classes
    
    # Get the probability of the predicted class
    prob = probability[0][prediction[0]]
    
    # Interpret the prediction result and display it with appropriate highlighting
    if prediction[0] == 0:
        result_text = f"Chronic Kidney Disease (Probability: {prob:.2%})"
        st.error(result_text)
    else:
        result_text = f"Not Chronic Kidney Disease (Probability: {prob:.2%})"
        st.success(result_text)
        



