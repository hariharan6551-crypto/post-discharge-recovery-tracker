import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="Post Discharge Recovery Tracker", layout="wide")

st.title("Post Discharge Social Support & Recovery Tracker")

# ===============================
# LOAD DATA
# ===============================

@st.cache_data
def load_data():
    return pd.read_csv("rhs_ml_dataset.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ===============================
# DASHBOARD VISUALS
# ===============================

st.subheader("Readmission Distribution")

if "Readmitted" in df.columns:
    fig1 = px.histogram(df, x="Readmitted", title="Readmission Count")
    st.plotly_chart(fig1, use_container_width=True)

if "Age" in df.columns:
    fig2 = px.histogram(df, x="Age", title="Age Distribution")
    st.plotly_chart(fig2, use_container_width=True)

# ===============================
# LOAD MODEL
# ===============================

@st.cache_resource
def load_model():
    return joblib.load("readmission_model.pkl")

model = load_model()

# ===============================
# PREDICTION SECTION
# ===============================

st.subheader("Predict Patient Readmission Risk")

age = st.number_input("Age", 18, 100, 50)
length_of_stay = st.number_input("Length of Stay (days)", 1, 60, 5)
social_support = st.selectbox("Social Support Level", ["Low", "Medium", "High"])

# Encode social support manually (adjust if your encoding differs)
support_map = {"Low": 0, "Medium": 1, "High": 2}
support_encoded = support_map[social_support]

if st.button("Predict Risk"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Length_of_Stay": [length_of_stay],
        "Social_Support_Level": [support_encoded]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"High Readmission Risk (Probability: {probability:.2f})")
    else:
        st.success(f"Low Readmission Risk (Probability: {probability:.2f})")
