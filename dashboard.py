import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")

st.title("Post Discharge Recovery Tracker")

# Load Dataset
df = pd.read_csv("rhs_ml_dataset.csv")

# Basic validation
required = ["Age", "Length_of_Stay", "Social_Support_Level", "Readmitted"]
for col in required:
    if col not in df.columns:
        st.error(f"Missing column: {col}")
        st.stop()

# Train Model (No PKL Used)
X = df[["Age", "Length_of_Stay", "Social_Support_Level"]]
y = df["Readmitted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Dashboard KPI
st.subheader("Overview")
st.metric("Total Patients", len(df))
st.metric("Readmission Rate (%)", round(df["Readmitted"].mean()*100,2))

# Donut Chart
fig = px.pie(df, names="Readmitted", hole=0.5)
st.plotly_chart(fig, use_container_width=True)

# Prediction Panel
st.subheader("Predictive Risk Panel")

age = st.number_input("Age", 18, 100, 50)
stay = st.number_input("Length of Stay", 1, 60, 5)
support = st.selectbox("Social Support Level (0=Low,1=Med,2=High)", [0,1,2])

if st.button("Predict"):
    input_data = pd.DataFrame({
        "Age": [age],
        "Length_of_Stay": [stay],
        "Social_Support_Level": [support]
    })
    pred = model.predict(input_data)[0]

    if pred == 1:
        st.error("High Readmission Risk")
    else:
        st.success("Low Readmission Risk")
