import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Post Discharge Recovery Tracker", layout="wide")

st.title("Post Discharge Social Support & Recovery Tracker")
st.markdown("AI-Based Predictive & Recovery Monitoring Dashboard")

# ---------------- LOAD DATA ---------------- #

@st.cache_data
def load_data():
    df = pd.read_csv("rhs_ml_dataset.csv")

    # ---- AUTO FIX COMMON DATASET COLUMN NAMES ---- #
    if "Sex_Breakdown" in df.columns:
        df.rename(columns={"Sex_Breakdown": "Gender"}, inplace=True)

    if "Indicator_value" in df.columns:
        df.rename(columns={"Indicator_value": "Readmitted"}, inplace=True)

    if "Time_Period" in df.columns:
        df.rename(columns={"Time_Period": "Year"}, inplace=True)

    # ---- CREATE MISSING COLUMNS IF NOT PRESENT ---- #
    if "Length_of_Stay" not in df.columns:
        df["Length_of_Stay"] = 5

    if "Social_Support_Level" not in df.columns:
        df["Social_Support_Level"] = 1

    if "Gender" not in df.columns:
        df["Gender"] = "Male"

    if "Readmitted" not in df.columns:
        df["Readmitted"] = 0

    if "Year" not in df.columns:
        df["Year"] = 2023

    return df


df = load_data()

# ---------------- FILTER PANEL ---------------- #

st.sidebar.header("Filter Panel")

year = st.sidebar.selectbox("Year", sorted(df["Year"].unique()))
gender = st.sidebar.selectbox("Gender", ["All"] + sorted(df["Gender"].unique()))

filtered_df = df[df["Year"] == year]

if gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == gender]

# ---------------- KPI METRICS ---------------- #

col1, col2, col3 = st.columns(3)

col1.metric("Total Patients", len(filtered_df))

col2.metric(
    "Readmission Rate (%)",
    f"{filtered_df['Readmitted'].mean()*100:.2f}"
)

col3.metric(
    "Avg Stay",
    f"{filtered_df['Length_of_Stay'].mean():.1f} Days"
)

# ---------------- YEARLY TREND ---------------- #

trend = df.groupby("Year")["Readmitted"].mean().reset_index()

fig_bar = px.bar(
    trend,
    x="Year",
    y="Readmitted",
    title="Yearly Readmission Trend"
)

st.plotly_chart(fig_bar, use_container_width=True)

# ---------------- DONUT CHART ---------------- #

fig_donut = px.pie(
    filtered_df,
    names="Readmitted",
    hole=0.5,
    title="Readmission Distribution"
)

st.plotly_chart(fig_donut, use_container_width=True)

# ---------------- FUNNEL CHART ---------------- #

funnel = filtered_df["Social_Support_Level"].value_counts().reset_index()
funnel.columns = ["Support Level", "Count"]

fig_funnel = go.Figure(go.Funnel(
    y=funnel["Support Level"],
    x=funnel["Count"]
))

st.plotly_chart(fig_funnel, use_container_width=True)

# ---------------- MACHINE LEARNING MODEL ---------------- #

df["Gender_Encoded"] = df["Gender"].map({"Male":0,"Female":1}).fillna(0)

X = df[["Length_of_Stay","Social_Support_Level","Gender_Encoded"]]
y = df["Readmitted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))

st.sidebar.success(f"Model Accuracy: {acc:.2f}")

# ---------------- REAL TIME PREDICTION ---------------- #

st.subheader("Predictive Risk Panel")

c1, c2, c3 = st.columns(3)

stay = c1.number_input("Length of Stay", 1, 60, 5)
support = c2.selectbox("Social Support Level", [0,1,2])
gender_input = c3.selectbox("Gender", ["Male","Female"])

gender_encoded = 0 if gender_input == "Male" else 1

if st.button("Run Prediction"):

    input_df = pd.DataFrame({
        "Length_of_Stay":[stay],
        "Social_Support_Level":[support],
        "Gender_Encoded":[gender_encoded]
    })

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"High Risk — Probability: {prob:.2f}")
    else:
        st.success(f"Low Risk — Probability: {prob:.2f}")
