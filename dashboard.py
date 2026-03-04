import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="AI Recovery Dashboard", layout="wide")

# -------------------------------------------------
# LOGIN SYSTEM
# -------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Secure Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "NHS2026":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("rhs_ml_dataset.csv")

if not os.path.exists("rhs_ml_dataset.csv"):
    st.error("Dataset not found.")
    st.stop()

df = load_data()

# -------------------------------------------------
# PREPROCESS DATE
# -------------------------------------------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

df["Year"] = df["date"].dt.year
df["Month"] = df["date"].dt.month
df["Date"] = df["date"].dt.date

# -------------------------------------------------
# CREATE AGGREGATED DATASET
# -------------------------------------------------
agg_df = df.groupby(["Year", "Month", "gender"]).agg(
    Total_Patients=("gender", "count"),
    Total_Readmissions=("readmitted", "sum"),
    Avg_Age=("age", "mean")
).reset_index()

agg_df["Readmission_Rate"] = (
    agg_df["Total_Readmissions"] / agg_df["Total_Patients"]
) * 100

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("Filters")

years = sorted(agg_df["Year"].unique())
months = sorted(agg_df["Month"].unique())
genders = sorted(agg_df["gender"].unique())

selected_year = st.sidebar.selectbox("Year", years)
selected_month = st.sidebar.selectbox("Month", months)
selected_gender = st.sidebar.selectbox("Gender", genders)

filtered = agg_df[
    (agg_df["Year"] == selected_year) &
    (agg_df["Month"] == selected_month) &
    (agg_df["gender"] == selected_gender)
]

# -------------------------------------------------
# DASHBOARD TITLE
# -------------------------------------------------
st.title("AI-Powered Post-Discharge Recovery Dashboard")

# -------------------------------------------------
# KPI SECTION
# -------------------------------------------------
if not filtered.empty:
    row = filtered.iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", int(row["Total_Patients"]))
    col2.metric("Total Readmissions", int(row["Total_Readmissions"]))
    col3.metric("Readmission Rate (%)", round(row["Readmission_Rate"], 2))
else:
    st.warning("No data available for selected filters.")
    st.stop()

st.divider()

# -------------------------------------------------
# BAR CHART - MONTHLY PATIENTS
# -------------------------------------------------
fig_bar = px.bar(
    agg_df,
    x="Month",
    y="Total_Patients",
    color="gender",
    title="Monthly Patients by Gender"
)
st.plotly_chart(fig_bar, use_container_width=True)

# -------------------------------------------------
# DONUT CHART - READMISSION DISTRIBUTION
# -------------------------------------------------
fig_donut = px.pie(
    names=["Readmitted", "Not Readmitted"],
    values=[
        row["Total_Readmissions"],
        row["Total_Patients"] - row["Total_Readmissions"]
    ],
    hole=0.5,
    title="Readmission Distribution"
)
st.plotly_chart(fig_donut, use_container_width=True)

# -------------------------------------------------
# FUNNEL CHART
# -------------------------------------------------
fig_funnel = go.Figure(go.Funnel(
    y=["Total Patients", "Readmitted", "Recovered"],
    x=[
        row["Total_Patients"],
        row["Total_Readmissions"],
        row["Total_Patients"] - row["Total_Readmissions"]
    ]
))
st.plotly_chart(fig_funnel, use_container_width=True)

# -------------------------------------------------
# DAILY VISITS CHART (FROM RAW DATA)
# -------------------------------------------------
daily_df = df.groupby("Date").size().reset_index(name="Patients")

fig_line = px.line(
    daily_df,
    x="Date",
    y="Patients",
    title="Daily Patient Visits"
)
st.plotly_chart(fig_line, use_container_width=True)

st.divider()

# -------------------------------------------------
# MACHINE LEARNING PROTOTYPE
# -------------------------------------------------
st.header("Predictive AI Prototype")

# Encode gender
le = LabelEncoder()
agg_df["gender_encoded"] = le.fit_transform(agg_df["gender"])

X = agg_df[["Year", "Month", "gender_encoded", "Total_Patients", "Avg_Age"]]
y = agg_df["Readmission_Rate"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

gender_encoded = le.transform([selected_gender])[0]

prediction_input = pd.DataFrame([[
    selected_year,
    selected_month,
    gender_encoded,
    row["Total_Patients"],
    row["Avg_Age"]
]], columns=X.columns)

predicted_rate = model.predict(prediction_input)[0]

st.subheader("Predicted Readmission Rate")
st.success(f"{round(predicted_rate, 2)} %")

st.success("System Running Successfully")
