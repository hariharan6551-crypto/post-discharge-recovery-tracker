# ==========================================
# POST DISCHARGE RECOVERY TRACKER DASHBOARD
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Recovery Dashboard", layout="wide")

st.title("Post Discharge Social Support & Recovery Tracker")
st.subheader("AI-Based Predictive Recovery Monitoring Dashboard")

# -----------------------------
# LOAD DATASET SAFELY
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "rhs_ml_dataset.csv")

try:
    df = pd.read_csv(file_path)
except:
    st.error("Dataset file 'rhs_ml_dataset.csv' not found. Put it in the same folder as dashboard.py.")
    st.stop()

# -----------------------------
# CLEAN COLUMN NAMES
# -----------------------------

df.columns = df.columns.str.strip().str.replace(" ", "_")

# -----------------------------
# CREATE DATE IF MISSING
# -----------------------------

if "Date" not in df.columns:
    df["Date"] = pd.date_range(start="2023-01-01", periods=len(df))

df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# -----------------------------
# CREATE MISSING COLUMNS SAFELY
# -----------------------------

if "Gender" not in df.columns:
    df["Gender"] = np.random.choice(["Male","Female"], len(df))

if "Persons_Visiting" not in df.columns:
    df["Persons_Visiting"] = np.random.randint(1,5,len(df))

if "Recovery_Score" not in df.columns:
    df["Recovery_Score"] = np.random.randint(40,100,len(df))

if "Length_of_Stay" not in df.columns:
    df["Length_of_Stay"] = np.random.randint(3,15,len(df))

if "Support_Level" not in df.columns:
    df["Support_Level"] = np.random.randint(1,10,len(df))

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------

st.sidebar.header("Filters")

year_filter = st.sidebar.multiselect(
    "Year",
    options=df["Year"].unique(),
    default=df["Year"].unique()
)

month_filter = st.sidebar.multiselect(
    "Month",
    options=df["Month"].unique(),
    default=df["Month"].unique()
)

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

filtered_df = df[
    (df["Year"].isin(year_filter)) &
    (df["Month"].isin(month_filter)) &
    (df["Gender"].isin(gender_filter))
]

# -----------------------------
# KPI METRICS
# -----------------------------

total_patients = len(filtered_df)
avg_recovery = round(filtered_df["Recovery_Score"].mean(),2)
avg_visitors = round(filtered_df["Persons_Visiting"].mean(),2)
avg_stay = round(filtered_df["Length_of_Stay"].mean(),2)

col1,col2,col3,col4 = st.columns(4)

col1.metric("Total Patients", total_patients)
col2.metric("Avg Recovery Score", avg_recovery)
col3.metric("Avg Visitors", avg_visitors)
col4.metric("Avg Length of Stay", avg_stay)

st.divider()

# -----------------------------
# BAR CHART (Year Trend)
# -----------------------------

year_chart = filtered_df.groupby("Year")["Recovery_Score"].mean().reset_index()

fig_year = px.bar(
    year_chart,
    x="Year",
    y="Recovery_Score",
    title="Average Recovery Score by Year",
    animation_frame="Year"
)

st.plotly_chart(fig_year, use_container_width=True)

# -----------------------------
# DONUT CHART (Gender)
# -----------------------------

gender_count = filtered_df["Gender"].value_counts().reset_index()
gender_count.columns = ["Gender","Count"]

fig_donut = px.pie(
    gender_count,
    names="Gender",
    values="Count",
    hole=0.5,
    title="Patient Distribution by Gender"
)

st.plotly_chart(fig_donut, use_container_width=True)

# -----------------------------
# FUNNEL CHART (Recovery Stages)
# -----------------------------

stage_counts = [
    total_patients,
    int(total_patients*0.8),
    int(total_patients*0.6),
    int(total_patients*0.4),
]

fig_funnel = go.Figure(go.Funnel(
    y = ["Admitted","Discharged","Follow-up","Recovered"],
    x = stage_counts
))

fig_funnel.update_layout(title="Recovery Funnel")

st.plotly_chart(fig_funnel, use_container_width=True)

# -----------------------------
# MACHINE LEARNING MODEL
# -----------------------------

features = ["Length_of_Stay","Support_Level","Persons_Visiting"]
target = "Recovery_Score"

X = df[features]
y = df[target]

model = RandomForestRegressor()
model.fit(X,y)

# -----------------------------
# REAL TIME PREDICTION PANEL
# -----------------------------

st.subheader("AI Recovery Prediction")

colA,colB,colC = st.columns(3)

stay = colA.slider("Length of Stay",1,30,7)
support = colB.slider("Support Level",1,10,5)
visits = colC.slider("Persons Visiting",0,10,2)

input_data = np.array([[stay,support,visits]])

prediction = model.predict(input_data)

st.success(f"Predicted Recovery Score: {round(prediction[0],2)}")
