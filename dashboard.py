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

# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():
    return pd.read_csv("rhs_ml_dataset.csv")

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# =========================
# REQUIRED COLUMNS CHECK
# =========================

required_columns = [
    "Age",
    "Length_of_Stay",
    "Social_Support_Level",
    "Readmitted",
    "Gender"
]

for col in required_columns:
    if col not in df.columns:
        st.error(f"Dataset missing required column: {col}")
        st.stop()

# Add Year column if not available
if "Year" not in df.columns:
    df["Year"] = 2023

# =========================
# SIDEBAR FILTER
# =========================

st.sidebar.header("Filter Panel")

selected_year = st.sidebar.selectbox(
    "Select Year",
    sorted(df["Year"].unique())
)

selected_gender = st.sidebar.selectbox(
    "Select Gender",
    ["All"] + sorted(df["Gender"].unique())
)

filtered_df = df[df["Year"] == selected_year]

if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]

# =========================
# KPI SECTION
# =========================

st.subheader("Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Patients", len(filtered_df))

if len(filtered_df) > 0:
    col2.metric(
        "Readmission Rate (%)",
        f"{filtered_df['Readmitted'].mean()*100:.2f}"
    )
    col3.metric(
        "Avg Length of Stay",
        f"{filtered_df['Length_of_Stay'].mean():.1f} Days"
    )
else:
    col2.metric("Readmission Rate (%)", "0")
    col3.metric("Avg Length of Stay", "0")

# =========================
# ANIMATED BAR CHART
# =========================

trend = df.groupby("Year")["Readmitted"].mean().reset_index()

fig_bar = px.bar(
    trend,
    x="Year",
    y="Readmitted",
    animation_frame="Year",
    range_y=[0, 1],
    title="Yearly Readmission Trend"
)

st.plotly_chart(fig_bar, use_container_width=True)

# =========================
# DONUT CHART
# =========================

if len(filtered_df) > 0:
    fig_donut = px.pie(
        filtered_df,
        names="Readmitted",
        hole=0.5,
        title="Readmission Distribution"
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# =========================
# FUNNEL CHART
# =========================

if len(filtered_df) > 0:
    funnel_data = (
        filtered_df["Social_Support_Level"]
        .value_counts()
        .reset_index()
    )
    funnel_data.columns = ["Support Level", "Count"]

    fig_funnel = go.Figure(go.Funnel(
        y=funnel_data["Support Level"],
        x=funnel_data["Count"]
    ))

    st.plotly_chart(fig_funnel, use_container_width=True)

# =========================
# MACHINE LEARNING MODEL
# =========================

df["Gender_Encoded"] = df["Gender"].map({"Male": 0, "Female": 1})

X = df[[
    "Age",
    "Length_of_Stay",
    "Social_Support_Level",
    "Gender_Encoded"
]]

y = df["Readmitted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
st.sidebar.success(f"Model Accuracy: {accuracy:.2f}")

# =========================
# PREDICTIVE PANEL
# =========================

st.subheader("Predictive Risk Panel")

colA, colB, colC, colD = st.columns(4)

age = colA.number_input("Age", 18, 100, 50)
stay = colB.number_input("Length of Stay (Days)", 1, 60, 5)
support = colC.selectbox("Social Support Level", [0, 1, 2])
gender_input = colD.selectbox("Gender", ["Male", "Female"])

gender_encoded = 0 if gender_input == "Male" else 1

if st.button("Run Prediction"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Length_of_Stay": [stay],
        "Social_Support_Level": [support],
        "Gender_Encoded": [gender_encoded]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"High Readmission Risk — Probability: {probability:.2f}")
    else:
        st.success(f"Low Readmission Risk — Probability: {probability:.2f}")
