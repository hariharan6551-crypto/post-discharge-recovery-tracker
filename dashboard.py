import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(layout="wide")

st.title("Post Discharge Recovery Tracker")

# ================= LOAD DATA =================

try:
    df = pd.read_csv("rhs_ml_dataset.csv")
except:
    st.error("Dataset file not found")
    st.stop()

# ================= CLEAN COLUMN NAMES =================

df.columns = df.columns.str.strip().str.lower()

# Try flexible matching
def find_column(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

age_col = find_column(["age","patient_age","age_years"])
stay_col = find_column(["length_of_stay","stay","los"])
support_col = find_column(["social_support_level","support_level"])
readmit_col = find_column(["readmitted","readmission"])
gender_col = find_column(["gender","sex"])

if not all([age_col, stay_col, support_col, readmit_col, gender_col]):
    st.error("Required columns not found in dataset")
    st.write("Available columns:", df.columns)
    st.stop()

# Add year if missing
if "year" not in df.columns:
    df["year"] = 2023

# ================= FILTERS =================

st.sidebar.header("Filters")

year = st.sidebar.selectbox("Year", sorted(df["year"].unique()))
gender = st.sidebar.selectbox("Gender", ["All"] + list(df[gender_col].unique()))

filtered = df[df["year"] == year]

if gender != "All":
    filtered = filtered[filtered[gender_col] == gender]

# ================= KPIs =================

col1,col2,col3 = st.columns(3)

col1.metric("Total Patients", len(filtered))

if len(filtered) > 0:
    col2.metric("Readmission %", round(filtered[readmit_col].mean()*100,2))
    col3.metric("Avg Stay", round(filtered[stay_col].mean(),1))
else:
    col2.metric("Readmission %", 0)
    col3.metric("Avg Stay", 0)

# ================= CHART =================

trend = df.groupby("year")[readmit_col].mean().reset_index()

fig = px.bar(trend,x="year",y=readmit_col,title="Yearly Readmission Trend")
st.plotly_chart(fig,use_container_width=True)

# ================= ML MODEL =================

df["gender_encoded"] = df[gender_col].astype(str).str.lower().map({"male":0,"female":1})

X = df[[age_col, stay_col, support_col, "gender_encoded"]]
y = df[readmit_col]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestClassifier()
model.fit(X_train,y_train)

accuracy = accuracy_score(y_test,model.predict(X_test))
st.sidebar.success(f"Model Accuracy: {round(accuracy,2)}")

# ================= PREDICTION PANEL =================

st.subheader("Predict Readmission Risk")

c1,c2,c3,c4 = st.columns(4)

age = c1.number_input("Age",18,100,50)
stay = c2.number_input("Length of Stay",1,60,5)
support = c3.number_input("Support Level (0-2)",0,2,1)
g = c4.selectbox("Gender",["Male","Female"])

g_enc = 0 if g.lower()=="male" else 1

if st.button("Predict"):
    input_df = pd.DataFrame({
        age_col:[age],
        stay_col:[stay],
        support_col:[support],
        "gender_encoded":[g_enc]
    })

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred==1:
        st.error(f"High Risk — Probability: {round(prob,2)}")
    else:
        st.success(f"Low Risk — Probability: {round(prob,2)}")
