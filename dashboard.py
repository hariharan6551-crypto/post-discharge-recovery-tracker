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
    st.error("rhs_ml_dataset.csv file not found")
    st.stop()

# ================= CHECK COLUMNS =================

needed = ["Age","Length_of_Stay","Social_Support_Level","Readmitted","Gender"]

for col in needed:
    if col not in df.columns:
        st.error(f"Missing column: {col}")
        st.stop()

if "Year" not in df.columns:
    df["Year"] = 2023

# ================= FILTERS =================

st.sidebar.header("Filters")

year = st.sidebar.selectbox("Year", sorted(df["Year"].unique()))
gender = st.sidebar.selectbox("Gender", ["All"] + list(df["Gender"].unique()))

filtered = df[df["Year"] == year]

if gender != "All":
    filtered = filtered[filtered["Gender"] == gender]

# ================= KPIs =================

col1,col2,col3 = st.columns(3)

col1.metric("Patients", len(filtered))

if len(filtered) > 0:
    col2.metric("Readmission %", round(filtered["Readmitted"].mean()*100,2))
    col3.metric("Avg Stay", round(filtered["Length_of_Stay"].mean(),1))
else:
    col2.metric("Readmission %", 0)
    col3.metric("Avg Stay", 0)

# ================= CHART =================

trend = df.groupby("Year")["Readmitted"].mean().reset_index()

fig = px.bar(trend,x="Year",y="Readmitted",title="Yearly Trend")
st.plotly_chart(fig,use_container_width=True)

# ================= ML MODEL =================

df["Gender_Encoded"] = df["Gender"].map({"Male":0,"Female":1})

X = df[["Age","Length_of_Stay","Social_Support_Level","Gender_Encoded"]]
y = df["Readmitted"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestClassifier()
model.fit(X_train,y_train)

accuracy = accuracy_score(y_test,model.predict(X_test))
st.sidebar.success(f"Model Accuracy: {round(accuracy,2)}")

# ================= PREDICTION PANEL =================

st.subheader("Predict Risk")

c1,c2,c3,c4 = st.columns(4)

age = c1.number_input("Age",18,100,50)
stay = c2.number_input("Stay Days",1,60,5)
support = c3.selectbox("Support Level",[0,1,2])
g = c4.selectbox("Gender",["Male","Female"])

g_enc = 0 if g=="Male" else 1

if st.button("Predict"):
    input_df = pd.DataFrame({
        "Age":[age],
        "Length_of_Stay":[stay],
        "Social_Support_Level":[support],
        "Gender_Encoded":[g_enc]
    })
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred==1:
        st.error(f"High Risk: {round(prob,2)}")
    else:
        st.success(f"Low Risk: {round(prob,2)}")
