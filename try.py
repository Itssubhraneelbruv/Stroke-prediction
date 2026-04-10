import streamlit as st
import pandas as pd
import joblib

model = joblib.load("stroke_model.pkl")
model_columns = joblib.load("stroke_columns.pkl")

st.set_page_config(page_title="Stroke Risk Predictor", page_icon="🩺")

st.title("Stroke Risk Predictor")
st.write("Enter patient details below.")

gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=0.0, max_value=120.0, value=45.0, step=1.0)

hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox(
    "Work Type",
    ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
)
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.0, step=0.1)
bmi = st.number_input("BMI", min_value=0.0, value=25.0, step=0.1)
smoking_status = st.selectbox(
    "Smoking Status",
    ["never smoked", "formerly smoked", "smokes", "Unknown"]
)

user_input = {
    "gender": gender,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "ever_married": ever_married,
    "work_type": work_type,
    "Residence_type": residence_type,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": smoking_status,
}

input_df = pd.DataFrame([user_input])

# match notebook preprocessing
input_df["ever_married"] = input_df["ever_married"].map({"No": 0, "Yes": 1})
input_df["Residence_type"] = input_df["Residence_type"].map({"Rural": 0, "Urban": 1})
input_df["smoking_status"] = input_df["smoking_status"].replace("Unknown", "missing")

input_df = pd.get_dummies(input_df, columns=["gender"], drop_first=True)
input_df = pd.get_dummies(input_df, columns=["smoking_status"], drop_first=True)
input_df = pd.get_dummies(input_df, columns=["work_type"], drop_first=True)

input_df = input_df.astype(float)
input_df = input_df.reindex(columns=model_columns, fill_value=0)

st.subheader("Captured Input")
st.write(user_input)

st.subheader("Processed Input DataFrame")
st.dataframe(input_df)

if st.button("Predict Stroke Risk"):
    pred_prob = model.predict_proba(input_df)[:, 1][0]
    pred_class = int(pred_prob > 0.2)

    st.subheader("Prediction Result")
    st.write(f"Stroke risk probability: **{pred_prob:.2%}**")
    st.write(f"Predicted class at threshold 0.2: **{pred_class}**")

    if pred_class == 1:
        st.warning("Higher predicted stroke risk.")
    else:
        st.success("Lower predicted stroke risk.")

st.caption("This is an educational ML demo and not medical advice.")