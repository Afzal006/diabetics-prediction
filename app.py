import streamlit as st
import pandas as pd
import joblib


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load("model/diabetes_model.pkl")


model = load_model()


# ==================================================
# TITLE
# ==================================================

st.title("🩺 Diabetes Prediction System")

st.write(
    "Enter the patient's information below to generate "
    "a machine-learning prediction."
)

st.info(
    "This application is for educational/project purposes "
    "and is not a medical diagnosis."
)


# ==================================================
# INPUT SECTION
# ==================================================

st.subheader("Patient Information")


col1, col2 = st.columns(2)


with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=300,
        value=120,
        step=1
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70,
        step=1
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20,
        step=1
    )


with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=900,
        value=80,
        step=1
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )


# ==================================================
# PREDICTION BUTTON
# ==================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Diabetes",
    use_container_width=True
)


# ==================================================
# PREDICTION
# ==================================================

if predict_button:

    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })

    # Convert zero values to missing values
    # for the same columns used during training

    columns_with_invalid_zero = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    input_data[columns_with_invalid_zero] = (
        input_data[columns_with_invalid_zero]
        .replace(0, float("nan"))
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probability = model.predict_proba(input_data)[0][1]

    probability_percentage = probability * 100


    # ==================================================
    # DISPLAY RESULT
    # ==================================================

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ Prediction: Diabetes detected"
        )

    else:

        st.success(
            "✅ Prediction: No diabetes detected"
        )


    st.metric(
        "Predicted Probability",
        f"{probability_percentage:.2f}%"
    )


    # ==================================================
    # PROBABILITY BAR
    # ==================================================

    st.progress(
        int(probability_percentage)
    )


    if probability < 0.30:

        st.success(
            "The model estimates a relatively low probability "
            "of diabetes."
        )

    elif probability < 0.70:

        st.warning(
            "The model estimates an intermediate probability "
            "of diabetes."
        )

    else:

        st.error(
            "The model estimates a relatively high probability "
            "of diabetes."
        )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Diabetes Prediction System | Machine Learning Project"
)