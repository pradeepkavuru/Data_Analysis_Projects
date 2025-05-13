import streamlit as st
import pandas as pd
import pickle
import joblib
from streamlit_option_menu import option_menu

# Load Data
df = pd.read_csv(r"C:\Users\prade\Downloads\Multiple_disease_dataset.csv")

# Load Encoders
gender_encoding = {'Male': 1, 'Female': 0}

with open("disease_label_encoder.pkl",'rb') as f:
    disease_encoding = pickle.load(f)
    
with open('Medication_encoding.pkl', 'rb') as f:
    medication_encoding = pickle.load(f)

# Load Model
model = joblib.load(r"C:\Users\prade\Downloads\Disease_model.pkl")

# Sidebar Menu
with st.sidebar:
    selected = option_menu("Health Assistant", ["🏠 Home", "📊 Project Overview", "🩺 Predict Health Condition"],
        icons=["house", "bar-chart-line", "activity"], default_index=0)

# ========== HOME PAGE ==========
if selected == "🏠 Home":
    st.title("🧠 Health Risk Prediction Using ML")
    st.markdown("This intelligent system predicts **health outcomes**, suggests **medication**, and recommends a **nutrition diet** based on user inputs.")

    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.markdown("### 🔍 How it works")
        st.write("We use multiple machine learning models, including Logistic Regression, K-Nearest Neighbors, Support Vector Classifier, Decision Tree, Random Forest, Gaussian Naive Bayes, and XGBoost, to predict possible health conditions based on symptoms, vital signs, and demographic data.")
    with col2:
        st.image(r"C:\Users\prade\Downloads\1disease.jpg", use_container_width=True)

    st.markdown("---")
    st.markdown("### ✅ Key Factors Considered:")
    st.markdown("- **Age & Gender**")
    st.markdown("- **Vital Signs**: Blood Pressure, Sugar, Cholesterol, Heart Rate")
    st.markdown("- **Reported Symptoms**")
    st.markdown("- **Trained on diverse health data**")

# ========== PROJECT OVERVIEW ==========
elif selected == "📊 Project Overview":
    st.title("📁 Project Overview")
    st.markdown("This dataset contains anonymized patient health records used to train a predictive model.")

    st.subheader("🔬 Sample of Dataset")
    st.dataframe(df.head())

    st.markdown(f"📄 **Total Records**: {df.shape[0]} | **Features**: {df.shape[1]}")
    st.image(r"C:\Users\prade\Downloads\3disease.jpg", use_container_width=True)

    st.markdown("---")
    st.subheader("🤖 Machine Learning")
    st.markdown("- **Algorithm Used**: - Logistic Regression- K-Nearest Neighbors (KNN)- Support Vector Classifier (SVC)- Decision Tree Classifier- Random Forest Classifier- Gaussian Naive Bayes- XGBoost Classifier")
    st.markdown("- **Model Selected**:We used XGBoost model for Prediction")
    st.markdown("- Handles both numeric and categorical inputs")
    st.markdown("- Validated with strong performance metrics")
    


# ========== PREDICTION PAGE ==========
else:
    st.title("🩺 Predict Health Condition")
    st.markdown("Fill the following details to get predictions:")

    st.markdown("### 👤 Demographic Info")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=30)
    with col2:
        gender = st.selectbox("Gender", df.gender.unique())

    st.markdown("### 💓 Vitals")
    col3, col4, col5, col6 = st.columns(4)
    with col3:
        blood_pressure = st.number_input("Blood Pressure", min_value=70, max_value=210, value=120)
    with col4:
        cholesterol = st.number_input("Cholesterol", min_value=120, max_value=299, value=160)
    with col5:
        blood_sugar = st.number_input("Blood Sugar", min_value=70, max_value=200, value=100)
    with col6:
        heart_rate = st.number_input("Heart Rate", min_value=60, max_value=130, value=85)

    st.markdown("### 🤒 Symptoms")
    symptom_options = ['Pain', 'Fatigue', 'no symptoms', 'Cough']
    all_symptoms = ['Cough', 'Pain', 'Fatigue']
    symptoms_selected = st.multiselect("Select Symptom(s)", symptom_options)

    if "no symptoms" in symptoms_selected:
        selected_symptoms = []
    else:
        selected_symptoms = symptoms_selected

    symptom_encoded = {symptom: (1 if symptom in selected_symptoms else 0) for symptom in all_symptoms}

    if st.button("🔍 Predict Outcome", type="primary"):
        input_data = {
            'age': age,
            'gender': gender_encoding[gender],
            'blood_pressure': blood_pressure,
            'cholesterol': cholesterol,
            'blood_sugar': blood_sugar,
            'heart_rate': heart_rate,
            **symptom_encoded
        }

        input_row = pd.DataFrame([input_data])
        st.markdown("### 📝 Input Summary")
        st.dataframe(input_row)

        prediction = model.predict(input_row)[0]
        prediction1 = disease_encoding.inverse_transform([prediction])[0]
        st.markdown("### 🧬 Predicted Health Condition:")
        st.success(f"🩺 {prediction1}")

        medication_lookup = {
            "Diabetes": "Insulin, Metformin",
            "Hypertension": "Consult Physician",
            "Flu": "Oseltamivir",
            "Heart Disease": "Aspirin, Beta-Blockers",
            "Healthy": "No Medication Needed"
        }

        diet_lookup = {
            "Diabetes": "Low-carb, high-fiber diet.",
            "Hypertension": "Low-sodium, DASH diet.",
            "Flu": "Hydration and Vitamin C-rich foods.",
            "Heart Disease": "Heart-healthy, low-fat diet.",
            "Healthy": "Balanced diet with all nutrients."
        }

        st.markdown("### 💊 Suggested Medication:")
        st.info(medication_lookup.get(prediction1, "No info available."))

        st.markdown("### 🥗 Recommended Nutrition Diet:")
        st.warning(diet_lookup.get(prediction1, "No specific recommendation."))
