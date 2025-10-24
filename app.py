import streamlit as st
import mysql.connector
from datetime import datetime
import joblib
import pandas as pd
import json
import google.generativeai as genai
import joblib
import lightgbm as lgb



# ----------------------
# Page Setup
# ----------------------
st.set_page_config(page_title="💪 Health AI Tracker", layout="wide")
st.title("💪 Health AI Tracker (Gemini 2.5 Powered)")
st.caption("Smart Fitness & Recovery Prediction using Gemini 2.5 + LGBM + MySQL")

# ----------------------
# MySQL Connection
# ----------------------
st.subheader("🔌 Connection Status")
try:
    db = st.secrets["mysql"]
    connection = mysql.connector.connect(
        host=db["host"],
        user=db["user"],
        password=db["password"],
        database=db["database"],
        port=db["port"]
    )
    st.success("✅ MySQL Connected")
except Exception as e:
    st.error(f"❌ MySQL connection error: {e}")
    connection = None

# ----------------------
# Load Model
# # ----------------------
# try:
#     model = joblib.load("fitness_model.pkl")
#     st.success("✅ Model Loaded Successfully")
# except Exception as e:
#     st.error(f"❌ Model Load Error: {e}")
#     model = None
# Load the model safely
try:
    model = joblib.load("fitness_model.pkl")
    
    # If model isn’t a LightGBM Booster or sklearn estimator, reload properly
    if not hasattr(model, "predict"):
        print("⚠️ Model reloaded using LightGBM Booster interface")
        model = lgb.Booster(model_file="fitness_model.pkl")
        
    print("✅ Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error Loading Model: {e}")
    model = None

# ----------------------
# Gemini API Setup (2.5)
# ----------------------
try:
    genai.configure(api_key=st.secrets["gemini"]["api_key"])
    st.success("✅ Gemini 2.5 API Connected")
except Exception as e:
    st.error(f"❌ Gemini setup failed: {e}")

# ----------------------
# User Input
# ----------------------
st.subheader("🏋️ Enter Your Workout Data")

col1, col2, col3 = st.columns(3)
with col1:
    user_id = st.number_input("User ID", min_value=1, step=1, value=1)
    age = st.number_input("Age", 1, 120, 22)
    gender = st.selectbox("Gender", ["Male", "Female"])
    height = st.number_input("Height (m)", 0.5, 2.5, 1.75)
with col2:
    weight = st.number_input("Weight (kg)", 1.0, 300.0, 70.0)
    max_bpm = st.number_input("Max BPM", 50, 250, 150)
    avg_bpm = st.number_input("Avg BPM", 50, 250, 100)
with col3:
    resting_bpm = st.number_input("Resting BPM", 30, 150, 70)
    session_duration = st.number_input("Session Duration (hrs)", 0.1, 5.0, step=0.1, value=1.0)
    workout_type = st.selectbox("Workout Type", ["HIIT", "Strength", "Yoga", "Cardio"])

col4, col5 = st.columns(2)
with col4:
    fat_percentage = st.number_input("Fat Percentage", 1.0, 50.0, 20.0)
    water_intake = st.number_input("Water Intake (L)", 0.1, 10.0, step=0.1, value=3.0)
with col5:
    workout_freq = st.number_input("Workout Frequency (days/week)", 1, 7, 4)
    stretch_score = st.slider("Stretch Score", 0, 10, 5)

# ----------------------
# Save & Predict
# ----------------------
if st.button("💾 Save & Get AI Advice"):
    try:
        if model is None:
            st.error("❌ Model not loaded.")
        else:
            # Derived metrics
            bmi = round(weight / (height ** 2), 2)
            hydration_need = round(weight * 0.04, 2)
            gender_male = 1 if gender == "Male" else 0
            workout_dict = {"HIIT": [1, 0, 0, 0], "Strength": [0, 1, 0, 0],
                            "Yoga": [0, 0, 1, 0], "Cardio": [0, 0, 0, 1]}
            wt_hi, wt_strength, wt_yoga, wt_cardio = workout_dict[workout_type]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # ✅ Keep features consistent with model (adjust columns to your training)
            X_pred = pd.DataFrame([{
                'Age': age,
                'Weight (kg)': weight,
                'Height (m)': height,
                'Max_BPM': max_bpm,
                'Avg_BPM': avg_bpm,
                'Resting_BPM': resting_bpm,
                'Session_Duration (hours)': session_duration,
                'Fat_Percentage': fat_percentage,
                'Water_Intake (liters)': water_intake,
                'Workout_Frequency (days/week)': workout_freq,
                'BMI': bmi,
                'hydration_need': hydration_need,
                'Gender_Male': gender_male,
                'Workout_Type_HIIT': wt_hi,
                'Workout_Type_Strength': wt_strength,
                'Workout_Type_Yoga': wt_yoga,
                'Workout_Type_Cardio': wt_cardio,
                'stretch_score': stretch_score
            }])

            # ✅ Safe predict
            recovery_time = float(model.predict(X_pred, predict_disable_shape_check=True)[0])
            st.success(f"🔥 Predicted Recovery Time: {recovery_time:.2f} hours")

            # ✅ Save to MySQL
            if connection is not None:
                cursor = connection.cursor()
                cursor.execute("""
                    INSERT INTO workout_input (
                        user_id, Age, Weight_kg, Height_m, Max_BPM, Avg_BPM, Resting_BPM,
                        Session_Duration_hours, Gender, Workout_Type, Fat_Percentage,
                        Water_Intake_liters, Workout_Frequency_days_week, BMI,
                        Hydration_Need, Gender_Male, Workout_Type_HIIT,
                        Workout_Type_Strength, Workout_Type_Yoga, Workout_Type_Cardio,
                        Stretch_Score, timestamp, Recovery_Time
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                        Age=VALUES(Age), Weight_kg=VALUES(Weight_kg), Height_m=VALUES(Height_m),
                        Max_BPM=VALUES(Max_BPM), Avg_BPM=VALUES(Avg_BPM), Resting_BPM=VALUES(Resting_BPM),
                        Session_Duration_hours=VALUES(Session_Duration_hours), Gender=VALUES(Gender),
                        Workout_Type=VALUES(Workout_Type), Fat_Percentage=VALUES(Fat_Percentage),
                        Water_Intake_liters=VALUES(Water_Intake_liters), Workout_Frequency_days_week=VALUES(Workout_Frequency_days_week),
                        BMI=VALUES(BMI), Hydration_Need=VALUES(Hydration_Need), Gender_Male=VALUES(Gender_Male),
                        Workout_Type_HIIT=VALUES(Workout_Type_HIIT), Workout_Type_Strength=VALUES(Workout_Type_Strength),
                        Workout_Type_Yoga=VALUES(Workout_Type_Yoga), Workout_Type_Cardio=VALUES(Workout_Type_Cardio),
                        Stretch_Score=VALUES(Stretch_Score), Recovery_Time=VALUES(Recovery_Time),
                        timestamp=VALUES(timestamp)
                """, (
                    user_id, age, weight, height, max_bpm, avg_bpm, resting_bpm,
                    session_duration, gender, workout_type, fat_percentage,
                    water_intake, workout_freq, bmi, hydration_need, gender_male,
                    wt_hi, wt_strength, wt_yoga, wt_cardio, stretch_score,
                    timestamp, recovery_time
                ))
                connection.commit()
                st.success("✅ Data Saved to MySQL")

            # ----------------------
            # Gemini AI Advice
            # ----------------------
            prompt = f"""
            You are a professional AI fitness coach.
            Based on this data:
            User ID: {user_id}, Age: {age}, Gender: {gender}, Workout: {workout_type}, Duration: {session_duration} hrs,
            BMI: {bmi}, Hydration Need: {hydration_need}L, Recovery Time: {recovery_time:.2f} hrs,
            Workout Frequency: {workout_freq}/week, Stretch Score: {stretch_score}.
            Give advice in JSON with:
            - nutrition_plan
            - workout_tips
            - recovery_advice
            - motivation
            """
            model_gem = genai.GenerativeModel("gemini-2.5-flash")
            response = model_gem.generate_content(prompt)
            text = response.text

            try:
                report = json.loads(text)
            except:
                report = {
                    "nutrition_plan": text,
                    "workout_tips": "",
                    "recovery_advice": "",
                    "motivation": "Keep pushing forward 💪"
                }

            st.markdown("## 🧠 Gemini AI Health Advice")
            st.markdown(f"### 🥗 Nutrition Plan\n{report.get('nutrition_plan', 'N/A')}")
            st.markdown(f"### 🏋️ Workout Tips\n{report.get('workout_tips', 'N/A')}")
            st.markdown(f"### 😴 Recovery Advice\n{report.get('recovery_advice', 'N/A')}")
            st.markdown(f"### 💪 Motivation\n{report.get('motivation', 'Keep going!')}")

    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")



