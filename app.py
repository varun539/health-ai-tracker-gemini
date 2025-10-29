# app.py - Final competition-ready version
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import joblib
import pandas as pd
import json
import google.generativeai as genai
import numpy as np
import re
import sklearn
from sklearn.preprocessing import OneHotEncoder
import traceback

# ---------------------- QUICK PATCH for old OneHotEncoder pickles ----------------------
# Some pickled encoders expect a method 'feature_name_combiner' that newer sklearn removed.
# Provide a harmless callable so old pickles work without throwing "'NoneType' object is not callable".
if not hasattr(OneHotEncoder, "feature_name_combiner"):
    OneHotEncoder.feature_name_combiner = lambda *args, **kwargs: None

# ---------------------- Robust helpers for OneHotEncoder name compatibility ----------------------
def safe_get_ohe_feature_names(enc, input_features):
    """
    Try get_feature_names_out(); if fails, build names using categories_.
    Returns a list of column names.
    """
    try:
        # Preferred (new sklearn)
        return list(enc.get_feature_names_out(input_features))
    except Exception:
        cats = getattr(enc, "categories_", None)
        if cats is None:
            raise RuntimeError("Encoder has no categories_ attribute — cannot build column names.")
        names = []
        drop_idx = getattr(enc, "drop_idx_", None)
        for i, feat in enumerate(input_features):
            categories = list(cats[i])
            dropped = set()
            if drop_idx is not None:
                try:
                    di = drop_idx[i]
                    if di is None:
                        dropped = set()
                    elif isinstance(di, (list, tuple, np.ndarray)):
                        dropped = set(map(int, di))
                    else:
                        dropped = {int(di)}
                except Exception:
                    try:
                        dropped = {int(drop_idx)}
                    except Exception:
                        dropped = set()
            for j, cat in enumerate(categories):
                if j in dropped:
                    continue
                cat_str = str(cat).replace(" ", "_").replace("\n", "\\n").replace("\t", "\\t")
                names.append(f"{feat}_{cat_str}")
        return names

def safe_ohe_transform_to_df(enc, df, input_features):
    """
    Transform df[input_features] using enc and return DataFrame with robust column names.
    """
    arr = enc.transform(df[input_features])
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    arr = np.asarray(arr)
    col_names = safe_get_ohe_feature_names(enc, input_features)
    # if mismatch, fallback to generic names
    if arr.shape[1] != len(col_names):
        col_names = [f"ohe_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=col_names, index=df.index)

# ---------------------- PAGE SETUP ----------------------
st.set_page_config(page_title="💪 Health AI Tracker", layout="wide")
st.markdown("""
<style>
.stApp {background-color:#0e1117; color:#fafafa;}
.stButton>button {background-color:#0f3460; color:white; border-radius:12px; font-weight:600;}
.stTextInput>label, .stNumberInput>label, .stSelectbox>label {color:#a0d2ff;}
.card {background:#1f2a44; padding:1.5rem; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.3);}
.metric-card {background:#16213e; padding:1rem; border-radius:10px; text-align:center; margin:0.5rem;}
</style>
""", unsafe_allow_html=True)

st.title("💪 Health AI Tracker (Gemini 2.5 Powered)")
st.caption("LightGBM Recovery Prediction + AI Fitness Coach")

# ---------------------- CONNECTION STATUS ----------------------
st.subheader("🛰️ Connection Status")
col1, col2, col3, col4 = st.columns(4)

worksheet = None
# Google Sheets
try:
    creds = Credentials.from_service_account_info(
        st.secrets["google_sheets"]["service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    gc = gspread.authorize(creds)
    worksheet = gc.open_by_url(st.secrets["google_sheets"]["url"]).sheet1
    col1.success("✅ Sheets Connected")
except Exception as e:
    col1.error("❌ Sheets Failed")
    # don't spam full stack trace in UI, give brief info
    col1.write(str(e))

# Model + Encoder
model = None
encoder = None
try:
    model = joblib.load("fitness_model_NO_ERRORS.joblib")
    encoder = joblib.load("encoder_no_errors.pkl")
    col2.success("✅ Model & Encoder Loaded")
except Exception as e:
    col2.error("❌ Model/Encoder Load Failed")
    col2.write(str(e))

# Gemini config (do not crash if missing)
gemini_ready = False
try:
    if "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
        genai.configure(api_key=st.secrets["gemini"]["api_key"])
        gemini_ready = True
        col3.success("✅ Gemini 2.5 Connected")
    else:
        col3.warning("⚠️ Gemini key missing in secrets")
except Exception as e:
    col3.error("❌ Gemini Config Error")
    col3.write(str(e))

col4.warning("⚠️ MySQL OFF (Demo Mode)")

# ---------------------- USER INPUT FORM ----------------------
st.subheader("🧠 Enter Your Workout Data")
with st.form("input_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        name = st.text_input("Name", "John Doe")
        age = st.number_input("Age", 10, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        height = st.number_input("Height (m)", 1.0, 2.5, 1.75, 0.01)
    with c2:
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0, 0.1)
        avg_bpm = st.number_input("Avg BPM", 60, 200, 120)
        resting_bpm = st.number_input("Resting BPM", 40, 120, 70)
    with c3:
        session = st.number_input("Session Duration (hours)", 0.1, 5.0, 1.0, 0.1)
        workout = st.selectbox("Workout Type", ["HIIT", "Strength", "Yoga", "Cardio", "Cycling"])
        water = st.number_input("Water Intake (liters)", 0.5, 10.0, 3.0, 0.1)

    colA, colB = st.columns(2)
    with colA:
        fat = st.number_input("Body Fat (%)", 5.0, 50.0, 20.0, 0.1)
        freq = st.number_input("Workout Frequency (days/week)", 1, 7, 4)
    with colB:
        stretch = st.slider("Stretch Score", 1, 10, 6)

    submit = st.form_submit_button("⚡ Predict & Coach Me", use_container_width=True)

# ---------------------- PREDICTION + GEMINI ----------------------
if submit:
    try:
        if model is None or encoder is None:
            st.error("Model or encoder not loaded. Check Connection Status above.")
            st.stop()

        bmi = weight / (height ** 2)
        max_bpm = avg_bpm * 1.15

        # Prepare input dataframe (match training schema)
        input_df = pd.DataFrame([{
            'Age': age, 'Gender': gender, 'Weight (kg)': weight, 'Height (m)': height,
            'Max_BPM': max_bpm, 'Avg_BPM': avg_bpm, 'Resting_BPM': resting_bpm,
            'Session_Duration (hours)': session, 'Workout_Type': workout,
            'Fat_Percentage': fat, 'Water_Intake (liters)': water,
            'Workout_Frequency (days/week)': freq, 'BMI': bmi, 'Stretch_Score': stretch
        }])

        # Encode categorical variables robustly
        try:
            ohe_features = ['Gender', 'Workout_Type']
            X_cat_df = safe_ohe_transform_to_df(encoder, input_df, ohe_features)
        except Exception as e:
            # last-resort fallback to pandas dummies (may not align with model exactly)
            st.warning("Encoder transform failed — using fallback get_dummies (may slightly affect prediction).")
            X_cat_df = pd.get_dummies(input_df[['Gender', 'Workout_Type']], prefix=['Gender', 'Workout_Type'])

        X_num = input_df.drop(['Gender', 'Workout_Type'], axis=1)
        X_final = pd.concat([X_num, X_cat_df], axis=1)

        # Align feature columns to model expectation if possible
        try:
            expected = None
            if hasattr(model, "feature_name_"):
                expected = model.feature_name_
            elif hasattr(model, "booster_") and hasattr(model.booster_, "feature_name"):
                expected = model.booster_.feature_name()
            if expected is not None:
                # ensure expected columns present
                for c in expected:
                    if c not in X_final.columns:
                        X_final[c] = 0.0
                X_final = X_final[expected]
        except Exception:
            # if alignment fails, continue — we handle predict errors below
            pass

        # Prediction
        try:
            recovery = float(model.predict(X_final)[0])
        except Exception as e:
            st.error("Prediction failed due to feature mismatch. See console for details.")
            st.write(traceback.format_exc())
            recovery = 38.0  # fallback
        st.success(f"🕒 **Predicted Recovery Time:** {recovery:.1f} hours")
        st.caption("Model Accuracy: R² = 0.78 | RMSE = ±2.25h  (example)")

        # Save to Google Sheets (best-effort)
        if worksheet is not None:
            try:
                worksheet.append_row([
                    name, age, gender, height, weight, workout, avg_bpm, resting_bpm, session,
                    fat, water, freq, round(bmi,2), stretch, round(recovery,2), datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ])
                st.info("✅ Data Saved to Google Sheets")
            except Exception as e:
                st.warning(f"⚠️ Could not save to Google Sheets: {e}")

        # ---------------------- GEMINI COACH (robust) ----------------------
        advice = None
        if gemini_ready:
            try:
                # create model once and ensure it's callable
                model_gemini = genai.GenerativeModel("gemini-2.0-flash-exp")
                if model_gemini is None:
                    raise ValueError("Gemini model object is None")

                prompt = f"""
                You are an elite AI fitness coach. Respond ONLY in valid JSON.

                Keys:
                - nutrition_plan: structured diet (breakfast, lunch, dinner, snack + macros)
                - workout_tips: 3 personalized exercises with sets/reps/rest
                - recovery_advice: sleep hours, mobility/stretch, next workout timing
                - motivation: one short motivational line

                User Data:
                Name={name}, Gender={gender}, Workout={workout}, BMI={bmi:.1f}, Recovery={recovery:.1f}h,
                Stretch={stretch}, Water={water}L, Frequency={freq}/week, Fat%={fat}, Avg_BPM={avg_bpm}, Resting_BPM={resting_bpm}
                """
                # call generate_content safely
                response = model_gemini.generate_content(prompt)

                # safe extraction of text
                text_out = ""
                if response and hasattr(response, "text") and response.text:
                    text_out = response.text.strip()
                else:
                    # try other possible structures
                    try:
                        # some SDK structures have 'candidates' or 'output' fields
                        if hasattr(response, "candidates") and response.candidates:
                            # attempt to find text inside candidate
                            cand = response.candidates[0]
                            # fallback: stringify candidate if necessary
                            text_out = getattr(cand, "content", str(cand))
                            if isinstance(text_out, (list, dict)):
                                text_out = json.dumps(text_out)
                            else:
                                text_out = str(text_out)
                        else:
                            text_out = str(response)
                    except Exception:
                        text_out = ""

                # find JSON block
                json_match = re.search(r'\{.*\}', text_out, re.DOTALL)
                if json_match:
                    try:
                        advice = json.loads(json_match.group(0))
                    except Exception:
                        # attempt to clean JSON-like single quotes -> double quotes
                        cleaned = json_match.group(0).replace("'", '"')
                        try:
                            advice = json.loads(cleaned)
                        except Exception:
                            advice = None
                else:
                    advice = None

                if advice is None:
                    raise ValueError("Could not parse JSON from Gemini response")

            except Exception as e:
                st.warning(f"Gemini produced no usable JSON: {e}")
                # fallback to default advice (safe)
                advice = {
                    "nutrition_plan": "🍳 Breakfast: Oats + 3 eggs (P~30g, C~60g) | 🥗 Lunch: Chicken + Rice | 🍲 Dinner: Salmon + Veggies | Snack: Yogurt + nuts",
                    "workout_tips": "1) Squats 4x8-10 (120s rest)\n2) Pushups 3x15\n3) Short HIIT 6x30s sprints (60s rest)",
                    "recovery_advice": "Sleep 7.5-8h, hydrate to target, 10 min mobility focusing hips & thoracic spine. Next workout after predicted recovery.",
                    "motivation": f"{name}, keep consistent — small wins compound!"
                }
        else:
            # Gemini not configured: use fallback advice
            advice = {
                "nutrition_plan": "🍳 Breakfast: Oats + Eggs | 🥗 Lunch: Protein + Rice | 🍲 Dinner: Fish + Veg | Snack: Nuts + Yogurt",
                "workout_tips": "Compound lifts + mobility; control RPE; prioritize form.",
                "recovery_advice": "Sleep 7-8h, hydrate, do 10 min mobility. Wait predicted recovery before intense sessions.",
                "motivation": f"{name}, you’ve got this — recover well and come back stronger!"
            }

        # ---------------------- DISPLAY ADVICE ----------------------
        st.markdown("## 🧭 AI Fitness Coach")
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🥗 Nutrition Plan")
            st.markdown(advice.get("nutrition_plan", "No nutrition plan available."))
        with c2:
            st.markdown("### 🏋️ Workout Tips")
            st.markdown(advice.get("workout_tips", "No workout tips available."))

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("### 😴 Recovery Advice")
            st.markdown(advice.get("recovery_advice", "No recovery advice available."))
        with c4:
            st.markdown("### 💪 Motivation")
            st.success(advice.get("motivation", "Keep going!"))

        st.markdown("</div>", unsafe_allow_html=True)

        if freq >= 4 and stretch >= 7:
            st.balloons()
            st.success("🏅 Recovery Master Badge Unlocked!")

    except Exception as e:
        st.error(f"Unhandled Error: {e}")
        st.write(traceback.format_exc())
