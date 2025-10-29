# 💪 Health AI Tracker (Gemini 2.5 Powered)

**LightGBM Recovery Prediction + AI Fitness Coach**

---

## 🧠 Overview

**Health AI Tracker** is an intelligent fitness assistant that predicts your body’s **recovery time** and provides a **personalized AI-generated fitness plan** using **LightGBM** and **Google Gemini 2.5**.

It combines **machine learning precision** with **AI-driven coaching** — helping users recover smarter, eat cleaner, and train more effectively.

---

## ⚡ Problem Statement

Most people who work out don’t know:
- How long they should rest before the next workout  
- What exact foods or routines improve recovery  

Existing fitness apps only track calories or steps — they **don’t understand your body’s actual recovery response**.  
This project solves that problem by predicting recovery and generating a **personalized AI plan**.

---

## 🧩 Solution

The app has two intelligent components:

1. **Recovery Time Predictor (LightGBM Model)**  
   Predicts how many hours your body needs to recover based on:
   - Heart Rate (Avg & Resting)
   - Workout Type
   - Water Intake
   - Body Fat %
   - Frequency of Workouts
   - Stretching Habits

2. **AI Fitness Coach (Gemini 2.5)**  
   Generates real-time, structured JSON output including:
   - 🥗 Nutrition Plan (meals + macros)
   - 🏋️ Workout Tips (sets, reps, rest)
   - 😴 Recovery Advice (sleep, stretching, timing)
   - 💪 Motivation (personalized quote)

---

## ⚙️ Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend | Streamlit |
| ML Model | LightGBM |
| AI Text Generation | Google Gemini 2.5 |
| Data Storage | Google Sheets API |
| Data Encoding | OneHotEncoder (sklearn) |
| Deployment | Streamlit Cloud |

---

## 🧱 Architecture Flow

1️⃣ User enters fitness data in the Streamlit UI  
2️⃣ Data is preprocessed → encoded → passed to the LightGBM model  
3️⃣ Predicted **Recovery Time** is displayed  
4️⃣ The same data is sent to **Gemini 2.5**, which generates full AI advice  
5️⃣ Data is saved automatically to **Google Sheets**










---

## 🌍 Real-World Use Cases

- **Fitness centers**: Automated recovery tracking  
- **Sports teams**: Athlete recovery & nutrition suggestions  
- **Health startups**: AI wellness assistants  
- **Personal use**: Daily fitness optimization

---

## 💎 Unique Features

✅ Combines **LightGBM analytics** + **Gemini creativity**  
✅ Saves all logs to **Google Sheets**  
✅ Built with **Streamlit**, simple and powerful UI  
✅ Real-time personalized JSON-based health plans  
✅ Ready for **deployment on Streamlit Cloud**

---

## 🧠 How to Run Locally



1️⃣ Install dependencies
pip install streamlit pandas joblib gspread google-generativeai lightgbm scikit-learn

# 2️⃣ Add your credentials in .streamlit/secrets.toml
[google_sheets]
service_account = {...}
url = "https://docs.google.com/spreadsheets/..."

[gemini]
api_key = "YOUR_GEMINI_API_KEY"

3️⃣ Run the app
streamlit run app.py




🚀 Deployment

This app runs perfectly on Streamlit Cloud.



You can also link your GitHub repo → Streamlit Cloud directly.

Steps:

Push your code to GitHub

Go to streamlit.io/cloud

Deploy 🎯

🧍 Author

👨‍💻 Varun B
B.Tech Data Science | Lovely Professional University
Aspiring Data Scientist & AI Biohacker







