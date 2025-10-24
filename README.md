# 💪 Health AI Tracker (Gemini 2.5 Powered)

## Overview
Health AI Tracker is an innovative fitness application that predicts recovery time using a LightGBM model and provides personalized health advice powered by Gemini 2.5 AI. Designed for fitness enthusiasts, it saves workout data to MySQL and Google Sheets, offering smart insights for optimal performance and recovery.

- **Problem**: Track workouts and get tailored recovery/advice without manual guesswork.
- **Solution**: AI-driven recovery prediction (MSE ~213.44) and Gemini-powered nutrition/workout tips.
- **Innovation**: Integrates Fivetran for live data sync, Google Cloud for scalability.

## Our Journey
- **Grind**: 11 days battling GCP quota hell, optimizing LightGBM on GPU (Tesla T4), and syncing with MySQL/Sheets.
- **Challenges**: Overcame API limits, refined 15-column dataset (Age, Weight, etc.), and handled Gemini 2.5 JSON output.
- **Efforts**: Tried Fivetran Connector SDK for real-time BigQuery sync—partial success, fully functional with Sheets.

## Tech Stack
- **Frontend**: Streamlit
- **Machine Learning**: LightGBM (GPU-accelerated)
- **AI**: Gemini 2.5 (via Google Generative AI API)
- **Backend**: MySQL, Google Sheets
- **Deployment**: Streamlit Cloud
- **Data Sync**: Fivetran Connector SDK (ready for BigQuery)

## Features
- Input workout data (age, weight, BPM, etc.).
- Predict recovery time (e.g., 54.70 hours).
- Get AI-generated nutrition plans, workout tips, and recovery advice.
- Save data to MySQL and sync with Google Sheets.

