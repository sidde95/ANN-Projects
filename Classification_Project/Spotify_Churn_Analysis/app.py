import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import pandas as pd
import warnings

# Suppress warnings from scikit-learn regarding feature names/future behavior
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
st.set_page_config(page_title="Spotify Churn Predictor", layout="centered")

# Preprocessing Model - 
with open('Classification_Project/Spotify_Churn_Analysis/processing.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

# Load the Keras model
model = load_model('Classification_Project/Spotify_Churn_Analysis/model.h5')

# --- STREAMLIT UI ---
st.title("🎧 Spotify Churn Prediction")
st.write("Predict whether a subscriber will churn based on their profile and usage metrics.")

# User Input Form
with st.form("spotify_churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Subscriber Profile")
        gender = st.selectbox("Gender", ['Female', 'Male', 'Other'])
        age = st.number_input("Age", min_value=16, max_value=80, value=30, step=1)
        country = st.selectbox("Country", ['US', 'CA', 'UK', 'DE', 'AU', 'FR', 'IN', 'PK', 'Other'])
        subscription_type = st.selectbox("Subscription Type", ['Free', 'Premium', 'Family', 'Student'])
        offline_listening = st.selectbox("Offline Listening Enabled?", [1, 0])

    with col2:
        st.subheader("Usage Metrics")
        listening_time = st.number_input("Daily Listening Time (minutes)", min_value=0, max_value=300, value=150, step=5)
        songs_played_per_day = st.number_input("Songs Played Per Day", min_value=0, max_value=100, value=50, step=1)
        skip_rate = st.slider("Average Skip Rate", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        device_type = st.selectbox("Primary Device Type", ['Mobile', 'Web', 'Desktop'])
        ads_listened_per_week = st.number_input("Ads Listened Per Week", min_value=0, max_value=100, value=5, step=1)

    submit = st.form_submit_button("Predict Churn")

# Prediction Logic
if submit:
    # 1. Create input DataFrame
    input_data = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'country': country,
        'subscription_type': subscription_type,
        'listening_time': listening_time,
        'songs_played_per_day': songs_played_per_day,
        'skip_rate': skip_rate,
        'device_type': device_type,
        'ads_listened_per_week': ads_listened_per_week,
        'offline_listening': offline_listening
    }])

    # Ensure feature order matches the training data/preprocessor expectation
    feature_order = [
        'gender', 'age', 'country', 'subscription_type', 'listening_time',
        'songs_played_per_day', 'skip_rate', 'device_type', 'ads_listened_per_week',
        'offline_listening'
    ]
    input_data = input_data[feature_order]

    # 2. Apply preprocessing
    processed_input = preprocessor.transform(input_data)

    # 3. Make prediction
    # Keras model predicts probability directly
    prediction_prob = model.predict(processed_input, verbose=0)[0][0]
    churn_prob_percent = prediction_prob * 100

    # Binary classification (using 0.5 threshold)
    churn_pred = 1 if prediction_prob >= 0.5 else 0

    # 4. Result Display (Simplified)
    st.subheader("Prediction Result")
    st.write(f"Churn Probability: **{churn_prob_percent:.2f}%**")

    if churn_pred == 1:
        st.error("Subscriber is likely to churn!")
    else:
        st.success("Subscriber is likely to stay.")