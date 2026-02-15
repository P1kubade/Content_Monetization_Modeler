import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the pre-trained model and feature schema
@st.cache_resource
def load_model():
    try:
        with open('revenue_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file 'revenue_model.pkl' not found. You need to train the model first.")
        st.stop()

model_data = load_model()
model = model_data['model']
model_features = model_data['features']

# 2. Build the UI
st.title("YouTube Ad Revenue Predictor")
st.markdown("---")
st.write("Enter the video metrics below to estimate the potential ad revenue.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Engagement Metrics")
    views = st.number_input("Views", min_value=0.0, value=15000.0, step=1000.0)
    likes = st.number_input("Likes", min_value=0.0, value=1200.0, step=100.0)
    comments = st.number_input("Comments", min_value=0.0, value=250.0, step=10.0)
    watch_time = st.number_input("Watch Time (Minutes)", min_value=0.0, value=20000.0, step=1000.0)

with col2:
    st.subheader("Video & Channel Context")
    video_length = st.number_input("Video Length (Minutes)", min_value=0.0, value=12.5, step=0.5)
    subscribers = st.number_input("Subscribers", min_value=0.0, value=50000.0, step=1000.0)
    
    # Categorical Inputs
    category = st.selectbox("Category", ['Education', 'Entertainment', 'Gaming', 'Lifestyle', 'Music', 'Tech'])
    device = st.selectbox("Device", ['Desktop', 'Mobile', 'Tablet', 'TV'])
    country = st.selectbox("Country", ['AU', 'CA', 'DE', 'IN', 'UK', 'US'])

# 3. Prediction Logic
if st.button("Predict Revenue", type="primary"):
    # Create DataFrame from user inputs
    input_df = pd.DataFrame({
        'views': [views],
        'likes': [likes],
        'comments': [comments],
        'watch_time_minutes': [watch_time],
        'video_length_minutes': [video_length],
        'subscribers': [subscribers],
        'category': [category],
        'device': [device],
        'country': [country]
    })
    
    # One-Hot Encode the user input
    input_encoded = pd.get_dummies(input_df, columns=['category', 'device', 'country'])
    
    # The dummy variable trap and missing columns: 
    # The user only inputs ONE category, device, and country. 
    # We must rebuild the exact column structure the model was trained on.
    for col in model_features:
        if col not in input_encoded.columns:
            input_encoded[col] = False # Or 0 depending on your pandas version
            
    # Force the strict column order expected by the model
    input_encoded = input_encoded[model_features]
    
    # Make the prediction
    prediction = model.predict(input_encoded)[0]
    
    # Display the result
    st.markdown("---")
    st.subheader("Prediction Result")
    # Cap negative predictions at $0 in case someone inputs extreme dummy values
    final_revenue = max(0, prediction) 
    st.success(f"Estimated Ad Revenue: **${final_revenue:,.2f}**")