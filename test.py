import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model
model = joblib.load('best_model_ml_new.pkl')

# Set up the Streamlit app
st.set_page_config(page_title="System Performance Predictor", layout="wide")

st.title("System Performance Predictor")
st.write("""
This app uses a trained machine learning model to predict system performance metrics 
based on various input parameters. Adjust the sliders to set your system parameters 
and see the predicted performance.
""")

# Create input sliders in the sidebar
st.sidebar.header("Input Parameters")

def user_input_features():
    cpu_usage = st.sidebar.slider('CPU Usage (%)', 0.0, 100.0, 50.0)
    cpu_user = st.sidebar.slider('CPU User (%)', 0.0, 100.0, 30.0)
    cpu_system = st.sidebar.slider('CPU System (%)', 0.0, 100.0, 20.0)
    cpu_idle = st.sidebar.slider('CPU Idle (%)', 0.0, 100.0, 50.0)
    memory_usage_percent = st.sidebar.slider('Memory Usage (%)', 0.0, 100.0, 50.0)
    memory_used_gb = st.sidebar.slider('Memory Used (GB)', 0.0, 64.0, 16.0)
    memory_available_gb = st.sidebar.slider('Memory Available (GB)', 0.0, 64.0, 16.0)
    swap_usage_percent = st.sidebar.slider('Swap Usage (%)', 0.0, 100.0, 10.0)
    gpu_load_percent = st.sidebar.slider('GPU Load (%)', 0.0, 100.0, 30.0)
    gpu_memory_usage_percent = st.sidebar.slider('GPU Memory Usage (%)', 0.0, 100.0, 30.0)
    network_bytes_sent_mb = st.sidebar.slider('Network Bytes Sent (MB)', 0.0, 1000.0, 100.0)
    network_bytes_recv_mb = st.sidebar.slider('Network Bytes Received (MB)', 0.0, 1000.0, 100.0)
    active_connections = st.sidebar.slider('Active Connections', 0, 10000, 100)
    
    data = {
        'cpu_usage': cpu_usage,
        'cpu_user': cpu_user,
        'cpu_system': cpu_system,
        'cpu_idle': cpu_idle,
        'memory_usage_percent': memory_usage_percent,
        'memory_used_gb': memory_used_gb,
        'memory_available_gb': memory_available_gb,
        'swap_usage_percent': swap_usage_percent,
        'gpu_load_percent': gpu_load_percent,
        'gpu_memory_usage_percent': gpu_memory_usage_percent,
        'network_bytes_sent_mb': network_bytes_sent_mb,
        'network_bytes_recv_mb': network_bytes_recv_mb,
        'active_connections': active_connections
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the input parameters
st.subheader('Input Parameters')
st.write(input_df)

# Make predictions
prediction = model.predict(input_df)

# Display predictions
st.subheader('Prediction')
st.write(f"The predicted system performance metric is: {prediction[0]:.2f}")

# Add some explanations
st.markdown("""
### About the Model
- The model was trained using various machine learning algorithms including Random Forest, Gradient Boosting, and XGBoost.
- The best performing model was selected based on cross-validation performance.
- Feature importance analysis was conducted to understand which parameters most affect system performance.
""")

