import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Titanic Survival Predictor | ANN",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Premium Design ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    * {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background: radial-gradient(circle at top right, #1e1e2f, #111119);
        color: #ffffff;
    }

    .stApp {
        background-color: transparent;
    }

    /* Glassmorphism containers */
    div.stBlock {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 2rem;
    }

    /* Sections */
    .section-container {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 20px;
        border-left: 5px solid #764ba2;
    }

    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    .prediction-card {
        text-align: center;
        padding: 20px;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .status-survived {
        color: #00ff88;
        font-weight: bold;
        font-size: 24px;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }

    .status-died {
        color: #ff4b4b;
        font-weight: bold;
        font-size: 24px;
        text-shadow: 0 0 10px rgba(255, 75, 75, 0.5);
    }

    /* Button styling */
    .stButton>button {
        background: var(--primary-gradient);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.4);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(118, 75, 162, 0.6);
    }

    /* Input styling */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="slider"] > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }
    
    label {
        color: #bbbbbb !important;
        font-weight: 500 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Model and Scaler ---
@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model('model/titanic_ann_model.h5')
        scaler = joblib.load('model/scaler.pkl')
        return model, scaler

    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

model, scaler = load_assets()

# --- SECTION 1: Header Area ---
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# 🚢 Titanic Survival Prediction System")
        st.markdown("### *Deep Learning Based Passenger Survival Prediction*")
    with col2:
        # Re-using the emoji since image generation was canceled, but styled it
        st.markdown("""
            <div style="text-align: center; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 50%;">
                <span style="font-size: 80px;">🧬</span>
            </div>
        """, unsafe_allow_html=True)




# --- SECTION 2: Project Description ---
st.markdown("""
<div class="section-container">
    <h2>Project Overview</h2>
    <p>This intelligent system utilizes an <b>Artificial Neural Network (ANN)</b> built with <b>TensorFlow</b> to predict the survival probability of Titanic passengers. 
    By analyzing historical patterns in passenger class, age, and fare, the deep learning model identifies critical survival factors with high precision.</p>
</div>
""", unsafe_allow_html=False)

st.markdown("---")

# --- SECTION 3: Passenger Input Form ---
st.markdown("## 📋 Passenger Information")
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pclass = st.selectbox(
            "Passenger Class", 
            options=[1, 2, 3],
            help="1st = Upper, 2nd = Middle, 3rd = Lower"
        )
        
    with col2:
        age = st.slider(
            "Age (Years)", 
            min_value=1, 
            max_value=100, 
            value=25
        )
        
    with col3:
        fare = st.number_input(
            "Fare Paid ($)", 
            min_value=0.0, 
            max_value=600.0, 
            value=32.0,
            step=1.0
        )

# --- SECTION 4: Prediction Button ---
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🚀 Predict Survival Status")

# --- Logic and Output ---
if predict_btn:
    if model and scaler:
        # Preprocessing (Task 4)
        input_features = np.array([[pclass, age, fare]])
        input_scaled = scaler.transform(input_features)
        
        # Prediction (Task 5 & 6)
        with st.spinner('Analysing passenger data...'):
            time.sleep(0.5) # For micro-animation effect
            prediction_prob = model.predict(input_scaled)[0][0]
            
        is_survived = prediction_prob > 0.5
        confidence = float(prediction_prob) if is_survived else 1.0 - float(prediction_prob)
        
        # --- SECTION 5: Prediction Output Area ---
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        o_col1, o_col2, o_col3 = st.columns(3)
        
        with o_col1:
            res_text = "SURVIVED" if is_survived else "NOT SURVIVED"
            status_class = "status-survived" if is_survived else "status-died"
            st.markdown(f"""
                <div class="prediction-card">
                    <p style="margin-bottom: 0;">PREDICTION</p>
                    <p class="{status_class}">{res_text}</p>
                </div>
            """, unsafe_allow_html=True)
            
        with o_col2:
            st.metric("Survival Probability", f"{float(prediction_prob)*100:.2f}%")
            
        with o_col3:
            st.metric("Confidence Score", f"{confidence*100:.2f}%")

        # --- SECTION 6: Visualization Area ---
        st.markdown("<br>", unsafe_allow_html=True)
        col_viz, col_desc = st.columns([2, 1])
        
        with col_viz:
            # Probability Meter (Task 6 Visualization)
            labels = ['Survival Probability', 'Mortality Risk']
            values = [float(prediction_prob), 1.0 - float(prediction_prob)]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels, 
                values=values, 
                hole=.6,
                marker_colors=['#00ff88', '#ff4b4b'],
                textinfo='percent+label'
            )])
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False,
                margin=dict(t=0, b=0, l=0, r=0),
                height=350
            ) 
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col_desc:
            st.markdown("""
                <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; height: 100%;">
                    <h4>Insight</h4>
                    <p>The ANN model suggests that passengers in higher classes with specific age groups and higher fares had significantly different survival rates.</p>
                    <p style="font-size: 0.8rem; color: #888;">Note: This result is based on the trained weights of a 3-layer Artificial Neural Network.</p>
                </div>
            """, unsafe_allow_html=True)

    else:
       st.error(
    "Model or Scaler not found in /model directory. "
    "Please ensure 'titanic_ann_model.h5' and 'scaler.pkl' are present."
)

st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: #555; font-size: 0.8rem;">
        &copy; 2024 Titanic Survival Prediction System | Powered by TensorFlow & Streamlit
    </div>
""", unsafe_allow_html=True)
