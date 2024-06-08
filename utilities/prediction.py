import streamlit as st
import pickle
import numpy as np

def add_predictions(input_data):
    model = pickle.load(open("./Model/model_RF.pkl", "rb"))
    scaler = pickle.load(open("./Model/scaler.pkl", "rb"))
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    input_array_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_array_scaled)
    prob_benign = model.predict_proba(input_array_scaled)[0][0]
    prob_malignant = model.predict_proba(input_array_scaled)[0][1]
    
    st.subheader("Cell Cluster Prediction")
    
    if prediction[0] == 0:
        diagnosis = "Benign"
        color = "#4CAF50"
    else:
        diagnosis = "Malignant"
        color = "#F44336"
    
    st.markdown(f"""
    <div style="display: flex; justify-content: center;">
        <div style="padding: 20px; border-radius: 10px; background-color: #f0f0f0; text-align: center;">
            <h2 style="color: {color};">The cell cluster is: <strong>{diagnosis}</strong></h2>
            <p style="font-size: 18px; color: #333;">
                <strong>Probability of being benign:</strong> {prob_benign:.2f} <br>
                <strong>Probability of being malignant:</strong> {prob_malignant:.2f}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding: 10px; border-radius: 10px; background-color: #fff8e1; text-align: center; font-size: 14px; color: #6c757d;">
        <em>This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.</em>
    </div>
    """, unsafe_allow_html=True)