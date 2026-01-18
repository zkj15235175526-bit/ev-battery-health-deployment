import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("rf_battery_health_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("EV Battery Health Prediction System")

st.write("Input charging and battery parameters below:")

soc = st.slider("State of Charge (%)", 0, 100, 50)
voltage = st.number_input("Voltage (V)", 300.0, 500.0, 400.0)
current = st.number_input("Current (A)", 0.0, 300.0, 150.0)
battery_temp = st.slider("Battery Temperature (°C)", 0, 60, 30)
ambient_temp = st.slider("Ambient Temperature (°C)", 0, 45, 25)
duration = st.slider("Charging Duration (min)", 10, 300, 120)
degradation = st.slider("Degradation Rate (%)", 0, 50, 10)
efficiency = st.slider("Efficiency (%)", 50, 100, 90)
cycles = st.slider("Charging Cycles", 0, 2000, 500)

input_data = np.array([[soc, voltage, current, battery_temp,
                        ambient_temp, duration, degradation,
                        efficiency, cycles]])

if st.button("Predict Battery Health"):
    scaled_input = scaler.transform(input_data)
    health = model.predict(scaled_input)[0]

    st.subheader(f"Predicted Battery Health Index: {health:.2f}")

    if health < 60:
        st.error("Battery replacement is recommended.")
    else:
        st.success("Battery condition is acceptable.")

