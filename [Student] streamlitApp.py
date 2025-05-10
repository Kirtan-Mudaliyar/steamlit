import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

st.title("📦 Order Delivery Time Prediction")

model_path = "order_delivery_model.pkl"

# Load model
if not os.path.exists(model_path):
    st.error("Model file not found! Please upload 'order_delivery_model.pkl'.")
    st.stop()

try:
    rf, xgb = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

features = [
    "purchase_dow", "purchase_month", "year", "product_size_cm3", "product_weight_g", 
    "geolocation_state_customer", "geolocation_state_seller", "distance"
]

def ensemble_predict(X):
    X_df = pd.DataFrame(X, columns=features)
    rf_pred = rf.predict(X_df)
    xgb_pred = xgb.predict(X_df)
    return (rf_pred + xgb_pred) / 2

# Sidebar image
st.sidebar.header("🔢 Input Parameters")
image_path = "supply_chain_optimisation.jpg"
if os.path.exists(image_path):
    image = Image.open(image_path)
    st.sidebar.image(image, caption="Supply Chain Optimization", use_container_width=True)

# Inputs
purchase_dow = st.sidebar.slider("Day of Week", 0, 6, 2)
purchase_month = st.sidebar.slider("Month", 1, 12, 6)
year = st.sidebar.selectbox("Year", [2017, 2018])
product_size_cm3 = st.sidebar.slider("Product Size (cm³)", 0.0, 50000.0, 9328.0)
product_weight_g = st.sidebar.slider("Product Weight (g)", 0.0, 30000.0, 1800.0)
geolocation_state_customer = st.sidebar.slider("Customer State Code", 1, 27, 10)
geolocation_state_seller = st.sidebar.slider("Seller State Code", 1, 27, 20)
distance = st.sidebar.slider("Distance (km)", 0.0, 3000.0, 475.35)

def predict_wait_time():
    try:
        # Clip overly large values
        clipped_weight = min(product_weight_g, 30000)
        clipped_size = min(product_size_cm3, 50000)
        clipped_distance = min(distance, 3000)

        if product_weight_g > 30000:
            st.warning("⚠️ Product weight clipped to 30,000g for safety.")
        if product_size_cm3 > 50000:
            st.warning("⚠️ Product size clipped to 50,000 cm³.")
        if distance > 3000:
            st.warning("⚠️ Distance clipped to 3000 km.")

        input_data = [[
            purchase_dow, purchase_month, year, clipped_size, clipped_weight,
            geolocation_state_customer, geolocation_state_seller, clipped_distance
        ]]

        prediction = ensemble_predict(input_data)
        return round(prediction[0])

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        return None

if st.sidebar.button("🚀 Predict Wait Time"):
    with st.spinner("Predicting..."):
        result = predict_wait_time()
    if result is not None:
        st.sidebar.success(f"### ⏳ Predicted Delivery Time: **{result} days**")
