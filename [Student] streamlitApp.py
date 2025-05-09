import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import os
import gdown

# Set page configuration
st.set_page_config(page_title="Timelytics", layout="wide")

# App title and caption
st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")
st.caption(
    "Timelytics is an ensemble model that utilizes XGBoost, Random Forests, and SVM to accurately forecast Order to Delivery (OTD) times.")
st.caption(
    "With Timelytics, businesses can identify potential bottlenecks and delays in their supply chain and take proactive measures to address them.")

# Download model from Google Drive (only once and cache it)
@st.cache_resource
def load_model_from_gdrive():
    model_path = "voting_model.pkl"
    if not os.path.exists(model_path):
        # Use fuzzy=True to allow link-based downloads
        gdown.download(
            "https://drive.google.com/file/d/1F8iQDIV8OZovlfupRaVzQ-lfV-_F3JoG/view?usp=sharing",
            model_path,
            fuzzy=True,
            quiet=False
        )
    return joblib.load(model_path)

# Load model
voting_model = load_model_from_gdrive()

# Prediction function
def waitime_predictor(
    purchase_dow, purchase_month, year, product_size_cm3,
    product_weight_g, geolocation_state_customer,
    geolocation_state_seller, distance):
    
    try:
        prediction = voting_model.predict(
            np.array([[purchase_dow, purchase_month, year,
                       product_size_cm3, product_weight_g,
                       geolocation_state_customer,
                       geolocation_state_seller, distance]])
        )
        return round(prediction[0])
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Sidebar inputs
with st.sidebar:
    st.header("Input Parameters")
    purchase_dow = st.number_input("Purchased Day of the Week", min_value=0, max_value=6, step=1, value=3)
    purchase_month = st.number_input("Purchased Month", min_value=1, max_value=12, step=1, value=1)
    year = st.number_input("Purchased Year", value=2018)
    product_size_cm3 = st.number_input("Product Size in cm³", value=9328)
    product_weight_g = st.number_input("Product Weight in grams", value=1800)
    geolocation_state_customer = st.number_input("Geolocation State of the Customer", value=10)
    geolocation_state_seller = st.number_input("Geolocation State of the Seller", value=20)
    distance = st.number_input("Distance", value=475.35)

    # Show image
    st.header("Supply Chain Visualization")
    image_path = "supply_chain_optimisation.jpg"
    if os.path.exists(image_path):
        img = Image.open(image_path)
        st.image(img, caption="Supply Chain Optimization", use_column_width=True)
    else:
        uploaded_image = st.file_uploader("Upload an Image (JPG/PNG)", type=["jpg", "png"])
        if uploaded_image:
            img = Image.open(uploaded_image)
            st.image(img, caption="Uploaded Image", use_column_width=True)

# Output prediction
with st.container():
    st.header("Output: Wait Time in Days")
    if st.button("Predict Wait Time"):
        with st.spinner("Predicting..."):
            prediction = waitime_predictor(
                purchase_dow, purchase_month, year,
                product_size_cm3, product_weight_g,
                geolocation_state_customer, geolocation_state_seller, distance)
            if prediction is not None:
                st.subheader(f"Estimated Delivery Time: **{prediction} days**")

# Sample dataset
sample_data = {
    "Purchased Day of the Week": [0, 3, 1],
    "Purchased Month": [6, 3, 1],
    "Purchased Year": [2018, 2017, 2018],
    "Product Size in cm³": [37206.0, 63714, 54816],
    "Product Weight in grams": [16250.0, 7249, 9600],
    "Geolocation State Customer": [25, 25, 25],
    "Geolocation State Seller": [20, 7, 20],
    "Distance": [247.94, 250.35, 4.915],
}

df = pd.DataFrame(sample_data)
st.header("Sample Dataset")
st.write(df)
