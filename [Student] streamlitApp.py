import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import os

# Set page configuration
st.set_page_config(page_title="Timelytics", layout="wide")

# Title and captions
st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")

st.caption(
    "Timelytics is an ensemble model that utilizes XGBoost, Random Forests, and SVM to accurately forecast Order to Delivery (OTD) times.")

st.caption(
    "With Timelytics, businesses can identify potential bottlenecks and delays in their supply chain and take proactive measures to address them.")

# Sidebar: Upload Model File
st.sidebar.header("Upload Model File")
uploaded_model = st.sidebar.file_uploader("Upload Model File (voting_model.pkl)", type=["pkl"])

# Load the trained ensemble model
@st.cache_resource
def load_model(uploaded_file):
    if uploaded_file is not None:
        try:
            with open("temp_model.pkl", "wb") as f:
                f.write(uploaded_file.read())  # Save file temporarily
            return joblib.load("temp_model.pkl")  # Load model
        except Exception as e:
            st.error(f"Error loading the model: {e}. Please upload a valid model file.")
            return None
    else:
        return None

# Load model only if uploaded
voting_model = load_model(uploaded_model)

# Wait time predictor function
def waitime_predictor(
    purchase_dow, purchase_month, year, product_size_cm3,
    product_weight_g, geolocation_state_customer,
    geolocation_state_seller, distance):
    
    if voting_model is None:
        st.error("Model failed to load. Cannot make predictions.")
        return None
    
    try:
        prediction = voting_model.predict(
            np.array([[purchase_dow, purchase_month, year,
                       product_size_cm3, product_weight_g,
                       geolocation_state_customer,
                       geolocation_state_seller, distance]])
        )
        return round(prediction[0])
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Sidebar Inputs
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

    # Display default image if exists, otherwise allow user upload
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

# Prediction Output Container
with st.container():
    st.header("Output: Wait Time in Days")
    if st.button("Predict Wait Time"):
        prediction = waitime_predictor(
            purchase_dow, purchase_month, year,
            product_size_cm3, product_weight_g,
            geolocation_state_customer, geolocation_state_seller, distance)
        
        if prediction is not None:
            with st.spinner("Predicting..."):
                st.subheader(f"Estimated Delivery Time: **{prediction} days**")

# Sample Dataset
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
