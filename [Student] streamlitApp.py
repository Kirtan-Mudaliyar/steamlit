import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import gdown
import os

# Set page configuration
st.set_page_config(page_title="Timelytics", layout="wide")

# Title and captions
st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")

st.caption(
    "Timelytics is an ensemble model that utilizes XGBoost, Random Forests, and SVM to accurately forecast Order to Delivery (OTD) times."
)

st.caption(
    "With Timelytics, businesses can identify potential bottlenecks and delays in their supply chain and take proactive measures to address them."
)

# Define model path
MODEL_PATH = "voting_model.pkl"
MODEL_URL = "https://drive.google.com/uc?id=1F8iQDIV8OZovlfupRaVzQ-lfV-_F3JoG"  # Corrected direct download link

# Function to download model
def download_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    if not os.path.exists(MODEL_PATH):  # Double-check if download was successful
        st.error("Failed to download the model. Please check the link.")

# Load the trained ensemble model
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)  # Use joblib.load() instead of joblib.dump()
    except Exception as e:
        st.error(f"Error loading the model: {e}. Please check the file format.")
        return None

# Download and load model
download_model()
voting_model = load_model()

# Wait time predictor function
def waitime_predictor(
    purchase_dow, purchase_month, year, product_size_cm3,
    product_weight_g, geolocation_state_customer,
    geolocation_state_seller, distance):
    if voting_model is None:
        st.error("Model failed to load. Cannot make predictions.")
        return None
    
    prediction = voting_model.predict(
        np.array([[purchase_dow, purchase_month, year,
                   product_size_cm3, product_weight_g,
                   geolocation_state_customer,
                   geolocation_state_seller, distance]])
    )
    return round(prediction[0])

# Sidebar inputs
with st.sidebar:
    try:
        img = Image.open("./assets/supply_chain_optimisation.jpg")
        st.image(img)
    except FileNotFoundError:
        st.warning("Image not found: supply_chain_optimisation.jpg")
    
    st.header("Input Parameters")
    purchase_dow = st.number_input("Purchased Day of the Week", min_value=0, max_value=6, step=1, value=3)
    purchase_month = st.number_input("Purchased Month", min_value=1, max_value=12, step=1, value=1)
    year = st.number_input("Purchased Year", value=2018)
    product_size_cm3 = st.number_input("Product Size in cm^3", value=9328)
    product_weight_g = st.number_input("Product Weight in grams", value=1800)
    geolocation_state_customer = st.number_input("Geolocation State of the Customer", value=10)
    geolocation_state_seller = st.number_input("Geolocation State of the Seller", value=20)
    distance = st.number_input("Distance", value=475.35)

# Output container
with st.container():
    st.header("Output: Wait Time in Days")
    if st.button("Predict Wait Time"):
        prediction = waitime_predictor(
            purchase_dow, purchase_month, year,
            product_size_cm3, product_weight_g,
            geolocation_state_customer, geolocation_state_seller, distance)
        with st.spinner(text="This may take a moment..."):
            if prediction is not None:
                st.write(prediction)

# Sample dataset
sample_data = {
    "Purchased Day of the Week": [0, 3, 1],
    "Purchased Month": [6, 3, 1],
    "Purchased Year": [2018, 2017, 2018],
    "Product Size in cm^3": [37206.0, 63714, 54816],
    "Product Weight in grams": [16250.0, 7249, 9600],
    "Geolocation State Customer": [25, 25, 25],
    "Geolocation State Seller": [20, 7, 20],
    "Distance": [247.94, 250.35, 4.915],
}

df = pd.DataFrame(sample_data)

st.header("Sample Dataset")
st.write(df)
