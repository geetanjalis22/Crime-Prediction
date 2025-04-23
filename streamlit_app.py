# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the model and label encoder
model = load_model('model/crime_model.h5')
label_encoder = joblib.load('model/label_encoder.pkl')

st.set_page_config(page_title="Crime Prediction App", layout="centered")
st.title("Chicago Crime Prediction")
st.markdown("Enter details to predict the most likely crime.")

# Input form
with st.form("crime_form"):
    latitude = st.number_input("Latitude", value=41.89, format="%.6f")
    longitude = st.number_input("Longitude", value=-87.65, format="%.6f")
    hour = st.slider("Hour of Day", 0, 23, 12)
    day = st.selectbox("Day of the Week", list(range(7)), format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
    month = st.selectbox("Month", list(range(1,13)))
    submit = st.form_submit_button("Predict")

# Prediction
if submit:
    input_data = np.array([[latitude, longitude, hour, day, month]])
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)[0]
    crime_type = label_encoder.inverse_transform([predicted_class])[0]

    st.success(f"Predicted Crime: **{crime_type}**")
    st.bar_chart(prediction[0])
# Load and show map data
st.header("Chicago Crime Map")
if st.checkbox("Show crime locations on map"):
    df = pd.read_csv('data/chicago_crime.csv', on_bad_lines='skip', low_memory=False)
    df = df[['Latitude', 'Longitude', 'Primary Type']].dropna()
    
    # Display map using pydeck
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=41.8781,
            longitude=-87.6298,
            zoom=10,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=df,
                get_position='[Longitude, Latitude]',
                get_color='[200, 30, 0, 160]',
                get_radius=100,
            ),
        ],
    ))
