# Chicago Crime Prediction Project

This project predicts the type of crime likely to occur at a specific location, time, and date in Chicago using a Deep Learning model. It also visualizes crime data on an interactive map using Streamlit.

## Features

- Predict the type of crime based on:
  - Latitude & Longitude
  - Hour of Day
  - Day of Week
  - Month
- Visualize historical crimes on an interactive map
- Deep Learning model using TensorFlow
- Deployed as a Streamlit web application

## Model Overview

- Neural Network with input features:
- Latitude, Longitude, Hour, Day, Month
- Label-encoded crime categories (`Primary Type`)
- Saved using `TensorFlow` and `joblib` for later inference

## Dataset

The dataset used is `Chicago_Crimes_2001_to_2004.csv`, available from the City of Chicago Data Portal.

Make sure to rename it to `chicago_crime.csv` and place it in the `data/` folder.

## Future Improvements

- Add heatmap or clustering on map
- Include NLP-based crime description analysis
- Extend to other cities
- Integrate with live data APIs

## Technologies & Concepts Used
## Machine Learning & Deep Learning
TensorFlow/Keras – for building and training the neural network.

Label Encoding – transforming categorical crime labels into numerical values.

Model Serialization – saving the trained model (.h5) and label encoder (.pkl) for reuse.

## Data Preprocessing & Analysis
Pandas – for loading and processing CSV data.

NumPy – for numerical operations and array handling.

Datetime handling – extracting features like hour, day of week, and month.

## Data Visualization & Mapping
Streamlit – building a lightweight interactive web app.

Pydeck / Deck.GL – for rendering crime data on an interactive map.

Mapbox – for map tiles and geolocation rendering (via Streamlit integration).

## File Structure & Organization
Modular Python files:

preprocess.py: handles data loading and preprocessing.

train_model.py: trains and saves the ML model.

streamlit_app.py: handles UI and predictions.

Separation of:

Data (data/chicago_crime.csv)

Model Artifacts (model/)

## Environment Management
Virtual Environment (venv) – for dependency isolation.

requirements.txt – to specify all project dependencies for easy setup.

## Deployment
Streamlit Deployment
