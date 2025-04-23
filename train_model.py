# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load and preprocess the dataset
def preprocess_data(path):
    df = pd.read_csv(path, on_bad_lines='skip', low_memory=False)

    # Drop rows with missing critical values
    df = df.dropna(subset=['Latitude', 'Longitude', 'Primary Type', 'Date'])

    # Parse date to extract features
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Hour'] = df['Date'].dt.hour
    df['Day'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month

    features = df[['Latitude', 'Longitude', 'Hour', 'Day', 'Month']]
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['Primary Type'])

    return train_test_split(features, labels, test_size=0.2, random_state=42), label_encoder

# Preprocess the data
(X_train, X_test, y_train, y_test), label_encoder = preprocess_data('data/chicago_crime.csv')

# Build the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(5,)),
    Dense(64, activation='relu'),
    Dense(len(set(y_train)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Save the model and label encoder
model.save('model/crime_model.h5')
joblib.dump(label_encoder, 'model/label_encoder.pkl')
