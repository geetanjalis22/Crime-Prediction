import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(path):
    df = pd.read_csv(path)

    df = df.dropna(subset=['Latitude', 'Longitude', 'Primary Type', 'Date'])

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month

    le = LabelEncoder()
    df['CrimeLabel'] = le.fit_transform(df['Primary Type'])

    features = df[['Latitude', 'Longitude', 'Hour', 'DayOfWeek', 'Month']]
    labels = df['CrimeLabel']

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, le
