import numpy as np
import tensorflow as tf
from preprocess import preprocess_data

model = tf.keras.models.load_model('model/crime_model.h5')
_, _, _, _, label_encoder = preprocess_data('data/chicago_crime.csv')

def predict_crime(lat, lon, hour, day, month):
    input_data = np.array([[lat, lon, hour, day, month]])
    prediction = model.predict(input_data)
    crime_idx = np.argmax(prediction)
    crime_label = label_encoder.inverse_transform([crime_idx])[0]
    return crime_label

# Test
if __name__ == "__main__":
    print(predict_crime(41.8781, -87.6298, 14, 2, 4))
