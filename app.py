import os
import numpy as np
import pandas as pd
import pyttsx3
from flask import Flask, render_template, request
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load dataset
excel = pd.read_csv('crop.csv')
crop_names = excel['CROP'].tolist()

# Encode crop names using LabelEncoder
le = preprocessing.LabelEncoder()
crop = le.fit_transform(crop_names)
nitrogen = excel['NITROGEN'].tolist()
phosphorus = excel['PHOSPHORUS'].tolist()
potassium = excel['POTASSIUM'].tolist()
temperature = excel['TEMPERATURE'].tolist()
humidity = excel['HUMIDITY'].tolist()
ph = excel['PH'].tolist()
rainfall = excel['RAINFALL'].tolist()
features = list(zip(nitrogen, phosphorus, potassium,
                    temperature, humidity, ph, rainfall))
features = np.array(features)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(features, crop)


def speak(audio):
    """Text-to-speech conversion using pyttsx3"""
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate-20)
    engine.setProperty('voice', voices[0].id)
    engine.say(audio)
    engine.runAndWait()


@app.route('/')
def home():
    """Render homepage"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Predict crop based on input features"""
    nitrogen = float(request.form['nitrogen'])
    phosphorus = float(request.form['phosphorus'])
    potassium = float(request.form['potassium'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Make prediction using KNN classifier
    prediction = knn.predict(
        [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    crop_name = le.inverse_transform(prediction)[0]

    # Convert text-to-speech and render prediction
    speak(
        f"According to the data you provided, the best crop to grow is {crop_name}")
    return render_template("index.html", prediction_text=f"The best crop to grow is {crop_name}")


if __name__ == '__main__':
    # Listen on appropriate port for deployment
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
