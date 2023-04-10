import pyttsx3
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from flask import Flask, render_template, request


app = Flask(__name__)


excel = pd.read_csv('crop.csv')
crop_names = excel['CROP'].tolist()

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


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(features, crop)


def speak(audio):
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate-20)
    engine.setProperty('voice', voices[0].id)
    engine.say(audio)
    engine.runAndWait()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    nitrogen = float(request.form['nitrogen'])
    phosphorus = float(request.form['phosphorus'])
    potassium = float(request.form['potassium'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    prediction = knn.predict(
        [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    crop_name = le.inverse_transform(prediction)[0]
    speak("Sir according to the data that you provided to me. The ratio of nitrogen in the soil is  " + str(nitrogen) + ". The ratio of phosphorus in the soil is  " + str(phosphorus) + ". The ratio of potassium in the soil is  " + str(potassium) + ". The temperature level around the field is  " +
          str(temperature) + ". The humidity level around the field is  " + str(humidity) + ". The ph type of the soil is  " + str(ph) + ". The amount of rainfall is  " + str(rainfall) + ".The best crop that you can grow is  " + crop_name)
    return render_template("index.html", prediction_text="The best crop to grow is {}".format(crop_name))


if __name__ == '__main__':
    app.run(debug=True)
