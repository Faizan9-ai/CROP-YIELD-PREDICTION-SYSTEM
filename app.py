

from flask import Flask,request, render_template # type: ignore
import numpy as np # type: ignore
import pickle
import sklearn # type: ignore
print(sklearn.__version__)
#loading models
import pickle

dtr = pickle.load(open(r"C:\Users\ADEEBSAYEED\Downloads\dtr.pkl", "rb"))
preprocessor = pickle.load(open(r"C:\Users\ADEEBSAYEED\Downloads\preprocessor.pkl", "rb"))


#flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item  = request.form['Item']

        features = np.array([[Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item]],dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1,-1)

        return render_template('index.html',prediction = prediction)

if __name__=="__main__":
    app.run(debug=True)
