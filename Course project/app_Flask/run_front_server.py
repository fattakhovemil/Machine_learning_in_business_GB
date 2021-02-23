import json
import urllib.request

from flask import Flask, render_template, redirect, url_for, request
from flask_wtf import FlaskForm
from requests.exceptions import ConnectionError
from wtforms import IntegerField, FloatField
from wtforms.validators import DataRequired

from Features import Features

features = Features.features

class ClientDataForm(FlaskForm):
    age = FloatField(features[0], validators=[DataRequired()])
    anaemia =IntegerField(features[1], validators=[DataRequired()])
    creatinine_phosphokinase = IntegerField(features[2], validators=[DataRequired()])
    diabetes = IntegerField(features[3], validators=[DataRequired()])
    ejection_fraction = IntegerField(features[4], validators=[DataRequired()])
    high_blood_pressure = IntegerField(features[5], validators=[DataRequired()])
    platelets = FloatField(features[6], validators=[DataRequired()])
    serum_creatinine = FloatField(features[7], validators=[DataRequired()])
    serum_sodium = IntegerField(features[8], validators=[DataRequired()])
    sex = IntegerField(features[9], validators=[DataRequired()])
    smoking = IntegerField(features[10], validators=[DataRequired()])
    time = IntegerField(features[11], validators=[DataRequired()])


app = Flask(__name__)
app.config.update(
    CSRF_ENABLED=True,
    SECRET_KEY='you-will-never-guess',
)

def get_prediction(data):
    body = {feature : data[feature] for feature in features}

    myurl = "http://localhost:5000/predict"
    req = urllib.request.Request(myurl)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(body)
    jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
    req.add_header('Content-Length', len(jsondataasbytes))
    response = urllib.request.urlopen(req, jsondataasbytes)
    return json.loads(response.read())['predictions']



@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predicted/<response>')
def predicted(response):
    response = json.loads(response)
    print(response)
    return render_template('predicted.html', response=response)


@app.route('/predict_form', methods=['GET', 'POST'])
def predict_form():
    form = ClientDataForm()
    data = dict()
    if request.method == 'POST':
        for feature in features:
            data[feature] = request.form.get(feature)


        try:
            response = str(get_prediction(data))
            print(response)
        except ConnectionError:
            response = json.dumps({"error": "ConnectionError"})
        return redirect(url_for('predicted', response=response))
    return render_template('form.html', form=form)


if __name__ == '__main__':
    app.run(host='localhost', port=5001, debug=True)