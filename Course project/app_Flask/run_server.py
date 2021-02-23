import dill
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import os
from Features import Features
dill._dill._reverse_typemap['ClassType'] = type
import flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime

# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def load_model(model_path):
	# load the pre-trained model
	global model
	with open(model_path, 'rb') as f:
		model = dill.load(f)

modelpath = '.././Models/heart_featured.dill'
load_model(modelpath)


@app.route("/", methods=["GET"])
def general():
	return """HEART FAILURE PREDICTION APP. This app predicts the likelihood of a person having an **Heart Attack** .
	Please use 'http://<address>/predict' to POST"""

features = Features.features

col = ['age','creatinine_phosphokinase','ejection_fraction',
	   'platelets','serum_creatinine','serum_sodium', 'time',
	   'anaemia_0', 'anaemia_1', 'diabetes_0', 'diabetes_1',
	   'high_blood_pressure_0', 'high_blood_pressure_1', 'sex_0', 'sex_1',
	   'smoking_0', 'smoking_1']

@app.route("/predict", methods=["POST"])
def predict():
	global data_pred
	data = {"success": False}
	dt = strftime("[%Y-%b-%d %H:%M:%S]")
	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		data_pred=dict()
		data_pred.fromkeys(features)
		request_json = flask.request.get_json()
		for feature in features:
			data_pred[feature] = request_json[feature]

		raw_data = pd.DataFrame(data_pred, index=[0])
		raw_data['anaemia'] = raw_data['anaemia'].apply(str)
		raw_data['diabetes'] = raw_data['diabetes'].apply(str)
		raw_data['high_blood_pressure'] = raw_data['high_blood_pressure'].apply(str)
		raw_data['smoking'] = raw_data['smoking'].apply(str)
		raw_data['sex'] = raw_data['sex'].apply(str)

		heart_raw = pd.read_csv('.././data/heart_failure_clinical_records_dataset.csv')

		heart_raw['anaemia'] = heart_raw['anaemia'].apply(str)
		heart_raw['diabetes'] = heart_raw['diabetes'].apply(str)
		heart_raw['high_blood_pressure'] = heart_raw['high_blood_pressure'].apply(str)
		heart_raw['smoking'] = heart_raw['smoking'].apply(str)
		heart_raw['sex'] = heart_raw['sex'].apply(str)

		heart = heart_raw.drop(columns = ['DEATH_EVENT'], axis=1)
		data_joined = pd.concat([raw_data, heart], axis = 0)

		df = pd.get_dummies(data_joined, columns=['anaemia', 'diabetes', 'high_blood_pressure', 'smoking', 'sex'])
		
		col_trans = ColumnTransformer(remainder='passthrough',
							  transformers = [('scaler', StandardScaler(with_mean=True, with_std=True),
							  [0,1,2,3,4,5,6])])
		trans = col_trans.fit_transform(df)

		transformed = pd.DataFrame(trans, columns = col)

		df_ = transformed[:len(raw_data)]

		try:
	#		csv=pd.read_csv('./Model/heart_failure_clinical_records_dataset.csv')
	#		csv.drop('DEATH_EVENT',inplace=True,axis=1)
			preds = model.predict_proba(df_)
	#		preds = model.predict_proba(csv)


		except AttributeError as e:
			logger.warning(f'{dt} Exception: {str(e)}')
			data['predictions'] = str(e)
			data['success'] = False
			return flask.jsonify(data)

		data["predictions"] = preds[:, 1][0]
		# indicate that the request was a success
		data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# then start the server
if __name__ == "__main__":
	print(("* Loading the model and Flask starting server..."
		"please wait until server has fully started"))
	port = int(os.environ.get('PORT', 5000))
	app.run(host='localhost', debug=True, port=port)