from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from diagnostics import model_predictions, dataframe_summary, missing_data, execution_time, outdated_packages_list
from scoring import score_model
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f)

output_folder_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    #call the prediction function you created in Step 3
    data_location = os.getcwd()+request.args.get('file_location')
    y_pred = model_predictions(prod_deployment_path, data_location)
    return json.dumps(y_pred.tolist()) #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():
    #check the score of the deployed model
    data_location = os.path.join(test_data_path, 'testdata.csv')
    score = score_model(prod_deployment_path, data_location)
    return json.dumps(score) #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():
    #check means, medians, and modes for each column
    summary = dataframe_summary(output_folder_path)
    return json.dumps(summary) #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    #check timing and percent NA values
    result = {
        'execution_time': execution_time(),
        'missing_data': missing_data(output_folder_path),
        'outdated_packages_list': outdated_packages_list()
    }

    return json.dumps(result) #add return value for all diagnostics

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
