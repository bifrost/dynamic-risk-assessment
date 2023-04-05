
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
from io import StringIO

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(data_location = os.path.join(test_data_path, 'testdata.csv')):
    #read the deployed model and a test dataset, calculate predictions
    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)

    df = pd.read_csv(data_location)

    x = df.iloc[:,1:-1].values.reshape(-1, 3)

    y_pred = model.predict(x)

    return y_pred #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    x = df.iloc[:,1:-1]

    result = np.array([x.mean(), x.median(), x.std()]).flatten().tolist()
    return result #return value should be a list containing all summary statistics

def missing_data():
    #calculate what percent of each column consists of NA values
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    x = df.iloc[:,1:-1]

    nas = list(x.isna().sum())
    result = [nas[i]/len(x.index) for i in range(len(nas))]
    return result

##################Function to get timings
def ingestion_timing():
    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    timing = timeit.default_timer() - starttime
    return timing

def training_timing():
    starttime = timeit.default_timer()
    os.system('python3 training.py')
    timing = timeit.default_timer() - starttime
    return timing

def execution_time():
    #calculate timing of training.py and ingestion.py

    result = [ingestion_timing(), training_timing()]
    return result #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of

    outdated = subprocess.check_output(['pip', 'list','--outdated']).decode("utf-8")
    df = pd.read_csv(StringIO(outdated), skiprows=1, sep=' +', engine='python')
    result = df.iloc[:,:-1].values

    return result.tolist()


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()






