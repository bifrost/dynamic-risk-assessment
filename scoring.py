from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

from data_util import get_features_target, load_model

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])

#################Function for model scoring
def score_model(model_path, data_location):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    df = pd.read_csv(data_location)
    df_x, df_y = get_features_target(df)

    model = load_model(model_path)

    y_pred = model.predict(df_x)

    score = metrics.f1_score(df_y, y_pred)

    with open(os.path.join(model_path, 'latestscore.txt'), 'w') as fp:
        fp.write(str(score)+'\n')

    return score

if __name__ == '__main__':
    data_location = os.path.join(test_data_path, 'testdata.csv')
    score_model(model_path, data_location)