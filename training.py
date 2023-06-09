from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil
from data_util import get_features_target

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

output_folder_path = config['output_folder_path']
output_model_path = config['output_model_path']

#################Function for training the model
def train_model(model_path, data_location):

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    df = pd.read_csv(os.path.join(data_location, 'finaldata.csv'))
    train, test = train_test_split(df, test_size=0.2, random_state=0)

    df_x, df_y = get_features_target(train)

    #use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)

    #fit the logistic regression to your data
    model.fit(df_x, df_y)

    #write the trained model to your workspace in a file called trainedmodel.pkl
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'wb') as file:
        pickle.dump(model, file)

    # save train and test data in the model path
    train.to_csv(os.path.join(model_path, 'train.csv'), index=False)
    test.to_csv(os.path.join(model_path, 'test.csv'), index=False)

    # looks like we need it for the submission
    shutil.copy(os.path.join(data_location, 'finaldata.csv'), os.path.join(model_path, 'finaldata.csv'))
    shutil.copy(os.path.join(data_location, 'ingestedfiles.txt'), os.path.join(model_path, 'ingestedfiles.txt'))

if __name__ == '__main__':
    train_model(output_model_path, output_folder_path)