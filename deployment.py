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


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f)

output_folder_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_model_path = os.path.join(config['output_model_path'])

if not os.path.exists(prod_deployment_path):
    os.makedirs(prod_deployment_path)

####################function for deployment
def deploy_model(model_path, ingested_path, prod_deployment_path):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    shutil.copy(os.path.join(model_path, 'trainedmodel.pkl'), os.path.join(prod_deployment_path, 'trainedmodel.pkl'))
    shutil.copy(os.path.join(model_path, 'latestscore.txt'), os.path.join(prod_deployment_path, 'latestscore.txt'))
    shutil.copy(os.path.join(ingested_path, 'ingestedfiles.txt'), os.path.join(prod_deployment_path, 'ingestedfiles.txt'))

if __name__ == '__main__':
    deploy_model(output_model_path, output_folder_path, prod_deployment_path)


