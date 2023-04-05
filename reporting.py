import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])

##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)

    df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    x = df.iloc[:,1:-1].values.reshape(-1, 3)
    y = df.iloc[:,-1:].values.reshape(-1, 1).ravel()

    # Note: depricated
    #cm = metrics.plot_confusion_matrix(model, x, y)
    #plt.savefig(os.path.join(model_path, 'confusionmatrix.png'))

    y_pred = model.predict(x)
    cm =  confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig = disp.plot()
    plt.savefig(os.path.join(model_path, 'confusionmatrix.png'), dpi=300)


if __name__ == '__main__':
    score_model()
