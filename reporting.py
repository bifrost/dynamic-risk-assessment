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
from data_util import get_features_target, load_model

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

test_data_path = config['test_data_path']
output_model_path = config['output_model_path']

##############Function for reporting
def generate_report(model_path, data_location):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    model = load_model(model_path)

    df = pd.read_csv(data_location)
    df_x, df_y = get_features_target(df)

    y_pred = model.predict(df_x)
    cm =  confusion_matrix(df_y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig = disp.plot()
    plt.savefig(os.path.join(model_path, 'confusionmatrix.png'), dpi=300)


if __name__ == '__main__':
    data_location = os.path.join(test_data_path, 'testdata.csv')
    generate_report(output_model_path, data_location)
