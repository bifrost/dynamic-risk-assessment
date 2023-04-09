import pickle
import os

def get_features_target(df):
    # get features and target
    df_y = df['exited']
    df_x = df.drop(columns=['corporation', 'exited'])

    return df_x, df_y

def load_model(model_path):
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb') as file:
        return pickle.load(file)