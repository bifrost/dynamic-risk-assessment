import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

def read_data(input_path, filenames):

    result = pd.DataFrame(columns=[
        'corporation',
        'lastmonth_activity',
        'lastyear_activity',
        'number_of_employees',
        'exited'
        ])

    for each_filename in filenames:
        path = f'{input_path}/{each_filename}'
        df = pd.read_csv(path)
        result = result.append(df)

    return result

def clean_data(df):
    return df.drop_duplicates()


#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    output_path = f'{os.getcwd()}/{output_folder_path}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_path = f'{os.getcwd()}/{input_folder_path}'
    filenames = [f for f in os.listdir(input_path) if f.endswith('.csv')]

    result = read_data(input_path, filenames)
    result = clean_data(result)

    result.to_csv(f'{output_path}/finaldata.csv', index=False)

    with open(f'{output_path}/ingestedfiles.txt', 'w') as fp:
        fp.write("\n".join(str(item) for item in filenames))

if __name__ == '__main__':
    merge_multiple_dataframe()
