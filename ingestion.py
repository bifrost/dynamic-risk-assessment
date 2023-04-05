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

    stats = []

    for each_filename in filenames:
        path = os.path.join(input_path, each_filename)
        df = pd.read_csv(path)
        stats.append(len(df.index))
        result = result.append(df)

    return result, stats

def write_records(fp, sourcelocation, filenames, stats):
    dateTimeObj = datetime.now()
    now = str(dateTimeObj.year)+ '/'+str(dateTimeObj.month)+ '/'+str(dateTimeObj.day)

    for i, _ in enumerate(filenames):
        record = [sourcelocation, filenames[i], stats[i], now]
        record = " ".join([str(o) for o in record])
        fp.write(record+'\n')

#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    output_path = os.path.join(output_folder_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_path = os.path.join(input_folder_path)
    filenames = [f for f in os.listdir(input_path) if f.endswith('.csv')]

    result, stats = read_data(input_path, filenames)
    result = result.drop_duplicates()

    result.to_csv(os.path.join(output_path, 'finaldata.csv'), index=False)

    with open(os.path.join(output_path, 'ingestedfiles.txt'), 'w') as fp:
        write_records(fp, input_path, filenames, stats)

if __name__ == '__main__':
    merge_multiple_dataframe()
