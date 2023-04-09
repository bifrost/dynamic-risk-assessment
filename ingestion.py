import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import glob

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

def read_data(input_path):

    filenames = glob.glob(f'{input_path}/*.csv')
    datasets = list(map(pd.read_csv, filenames))
    stats = [len(ds) for ds in datasets]
    result = pd.concat(datasets)

    return result, filenames, stats

def write_records(fp, sourcelocation, filenames, stats):
    dateTimeObj = datetime.now()
    now = str(dateTimeObj.year)+ '/'+str(dateTimeObj.month)+ '/'+str(dateTimeObj.day)

    for i, _ in enumerate(filenames):
        record = [sourcelocation, filenames[i], stats[i], now]
        record = " ".join([str(o) for o in record])
        fp.write(record+'\n')

#############Function for data ingestion
def merge_multiple_dataframe(input_folder_path, output_folder_path):
    #check for datasets, compile them together, and write to an output file
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_file_path = os.path.join(output_folder_path, 'finaldata.csv')
    ingested_file_path = os.path.join(output_folder_path, 'ingestedfiles.txt')

    #['corporation', 'lastmonth_activity', 'lastyear_activity',  'number_of_employees', 'exited']

    result, filenames, stats = read_data(input_folder_path)

    result = result.drop_duplicates()

    result.to_csv(output_file_path, index=False)

    with open(ingested_file_path, 'w') as fp:
        write_records(fp, input_folder_path, filenames, stats)

if __name__ == '__main__':
    merge_multiple_dataframe(input_folder_path, output_folder_path)
