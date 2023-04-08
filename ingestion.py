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

def read_data(input_path, filenames, result):

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
def merge_multiple_dataframe(input_folder_path, output_folder_path, append_data):
    #check for datasets, compile them together, and write to an output file
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_file_path = os.path.join(output_folder_path, 'finaldata.csv')
    filenames = [f for f in os.listdir(input_folder_path) if f.endswith('.csv')]
    ingested_file_path = os.path.join(output_folder_path, 'ingestedfiles.txt')

    if append_data == True:
        mode = 'a'
        df = pd.read_csv(output_file_path)
    else:
        mode = 'w'
        df = pd.DataFrame(columns=['corporation', 'lastmonth_activity', 'lastyear_activity',  'number_of_employees', 'exited'])

    result, stats = read_data(input_folder_path, filenames, df)
    result = result.drop_duplicates()

    result.to_csv(output_file_path, index=False)

    with open(ingested_file_path, mode) as fp:
        write_records(fp, input_folder_path, filenames, stats)

if __name__ == '__main__':
    merge_multiple_dataframe(input_folder_path, output_folder_path, False)
