
import json
import os
import sys
import logging
import time
from ingestion import merge_multiple_dataframe
from training import train_model
from scoring import score_model
from deployment import deploy_model
from reporting import generate_report
from apicalls import run_diagnostics

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [logging.StreamHandler()]
)

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

prod_deployment_path = config['prod_deployment_path']
input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
output_model_path = config['output_model_path']
test_data_path = config['test_data_path']

ingested_file_path = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
latest_score_path = os.path.join(prod_deployment_path, 'latestscore.txt')
final_data_path = os.path.join(output_folder_path, 'finaldata.csv')
test_data_file = os.path.join(test_data_path, 'testdata.csv')

def process_automation():
# Precess automation

    ##################Check and read new data
    #first, read ingestedfiles.txt
    logging.info('Check and read new data')

    try:
        with open(ingested_file_path, 'r') as f:
            ingestedfiles = f.read()
    except:
         # ingestedfiles.txt does not exists
        ingestedfiles = ''

    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    new_filenames = [f for f in os.listdir(input_folder_path) if f.endswith('.csv') and ingestedfiles.find(f) == -1]

    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here
    if len(new_filenames) == 0:
        logging.info('No new data to ingest, stop execution')
        sys.exit(0)

    logging.info('Files to ingest %s' % new_filenames)

    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    logging.info('Checking for model drift')

    logging.info('Data Ingestion')
    merge_multiple_dataframe(input_folder_path, output_folder_path, False)

    logging.info('Read old model score')
    try:
        with open(latest_score_path, 'r') as fp:
            latest_score = float(fp.read())
    except:
        # latestscore.txt does not exists
        latest_score = 1.0
    logging.info('Old model score %f' % latest_score)

    try:
        new_score = score_model(prod_deployment_path, final_data_path)
    except:
        # we have no model
        new_score = 0.0
    logging.info('New model score %f' % new_score)


    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here
    if new_score >= latest_score:
        logging.info('No model drift found')
        sys.exit(0)

    logging.info('Model drift found')

    ##################Re-training
    logging.info('Re-train model')
    train_model(output_model_path, output_folder_path)

    logging.info('Score model')
    # we need a new score - but what test data should be used?
    # testdata.csv is old data and finaldata.csv has been used for training!!!
    score_model(output_model_path, final_data_path)

    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
    logging.info('Re-deploy model')
    deploy_model(output_model_path, output_folder_path, prod_deployment_path)

    # wait for restart of deployed model
    time.sleep(1)

    ##################Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model
    logging.info('Run diagnostics')
    run_diagnostics(output_model_path)

    logging.info('Generate report')
    # What data should be used?
    # testdata.csv is old data and finaldata.csv has been used for training!!!
    generate_report(output_model_path, test_data_file)


if __name__ == '__main__':
    process_automation()