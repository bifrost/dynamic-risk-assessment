import subprocess
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

with open('config.json','r') as f:
    config = json.load(f)

output_model_path = os.path.join(config['output_model_path'])

def run_diagnostics(model_path):

    #Call each API endpoint and store the responses
    curl_command = f'curl -X POST {URL}prediction?file_location=/testdata/testdata.csv'
    prediction = subprocess.run(curl_command.split(), capture_output=True).stdout

    curl_command = f'curl {URL}scoring'
    scoring = subprocess.run(curl_command.split(), capture_output=True).stdout

    curl_command = f'curl {URL}summarystats'
    summarystats = subprocess.run(curl_command.split(), capture_output=True).stdout

    curl_command = f'curl {URL}diagnostics'
    diagnostics = subprocess.run(curl_command.split(), capture_output=True).stdout

    #combine all API responses
    responses = {
        'prediction': json.loads(prediction),
        'scoring': json.loads(scoring),
        'summarystats': json.loads(summarystats),
        'diagnostics': json.loads(diagnostics)
    } #combine reponses here

    #write the responses to your workspace
    with open(os.path.join(model_path, 'apireturns.txt'), 'w') as fp:
        json.dump(responses, fp, indent = 6)

if __name__ == '__main__':
    run_diagnostics(output_model_path)

