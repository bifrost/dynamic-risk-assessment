import subprocess
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

with open('config.json','r') as f:
    config = json.load(f)

model_path = os.path.join(config['output_model_path'])

#Call each API endpoint and store the responses
curl_command = f'curl -X POST {URL}prediction?file_location=/testdata/testdata.csv'
response1 = subprocess.run(curl_command.split(), capture_output=True).stdout

curl_command = f'curl {URL}scoring'
response2 = subprocess.run(curl_command.split(), capture_output=True).stdout

curl_command = f'curl {URL}summarystats'
response3 = subprocess.run(curl_command.split(), capture_output=True).stdout

curl_command = f'curl {URL}diagnostics'
response4 = subprocess.run(curl_command.split(), capture_output=True).stdout

#combine all API responses
responses = {
    'response1': json.loads(response1),
    'response2': json.loads(response2),
    'response3': json.loads(response3),
    'response4': json.loads(response4)
} #combine reponses here

#write the responses to your workspace
with open(os.path.join(model_path, 'apireturns.txt'), 'w') as fp:
    json.dump(responses, fp, indent = 6)




