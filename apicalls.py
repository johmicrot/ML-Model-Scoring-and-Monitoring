import requests
import json
import os

# Load configuration to get the output_model_path
with open('config.json', 'r') as f:
    config = json.load(f)

output_model_path = config['output_model_path']

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# Define the endpoints
prediction_url = f"{URL}/prediction"
scoring_url = f"{URL}/scoring"
stats_url = f"{URL}/summarystats"
diagnostics_url = f"{URL}/diagnostics"

# Call each API endpoint and store the responses

# For the prediction endpoint, we need to provide the dataset path
# Using 'testdata/testdata.csv' as per the instruction
response1 = requests.post(
    prediction_url,
    json={'dataset_path': 'testdata/testdata.csv'}
).json()

response2 = requests.get(scoring_url).json()

response3 = requests.get(stats_url).json()

response4 = requests.get(diagnostics_url).json()

# Combine all API responses
responses = {
    'prediction': response1,
    'scoring': response2,
    'summarystats': response3,
    'diagnostics': response4
}

# Write the responses to your workspace
# Save to apireturns.txt in the output_model_path directory
output_file_path = os.path.join(output_model_path, 'apireturns.txt')

with open(output_file_path, 'w') as f:
    f.write(json.dumps(responses, indent=4))
