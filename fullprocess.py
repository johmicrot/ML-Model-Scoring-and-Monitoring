import os
import json
import shutil
import sys

import pandas as pd

import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting

################## Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
prod_deployment_path = config['prod_deployment_path']
output_folder_path = config['output_folder_path']
output_model_path = config['output_model_path']
test_data_path = config['test_data_path']


##################Check and read new data
#first, read ingestedfiles.txt

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt

def check_new_data():
    """
    Check for new data in the source data folder that hasn't been ingested yet.
    Returns True if new data is found, False otherwise.
    """
    # Read ingestedfiles.txt from deployment directory
    ingested_files_file = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
    if os.path.exists(ingested_files_file):
        with open(ingested_files_file, 'r') as f:
            ingested_files = f.read().splitlines()
    else:
        ingested_files = []

    # Get list of files in the input data folder
    source_files = os.listdir(input_folder_path)
    source_files = [f for f in source_files if f.endswith('.csv')]

    # Determine if there are new files
    new_files = [f for f in source_files if f not in ingested_files]

    if new_files:
        print("New data found. Proceeding to ingestion.")
        return True
    else:
        print("No new data found. Exiting the process.")
        return False



##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here

new_data = check_new_data()
if not new_data:
    sys.exit()

# Ingest the new data
ingestion.merge_multiple_dataframe()

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data

def check_model_drift():
    """
    Check if the model has drifted by comparing the new model score with the deployed model score.
    Returns True if model drift is detected, False otherwise.
    """
    # Load the deployed model score
    deployed_score_file = os.path.join(prod_deployment_path, 'latestscore.txt')
    if os.path.exists(deployed_score_file):
        with open(deployed_score_file, 'r') as f:
            deployed_score = float(f.read())
    else:
        print("Deployed score not found. Exiting the process.")
        sys.exit()

    # Score the model with the new data
    new_score = scoring.score_model()

    print(f"Deployed model score: {deployed_score}")
    print(f"New model score: {new_score}")

    # Check for model drift
    if new_score < deployed_score:
        print("Model drift detected. Proceeding to retraining and redeployment.")
        return True
    else:
        print("No model drift detected. Exiting the process.")
        return False

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
model_drift = check_model_drift()
if not model_drift:
    sys.exit()


##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
# Retrain the model with the new data
training.train_model()
deployment.store_model_into_pickle()

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
reporting.score_model()

confusion_matrix_path = os.path.join(output_model_path, 'confusionmatrix.png')
confusion_matrix_new_path = os.path.join(output_model_path, 'confusionmatrix2.png')
if os.path.exists(confusion_matrix_path):
    os.rename(confusion_matrix_path, confusion_matrix_new_path)

# Run apicalls.py to call the API endpoints and save the responses
import subprocess
subprocess.run(['python', 'apicalls.py'])

# Rename the apireturns.txt file
apireturns_path = os.path.join(output_model_path, 'apireturns.txt')
apireturns_new_path = os.path.join(output_model_path, 'apireturns2.txt')
if os.path.exists(apireturns_path):
    os.rename(apireturns_path, apireturns_new_path)

print("Process automation completed successfully.")




