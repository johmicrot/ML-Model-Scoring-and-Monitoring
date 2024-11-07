from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


output_model_path = os.path.join(config['output_model_path'])
output_folder_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


#################### Function for deployment
def store_model_into_pickle():
    # Ensure the deployment directory exists
    if not os.path.exists(prod_deployment_path):
        os.makedirs(prod_deployment_path)

    # Paths of the files to copy
    trained_model_src = os.path.join(output_model_path, 'trainedmodel.pkl')
    latest_score_src = os.path.join(output_model_path, 'latestscore.txt')
    ingested_files_src = os.path.join(output_folder_path, 'ingestedfiles.txt')

    # Copy the files to the deployment directory
    shutil.copy(trained_model_src, prod_deployment_path)
    shutil.copy(latest_score_src, prod_deployment_path)
    shutil.copy(ingested_files_src, prod_deployment_path)

    print(f"Files copied to {prod_deployment_path}")

if __name__ == '__main__':
    store_model_into_pickle()
        

