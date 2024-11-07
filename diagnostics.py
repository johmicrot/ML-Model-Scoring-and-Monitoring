import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import sys
import re

################## Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

################## Function to get model predictions
def model_predictions(model_path, test_data):

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    X = test_data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y_true = test_data['exited']
    # Make predictions
    predictions = model.predict(X)

    return y_true, predictions.tolist()  # Return predictions as a list

################## Function to get summary statistics
def dataframe_summary():
    # Select numeric columns
    # Read the ingested dataset
    data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    # Calculate summary statistics
    summary_statistics = []
    for col in numeric_cols:
        mean = data[col].mean()
        median = data[col].median()
        std = data[col].std()
        summary_statistics.append({'column': col, 'mean': mean, 'median': median, 'std': std})

    return summary_statistics

################## Function to check for missing data
def missing_data():
    data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    # Calculate the percentage of missing data in each column
    total_rows = data.shape[0]
    percent_missing = data.isnull().sum() / total_rows * 100

    missing_data_info = percent_missing.to_dict()  # Convert to dictionary for clarity

    return missing_data_info

################## Function to get timings
def execution_time():
    import ingestion
    import training

    # Time ingestion.py
    start_time = timeit.default_timer()
    ingestion.merge_multiple_dataframe()
    ingestion_time = timeit.default_timer() - start_time

    # Time training.py
    start_time = timeit.default_timer()
    training.train_model()
    training_time = timeit.default_timer() - start_time

    return {'ingestion_time': ingestion_time, 'training_time': training_time}

################## Function to check dependencies
def outdated_packages_list():
    # Get the list of outdated packages in JSON format
    outdated = subprocess.check_output([sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json']).decode('utf-8')
    outdated_packages = json.loads(outdated)

    # Prepare the list to return
    package_list = []
    for package in outdated_packages:
        package_info = {
            'name': package['name'],
            'current_version': package['version'],
            'latest_version': package['latest_version']
        }
        package_list.append(package_info)

    return package_list

if __name__ == '__main__':
    model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    _, predictions = model_predictions(model_path, test_data)
    print("Model Predictions:")
    print(predictions)
    print()


    stats = dataframe_summary()
    print("Summary Statistics:")
    for stat in stats:
        print(stat)
    print()

    missing = missing_data()
    print("Missing Data Percentage:")
    for col, percent in missing.items():
        print(f"{col}: {percent}%")
    print()

    # Get execution times
    times = execution_time()
    print("Execution Times:")
    print(times)
    print()

    # Get list of outdated packages
    outdated = outdated_packages_list()
    print("Outdated Packages:")
    for package in outdated:
        print(f"{package['name']}: Current version {package['current_version']}, Latest version {package['latest_version']}")
