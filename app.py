from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
# import create_prediction_model
# import diagnosis
# import predict_exited_from_saved_model
import json
import os

import diagnostics  # Import diagnostics.py
import scoring      # Import scoring.py
from diagnostics import model_predictions, dataframe_summary, execution_time, missing_data, outdated_packages_list
from scoring import score_model
######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = os.path.join(config['prod_deployment_path'])
# prediction_model = None


#######################Prediction Endpoint

@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    """
    This endpoint takes a dataset's file location as input and returns
    the model predictions.
    """
    # Get the dataset file location from the POST request
    data = request.get_json()
    if data is None or 'dataset_path' not in data:
        return jsonify({'message': 'No dataset_path provided'}), 400


    dataset_path = data['dataset_path']
    # Read the dataset
    try:
        dataset = pd.read_csv(dataset_path)
    except Exception as e:
        return jsonify({'message': f'Error reading dataset: {str(e)}'}), 400

    model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    y_true, y_pred = model_predictions(model_path, dataset)
    # Load the deployed model
    model = pickle.load(open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb'))
    #
    #
    #
    # # Prepare features
    # try:
    #     X = dataset[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    # except KeyError as e:
    #     return jsonify({'message': f'Missing columns in dataset: {str(e)}'}), 400
    #
    # # Make predictions
    # predictions = model.predict(X)

    # Return predictions
    return jsonify({'predictions': y_pred}), 200

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    """
    This endpoint returns the F1 score of the deployed model.
    """
    # Run the scoring function and get the F1 score
    f1_score = score_model()

    # Return the F1 score
    return jsonify({'f1_score': f1_score}), 200

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summarystats():
    """
    This endpoint returns summary statistics of the ingested data.
    """
    # Run the summary statistics function
    stats = dataframe_summary()

    # Return the summary statistics
    return jsonify({'summary_statistics': stats}), 200

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics_endpoint():
    """
    This endpoint returns diagnostics information such as execution times,
    missing data percentages, and outdated packages.
    """
    # Execution times
    timing = execution_time()

    # Missing data
    missing = missing_data()

    # Outdated packages
    outdated = outdated_packages_list()

    # Return the diagnostics
    return jsonify({
        'execution_time': timing,
        'missing_data': missing,
        'outdated_packages': outdated
    }), 200


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
