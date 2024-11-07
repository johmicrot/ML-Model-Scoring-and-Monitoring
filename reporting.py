import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import confusion_matrix
from diagnostics import model_predictions
############### Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


############## Function for reporting
def score_model(test_data, model_path):
    """
    This function loads the deployed model and test data,
    calculates a confusion matrix, and saves the confusion matrix
    plot to the workspace.
    """

    # Make predictions
    y_true, y_pred = model_predictions(model_path, test_data)
    # Calculate the confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # Ensure the output directory exists
    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)

    # Save the confusion matrix plot
    plt.savefig(os.path.join(output_model_path, 'confusionmatrix.png'))
    plt.close()

if __name__ == '__main__':
    # Load the test data
    test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    # Load the deployed model
    model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    score_model(test_data, model_path)
