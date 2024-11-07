import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




# Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



# Function for data ingestion
def merge_multiple_dataframe():
    # Ensure the output directory exists
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Get all CSV files in the input directory
    filenames = [ f for f in os.listdir(input_folder_path) if f.endswith('.csv') ]

    # Initialize a list to hold DataFrames
    df_list = []

    # Initialize a list to record ingested files
    ingested_files = []

    # Read and append each CSV file to the list
    for filename in filenames:
        file_path = os.path.join(input_folder_path, filename)
        df = pd.read_csv(file_path)
        df_list.append(df)
        ingested_files.append(filename)

    # Concatenate all DataFrames
    combined_df = pd.concat(df_list, ignore_index=True)

    # Remove duplicate rows
    combined_df.drop_duplicates(inplace=True)

    # Save the combined DataFrame to finaldata.csv
    output_file_path = os.path.join(output_folder_path, 'finaldata.csv')
    combined_df.to_csv(output_file_path, index=False)

    # Save the list of ingested files to ingestedfiles.txt
    ingested_files_path = os.path.join(output_folder_path, 'ingestedfiles.txt')
    with open(ingested_files_path, 'w') as f:
        for filename in ingested_files:
            f.write(f"{filename}\n")

    print(f"Ingested data saved to {output_file_path}")
    print(f"Ingested files recorded in {ingested_files_path}")


if __name__ == '__main__':
    merge_multiple_dataframe()