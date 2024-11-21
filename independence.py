import sys
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json

samples = 1000
rare = 0.2
reps = [0, 1, 5, 10, 50]


def calculate_precision(df_repped_in, df_sampled_in, target_col):
    # Make copies of the input DataFrames
    df_repped = df_repped_in.copy()
    df_sampled = df_sampled_in.copy()

    # Identify categorical columns
    categorical_cols = df_repped.select_dtypes(include=['object']).columns.tolist()

    # Convert all columns to strings to ensure uniformity
    df_repped[categorical_cols] = df_repped[categorical_cols].astype(str)
    df_sampled[categorical_cols] = df_sampled[categorical_cols].astype(str)

    # Encode the target column if it is categorical
    le = LabelEncoder()
    df_repped[target_col] = le.fit_transform(df_repped[target_col])
    df_sampled[target_col] = le.transform(df_sampled[target_col])

    # Split the data into features and target
    X_repped = df_repped.drop(columns=[target_col])
    y_repped = df_repped[target_col]
    X_sampled = df_sampled.drop(columns=[target_col])
    y_sampled = df_sampled[target_col]

    # Update the list of categorical columns after dropping the target column
    categorical_cols = [col for col in categorical_cols if col != target_col]

    # Apply OneHotEncoder to categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    # Transform the features
    X_repped = preprocessor.fit_transform(X_repped)
    X_sampled = preprocessor.transform(X_sampled)

    # Train a RandomForestClassifier on the repped data
    #model = RandomForestClassifier(random_state=42)
    model = RandomForestClassifier(max_features='sqrt', min_samples_split=10, min_samples_leaf=4, random_state=42)
    model.fit(X_repped, y_repped)

    # Predict the target column for the sampled data
    y_pred = model.predict(X_sampled)

    # Calculate the precision
    accuracy = accuracy_score(y_sampled, y_pred)

    return accuracy

def sample_and_exclude(df, df_out, num_rows=samples):
    # Select num_rows random rows from df_out
    if len(df_out) >= num_rows:
        df_sampled = df_out.sample(n=num_rows, random_state=42)
    else:
        df_sampled = df_out  # If there are fewer than num_rows rows, return all rows in df_out

    # Get the indices of the sampled rows
    sampled_indices = df_sampled.index

    # Exclude the sampled rows from df to get df_remain
    df_remain = df.drop(sampled_indices)

    return df_remain, df_sampled

def filter_rare_values(df, col):
    # Calculate the frequency of each value in the specified column
    value_counts = df[col].value_counts(normalize=True)
    
    # Identify values that occur in fewer than 20% of the rows
    rare_values = value_counts[value_counts < rare].index
    
    # Filter the DataFrame to include only rows with rare values in the specified column
    df_out = df[df[col].isin(rare_values)]
    
    return df_out

def get_categorical_columns(df):
    categorical_columns = []
    
    for column in df.columns:
        if df[column].dtype == 'object':
            # Column is of type string
            categorical_columns.append(column)
        elif pd.api.types.is_numeric_dtype(df[column]):
            # Column is numeric, check the number of distinct values
            if df[column].nunique() <= 20:
                categorical_columns.append(column)
    
    return categorical_columns

def convert_datetime_to_timestamp(df):
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            df[col] = df[col].astype(int) / 10**9
    return df

def read_parquet_files(directory):
    # List to store the dictionaries
    data_list = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".parquet"):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            
            # Read the Parquet file into a DataFrame
            df = pd.read_parquet(file_path)
            df = convert_datetime_to_timestamp(df)
            cats = get_categorical_columns(df)
            rare_cats = []
            for col in cats:
                df_rare = filter_rare_values(df, col)
                if len(df_rare) >= samples:
                    rare_cats.append(col)
            
            # Create a dictionary with the file name and DataFrame
            data_dict = {
                'file_name': filename,
                'cats': cats,
                'rare_cats': rare_cats,
                'dataframe': df
            }
            
            # Append the dictionary to the list
            data_list.append(data_dict)
    
    return data_list

def select_random_cols(x, num_values=20):
    random.seed(42)
    if len(x) >= num_values:
        return random.sample(x, num_values)
    else:
        return x

def append_replicates(df_remain, df_sampled, rep):
    # Create a list to store the replicated rows
    replicated_rows = []

    # Iterate over each row in df_sampled
    for _, row in df_sampled.iterrows():
        # Append rep copies of the row to the list
        replicated_rows.extend([row] * rep)

    # Convert the list of replicated rows to a DataFrame
    df_replicated = pd.DataFrame(replicated_rows)

    # Append the replicated rows to df_remain
    df_repped = pd.concat([df_remain, df_replicated], ignore_index=True)

    return df_repped

def job_exists(results, file_name, column, rep):
    for result in results:
        if result['file_name'] == file_name and result['column'] == column and result['replicates'] == rep:
            return True
    return False

def do_work(job_num):
    print(f"Running do_work with job_num {job_num}")
    data_list = read_parquet_files('original_data_parquet')
    os.makedirs('independence_results', exist_ok=True)
    res_path = os.path.join('independence_results', f'results.{job_num}.json')
    if os.path.exists(res_path):
        print(f"Results file {res_path} already exists")
        quit()
    # Print the file names and the first few rows of each DataFrame
    this_job = 0
    for dataset in data_list:
        df = dataset['dataframe']
        job_cols = select_random_cols(dataset['rare_cats'])
        for col in job_cols:
            for rep in reps:
                if this_job < int(job_num):
                    this_job += 1
                    continue
                df_out = filter_rare_values(df, col)
                df_remain, df_sampled = sample_and_exclude(df, df_out)
                if len(df) != len(df_remain) + len(df_sampled):
                    print(f"Error: Length mismatch for {dataset['file_name']} and column {col}")
                    quit()
                print(f"File: {dataset['file_name']}")
                print(f"Do independence test for column {col}")
                print(f"Do {rep} replicates")
                matches = 0
                num_predictions = 0
                results = []
                for _, row in df_sampled.iterrows():
                    num_predictions += 1
                    df_this_sample = pd.DataFrame([row])
                    target_val = df_this_sample[col].values[0]
                    df_repped = append_replicates(df_remain, df_this_sample, rep)
                    if len(df_repped) != len(df_remain) + (len(df_this_sample) * rep):
                        print(f"Error: Length mismatch for {dataset['file_name']} and column {col} with {rep} replicates")
                        quit()
                    prec = calculate_precision(df_repped, df_this_sample, col)
                    matches += prec
                    results.append({'target_val': str(target_val), 'precision': prec})
                    finals = {'prec': matches / num_predictions,
                            'dataset': dataset['file_name'],
                            'column': col,
                            'replicates': rep,
                            'results': results}
                    with open(res_path, 'w') as json_file:
                        json.dump(finals, json_file, indent=4)
                quit()
    print(f"Job {job_num} not found (this_job={this_job})")

def main():
    if len(sys.argv) > 1:
        do_work(sys.argv[1])
    else:
        print("No command line parameters were provided.")

if __name__ == "__main__":
    main()