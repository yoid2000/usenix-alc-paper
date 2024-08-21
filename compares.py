import anonymeter_mods
import argparse
import os
import pandas as pd
import json
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from alscore import ALScore
#import sdv
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import CopulaGANSynthesizer
import pprint

pp = pprint.PrettyPrinter(indent=4)

if 'ALC_TEST_DIR' in os.environ:
    base_path = os.getenv('ALC_TEST_DIR')
else:
    base_path = os.getcwd()
if 'ALC_TEST_CODE' in os.environ:
    code_path = os.getenv('ALC_TEST_CODE')
else:
    code_path = None
orig_path = os.path.join(base_path, 'original_data_parquet')
syn_path = os.path.join(base_path, 'synDatasets')
os.makedirs(syn_path, exist_ok=True)

plots_path = os.path.join(base_path, 'plots')
os.makedirs(plots_path, exist_ok=True)
# This is the total number of attacks that will be run
num_attacks = 500000
# This is the number of attacks per slurm job, and determines how many slurm jobs are created
num_attacks_per_job = 100
max_subsets = 200
debug = False

def make_one_syn(job_num):
    files = os.listdir(orig_path)
    file_num = int(job_num / 2)
    if file_num >= len(files):
        print(f"file_num {file_num} from {job_num} is out of range")
        sys.exit(1)
    file_name = files[file_num]
    file_base = file_name[:-8]
    data_path = os.path.join(syn_path, file_base)
    if job_num % 2 == 0:
        # Do the full dataset
        print(f"Do full: job_num {job_num}, file_num {file_num}, file_base {file_base}")
        parquet_path = os.path.join(orig_path, file_name)
        this_syn_path = os.path.join(data_path, 'full_syn')
    else:
        # Do the part dataset
        print(f"Do part: job_num {job_num}, file_num {file_num}, file_base {file_base}")
        parquet_path = os.path.join(data_path, 'part_raw')
        this_syn_path = os.path.join(data_path, 'part_syn')
    if os.path.exists(os.path.join(this_syn_path, f'{file_base}.csv')):
        print(f"File already exists at {this_syn_path}. Quitting...")
        return
    print(f"Reading file at {parquet_path}")
    print(f"Will write synthetic data to {this_syn_path}")
    df = pd.read_parquet(parquet_path)

    # Synthesize the full dataset
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    synthesizer = CopulaGANSynthesizer(metadata)
    synthesizer.fit(df)
    df_syn = synthesizer.sample(num_rows=len(df))
    df_syn.to_csv(os.path.join(this_syn_path, f'{file_base}.csv'), index=False)
    df_syn.to_parquet(os.path.join(this_syn_path, f'{file_base}.parquet'), index=False)

def make_syn():
    files = os.listdir(orig_path)
    for file_name in files:
        # strip '.parquet'
        file_base = file_name[:-8]
        parquet_path = os.path.join(orig_path, file_name)
        df = pd.read_parquet(parquet_path)
        data_path = os.path.join(syn_path, file_base)
        os.makedirs(data_path, exist_ok=True)
        # The synthetic data from all the data
        full_syn_path = os.path.join(data_path, 'full_syn')
        os.makedirs(full_syn_path, exist_ok=True)
        # The synthetic data from part of the data
        part_syn_path = os.path.join(data_path, 'part_syn')
        os.makedirs(part_syn_path, exist_ok=True)
        # The metadata
        meta_path = os.path.join(data_path, 'meta')
        os.makedirs(meta_path, exist_ok=True)
        # The rows to attack
        test_raw_path = os.path.join(data_path, 'test')
        os.makedirs(test_raw_path, exist_ok=True)
        # The data to use for the ALC baseline
        part_raw_path = os.path.join(data_path, 'part_raw')
        os.makedirs(part_raw_path, exist_ok=True)

        df_test_raw = df.sample(n=1000, random_state=42)
        df_test_raw.to_parquet(os.path.join(test_raw_path, f'{file_base}.parquet'), index=False)
        df_part_raw = df.drop(df_test_raw.index)
        df_part_raw.to_parquet(os.path.join(part_raw_path, f'{file_base}.parquet'), index=False)

        # Make and save the metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        sdv_metadata = metadata.to_dict()
        pp.pprint(sdv_metadata)
        metadata.validate_data(data=df)
        meta_path = os.path.join(meta_path, f'{file_base}.json')
        with open(meta_path, 'w') as f:
            json.dump(sdv_metadata, f, indent=4)

    exe_path = os.path.join(code_path, 'compares.py')
    venv_path = os.path.join(base_path, '.venv', 'bin', 'activate')
    slurm_dir = os.path.join(base_path, 'slurm_syn_out')
    os.makedirs(slurm_dir, exist_ok=True)
    slurm_out = os.path.join(slurm_dir, 'out.%a.out')
    num_jobs = (len(files) * 2) - 1
    # Define the slurm template
    slurm_template = f'''#!/bin/bash
#SBATCH --job-name=make_syn
#SBATCH --output={slurm_out}
#SBATCH --time=7-0
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-{num_jobs}
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
source {venv_path}
python {exe_path} make_one_syn $arrayNum
'''
    # write the slurm template to a file attack.slurm
    with open(os.path.join(base_path, 'make_syn.slurm'), 'w', encoding='utf-8') as f:
        f.write(slurm_template)

def init_variants():
    for v_label in variants.keys():
        variants[v_label] = []

def convert_datetime_to_timestamp(df):
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            df[col] = df[col].astype(int) / 10**9
    return df

def fit_encoders(dfs):
    # Get the string columns
    string_columns = dfs[0].select_dtypes(include=['object']).columns

    encoders = {col: LabelEncoder() for col in string_columns}

    for col in string_columns:
        # Concatenate the values from all DataFrames for this column
        values = pd.concat(df[col] for df in dfs).unique()
        # Fit the encoder on the unique values
        encoders[col].fit(values)

    return encoders

def transform_df(df, encoders):
    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])
    return df

def transform_df_with_update(df, encoders):
    for col, encoder in encoders.items():
        if col in df.columns:
            unique_values = pd.Series(df[col].unique())
            unseen_values = unique_values[~unique_values.isin(encoder.classes_)]
            encoder.classes_ = np.concatenate([encoder.classes_, unseen_values])
            df[col] = encoder.transform(df[col])
    return df

def find_most_frequent_value(lst, fraction):
    if len(lst) == 0:
        return None

    # Count the frequency of each value in the list
    counter = Counter(lst)
    
    # Find the most common value and its count
    most_common_value, most_common_count = counter.most_common(1)[0]
    
    # Check if the most common value accounts for at least the given fraction of total entries
    if most_common_count / len(lst) > fraction:
        return most_common_value
    else:
        return None

def build_and_train_model(df, target_col, target_type):
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # If the target is categorical, encode it to integers
    if target_type == 'categorical':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Build and train the model
    if target_type == 'categorical':
        #print(f"building RandomForestClassifier with shape {X.shape}")
        try:
            model = RandomForestClassifier(random_state=42)
            #print("finished building RandomForestClassifier")
        except Exception as e:
            #print(f"A RandomForestClassifier error occurred: {e}")
            sys.exit(1)
    elif target_type == 'continuous':
        #print(f"building RandomForestRegressor with shape {X.shape}")
        try:
            model = RandomForestRegressor(random_state=42)
            #print("finished building RandomForestRegressor")
        except Exception as e:
            #print(f"A RandomForestRegressor error occurred: {e}")
            sys.exit(1)
    else:
        raise ValueError("target_type must be 'categorical' or 'continuous'")

    model.fit(X, y)
    return model

def make_config():
    ''' I want to generate num_attacks attacks. Each attack will be on a given secret
    column in a given table with given known columns.
    '''
    measure_jobs = []
    files = os.listdir(orig_path)
    for file_name in files:
        # strip '.parquet'
        file_base = file_name[:-8]
        data_path = os.path.join(syn_path, file_base)
        meta_path = os.path.join(data_path, 'meta', f'{file_base}.json')
        with open(meta_path, 'r') as f:
            meta_data = json.load(f)
        for secret_column, col_type in meta_data['columns'].items():
            print(f"table {file_base}, secret_column: {secret_column} is {col_type['sdtype']}")
            if col_type['sdtype'] != 'categorical':
                continue
            columns = list(meta_data['columns'].keys())
            aux_cols = [col for col in columns if col not in secret_column]
            measure_jobs.append({
                'dir_name': file_base,
                'secret': secret_column,
                'aux_cols': aux_cols,
            })
    random.shuffle(measure_jobs)
    for index, job in enumerate(measure_jobs):
        job['index'] = index

    # Write attack_jobs into a JSON file
    with open(os.path.join(base_path, 'measure_jobs.json'), 'w') as f:
        json.dump(measure_jobs, f, indent=4)

    exe_path = os.path.join(code_path, 'compares.py')
    venv_path = os.path.join(base_path, 'sdx_venv', 'bin', 'activate')
    slurm_dir = os.path.join(base_path, 'slurm_measure_out')
    os.makedirs(slurm_dir, exist_ok=True)
    slurm_out = os.path.join(slurm_dir, 'out.%a.out')
    num_jobs = len(measure_jobs) - 1
    # Define the slurm template
    slurm_template = f'''#!/bin/bash
#SBATCH --job-name=measures
#SBATCH --output={slurm_out}
#SBATCH --time=7-0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-{num_jobs}
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
source {venv_path}
python {exe_path} measure $arrayNum
'''
    # write the slurm template to a file measure.slurm
    with open(os.path.join(base_path, 'measure.slurm'), 'w', encoding='utf-8') as f:
        f.write(slurm_template)

def get_valid_combs(tm, secret_col, aux_cols):
    # We want the column combinations that containt secret_col and have at least
    # one other column. tm.catalog contains every subset combination that was created.
    if tm.catalog is None:
        tm.build_catalog()
    valid_combs = []
    for catalog_entry in tm.catalog:
        # check to see if every column in catalog_entry is in aux_cols
        if not all(col in aux_cols + [secret_col] for col in catalog_entry['columns']):
            continue
        if secret_col in catalog_entry['columns'] and len(catalog_entry['columns']) > 1:
            valid_combs.append(catalog_entry['columns'])
    return valid_combs

def align_column_types(df, df2, df2_name):
    # Get the column types of the reference dataframe
    df_types = df.dtypes
    
    # Get the column types of the current dataframe in the list
    df2_types = df2.dtypes
    
    # Check for differences in column types
    differing_columns = df_types[df_types != df2_types]
    
    if not differing_columns.empty:
        #print(f"DataFrame {df2_name} has differing column types:")
        #print(differing_columns)
        
        # Force the differing columns to match the types of df
        for col in differing_columns.index:
            df2[col] = df2[col].astype(df_types[col])
                
    return df2

def do_inference_measures(job):
    data_path = os.path.join(syn_path, job['dir_name'])
    # We'll run the attacks on the full_syn_path
    full_syn_path = os.path.join(data_path, 'full_syn', f'{job["dir_name"]}.parquet')
    df_full_syn = pd.read_parquet(full_syn_path)
    print(f'df_full_syn shape: {df_full_syn.shape}')
    # For stadler and gioni, we do baseline on part_syn_path
    part_syn_path = os.path.join(data_path, 'part_syn', f'{job["dir_name"]}.parquet')
    df_part_syn = pd.read_parquet(part_syn_path)
    print(f'df_part_syn shape: {df_part_syn.shape}')
    # The rows to attack
    test_path = os.path.join(data_path, 'test', f'{job["dir_name"]}.parquet')
    df_test = pd.read_parquet(test_path)
    print(f'df_test shape: {df_test.shape}')
    # We do ALC baseline on part_raw_path
    part_raw_path = os.path.join(data_path, 'part_raw', f'{job["dir_name"]}.parquet')
    df_part_raw = pd.read_parquet(part_raw_path)
    print(f'df_part_raw shape: {df_part_raw.shape}')

    # set aux_cols to all columns except the secret column
    aux_cols = [col for col in df_full_syn.columns if col not in [job['secret']]]
    print(f'aux_cols: {aux_cols}')
    secret_col = job['secret']
    print(f'secret_col: {secret_col}')
    regression = False
    secret_col_type = 'categorical'
    # Because I'm modeling the control and syn dataframes, and because the models
    # don't play well with string or datetime types, I'm just going to convert everthing
    df_full_syn = convert_datetime_to_timestamp(df_full_syn)
    df_part_syn = convert_datetime_to_timestamp(df_part_syn)
    df_part_raw = convert_datetime_to_timestamp(df_part_raw)
    df_test = convert_datetime_to_timestamp(df_test)
    encoders = fit_encoders([df_part_raw, df_full_syn, df_part_syn, df_test])

    df_full_syn = transform_df(df_full_syn, encoders)
    df_part_syn = transform_df(df_part_syn, encoders)
    df_part_raw = transform_df(df_part_raw, encoders)
    df_test = transform_df(df_test, encoders)

    df_part_syn = align_column_types(df_part_raw, df_part_syn, 'df_part_syn')
    df_full_syn = align_column_types(df_part_raw, df_full_syn, 'df_full_syn')
    df_test = align_column_types(df_part_raw, df_test, 'df_test')

    attack_cols = aux_cols + [secret_col]
    print(f'attack_cols: {attack_cols}')
    print("build alc baseline model")
    model_part_raw = build_and_train_model(df_part_raw[attack_cols], secret_col, secret_col_type)
    # model_attack is used to generate a groundhog day type attack
    print("build stadler baseline model")
    model_part_syn = build_and_train_model(df_part_syn[attack_cols], secret_col, secret_col_type)
    print("build stadler attack model")
    model_full_syn = build_and_train_model(df_full_syn[attack_cols], secret_col, secret_col_type)

    exact_matches = df_test[df_test.isin(df_part_raw)].dropna()
    num_exact_matches = exact_matches.shape[0]
    print(f'There are {num_exact_matches} exact matches between df_test and df_part_raw')
    print(exact_matches.head(5))
    exact_matches = df_part_syn[df_part_syn.isin(df_part_raw)].dropna()
    num_exact_matches = exact_matches.shape[0]
    print(f'There are {num_exact_matches} exact matches between df_part_syn and df_part_raw')
    print(exact_matches.head(5))
    exact_matches = df_full_syn[df_full_syn.isin(df_part_raw)].dropna()
    num_exact_matches = exact_matches.shape[0]
    print(f'There are {num_exact_matches} exact matches between df_full_syn and df_part_raw')
    print(exact_matches.head(5))

    num_alc_base_correct = 0
    num_stadler_attack_correct = 0
    num_stadler_base_correct = 0
    num_giomi_attack_correct = 0
    num_giomi_base_correct = 0

    attacks = []
    modal_value = df_part_raw[secret_col].mode().iloc[0]
    num_modal_rows = df_part_raw[df_part_raw[secret_col] == modal_value].shape[0]
    modal_percentage = round(100*(num_modal_rows / len(df_part_raw)), 2)
    for index, row in df_test.iterrows():
        # My old code had the row as a df, so convert here for backwards compatibility
        df_target = row.to_frame().T
        df_target = align_column_types(df_part_raw, df_target, 'df_target')

        print(".", end='', flush=True)
        secret_value = df_target[secret_col].iloc[0]
        # Count the number of rows that contian secret_value in column secret_col
        num_secret_rows = df_part_raw[secret_col].value_counts().get(secret_value, 0)
        secret_percentage = round(100*(num_secret_rows / len(df_part_raw)), 2)
        this_attack = {
            'secret_value': str(secret_value),
            'secret_percentage': secret_percentage,
            'secret_col': secret_col,
            'dataset': job['dir_name'],
            'modal_value': str(modal_value),
            'modal_percentage': modal_percentage,
            'num_known_cols': len(aux_cols),
        }
        # Now get the alc baseline prediction
        try:
            alc_base_pred_value = model_part_raw.predict(df_target.drop(secret_col, axis=1))
            # proba[0] is a list of probability values, indexed by the column values
            proba = model_part_raw.predict_proba(df_target.drop(secret_col, axis=1))
            alc_base_pred_value = alc_base_pred_value[0]
        except Exception as e:
            print(f"A model_part_raw.predict() Error occurred: {e}")
            sys.exit(1)
        # convert model_base_pred_value to a series
        alc_base_pred_value_series = pd.Series(alc_base_pred_value, index=df_target.index)
        alc_base_answer = anonymeter_mods.evaluate_inference_guesses(guesses=alc_base_pred_value_series, secrets=df_target[secret_col], regression=regression).sum()
        if alc_base_answer not in [0,1]:
            print(f"Error: unexpected alc_base_answer {alc_base_answer}")
            sys.exit(1)
        num_alc_base_correct += alc_base_answer
        this_attack['alc_base_pred_value'] = str(alc_base_pred_value)
        this_attack['alc_base_answer'] = int(alc_base_answer)
        this_attack['alc_base_probability'] = float(proba[0][int(alc_base_pred_value)])

        # Now run the stadler attack
        try:
            stadler_attack_pred_value = model_full_syn.predict(df_target.drop(secret_col, axis=1))
            stadler_attack_pred_value = stadler_attack_pred_value[0]
        except Exception as e:
            print(f"A model.predict() Error occurred: {e}")
            sys.exit(1)
        # convert stadler_attack_pred_value to a series
        stadler_attack_pred_value_series = pd.Series(stadler_attack_pred_value, index=df_target.index)
        stadler_attack_answer = anonymeter_mods.evaluate_inference_guesses(guesses=stadler_attack_pred_value_series, secrets=df_target[secret_col], regression=regression).sum()
        if stadler_attack_answer not in [0,1]:
            print(f"Error: unexpected stadler_attack_answer {stadler_attack_answer}")
            sys.exit(1)
        num_stadler_attack_correct += stadler_attack_answer
        this_attack['stadler_attack_pred_value'] = str(stadler_attack_pred_value)
        this_attack['stadler_attack_answer'] = int(stadler_attack_answer)

        # Now run the stadler baseline
        try:
            stadler_base_pred_value = model_part_syn.predict(df_target.drop(secret_col, axis=1))
            stadler_base_pred_value = stadler_base_pred_value[0]
        except Exception as e:
            print(f"A model.predict() Error occurred: {e}")
            sys.exit(1)
        # convert stadler_base_pred_value to a series
        stadler_base_pred_value_series = pd.Series(stadler_base_pred_value, index=df_target.index)
        stadler_base_answer = anonymeter_mods.evaluate_inference_guesses(guesses=stadler_base_pred_value_series, secrets=df_target[secret_col], regression=regression).sum()
        if stadler_base_answer not in [0,1]:
            print(f"Error: unexpected stadler_base_answer {stadler_base_answer}")
            sys.exit(1)
        num_stadler_base_correct += stadler_base_answer
        this_attack['stadler_base_pred_value'] = str(stadler_base_pred_value)
        this_attack['stadler_base_answer'] = int(stadler_base_answer)

        # Run the giomi attack
        ans = anonymeter_mods.run_anonymeter_attack(
                                        targets=df_target,
                                        basis=df_full_syn[attack_cols],
                                        aux_cols=aux_cols,
                                        secret=secret_col,
                                        regression=regression)
        giomi_attack_pred_value_series = ans['guess_series']
        giomi_attack_pred_value = giomi_attack_pred_value_series.iloc[0]
        giomi_attack_answer = anonymeter_mods.evaluate_inference_guesses(guesses=giomi_attack_pred_value_series, secrets=df_target[secret_col], regression=regression).sum()
        if giomi_attack_answer not in [0,1]:
            print(f"Error: unexpected giomi_attack_answer {giomi_attack_answer}")
            sys.exit(1)
        num_giomi_attack_correct += giomi_attack_answer
        this_attack['giomi_attack_pred_value'] = str(giomi_attack_pred_value)
        this_attack['giomi_attack_answer'] = int(giomi_attack_answer)

        # Run the giomi baseline
        ans = anonymeter_mods.run_anonymeter_attack(
                                        targets=df_target,
                                        basis=df_part_syn[attack_cols],
                                        aux_cols=aux_cols,
                                        secret=secret_col,
                                        regression=regression)
        giomi_base_pred_value_series = ans['guess_series']
        giomi_base_pred_value = giomi_base_pred_value_series.iloc[0]
        giomi_base_answer = anonymeter_mods.evaluate_inference_guesses(guesses=giomi_base_pred_value_series, secrets=df_target[secret_col], regression=regression).sum()
        if giomi_base_answer not in [0,1]:
            print(f"Error: unexpected giomi_base_answer {giomi_base_answer}")
            sys.exit(1)
        num_giomi_base_correct += giomi_base_answer
        this_attack['giomi_base_pred_value'] = str(giomi_base_pred_value)
        this_attack['giomi_base_answer'] = int(giomi_base_answer)

        attacks.append(this_attack)
    print(f"\nnum_alc_base_correct: {num_alc_base_correct}\nnum_stadler_attack_correct: {num_stadler_attack_correct}\nnum_stadler_base_correct: {num_stadler_base_correct}\nnum_giomi_attack_correct: {num_giomi_attack_correct}\nnum_giomi_base_correct: {num_giomi_base_correct}")
    return attacks


def measure(job_num):
    with open(os.path.join(base_path, 'measure_jobs.json'), 'r') as f:
        jobs = json.load(f)

    if job_num < 0 or job_num >= len(jobs):
        print(f"Invalid job number: {job_num}")
        return

    job = jobs[job_num]
    instances_path = os.path.join(base_path, 'instances')
    # Make a file_name and file_path
    # make a string that contains the column names in job['columns'] separated by '_'
    file_name = f"{job['dir_name']}.{job_num}.json"
    file_path = os.path.join(instances_path, file_name)
    # check if file_path already exists
    if os.path.exists(file_path):
        print(f"File already exists at {file_path}. Quitting...")
        return
    measures = do_inference_measures(job)
    with open(file_path, 'w') as f:
        json.dump(measures, f, indent=4)

def gather(instances_path):
    measures = []
    # check to see if measures.parquet exists
    if os.path.exists(os.path.join(base_path, 'measures.parquet')):
        # read it as a DataFrame
        print("Reading measures.parquet")
        df = pd.read_parquet(os.path.join(base_path, 'measures.parquet'))
    else:
        all_files = list(os.listdir(instances_path))
        # loop through the index and filename of all_files
        for i, filename in enumerate(all_files):
            if not filename.endswith('.json'):
                print(f"!!!!!!!!!!!!!!! bad filename: {filename}!!!!!!!!!!!!!!!!!!!!!!!")
                continue
            with open(os.path.join(instances_path, filename), 'r') as f:
                if i % 100 == 0:
                    print(f"Reading {i+1} of {len(all_files)} {filename}")
                res = json.load(f)
                measures += res
        print(f"Total measures: {len(measures)}")
        # convert measures to a DataFrame
        df = pd.DataFrame(measures)
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].astype(int)
                except ValueError:
                    try:
                        df[col] = df[col].astype(float)
                    except ValueError:
                        pass
        # print the dtypes of df
        pp.pprint(df.dtypes)
        # save the dataframe to a parquet file
        df.to_parquet(os.path.join(base_path, 'measures.parquet'))
        # save the dataframe to a csv file
        df.to_csv(os.path.join(base_path, 'measures.csv'))
    return df

def update_max_als(max_als, max_info, label, stats):
    als_label = f"{label}_als"
    prec_label = f"{label}_precision"
    cov_label = f"{label}_coverage"
    if stats[als_label] > max_als:
        max_als = stats[als_label]
        if cov_label in stats:
            cov = stats[cov_label]
        else:
            cov = 1.0
        max_info = {'label':label, 'als':max_als, 'precision':stats[prec_label], 'coverage':cov}
    return max_als, max_info

def get_base_pred(df, target_coverage):
    df_copy = df.copy()
    df_copy = df_copy.sort_values(by='model_base_probability', ascending=False)
    num_rows = int(round(target_coverage * len(df)))
    return df_copy.head(num_rows)

def add_to_dump(df, label, slice_name):
    cols = ['secret_value', 'secret_percentage', 'secret_col_type', 'modal_value', 'modal_percentage', 'num_known_cols', 'dataset',]
    val_col = f"{label}_value"
    ans_col = f"{label}_answer"
    cols += [val_col, ans_col]
    df_filtered_1 = df[df[ans_col] == 1]
    path = os.path.join(base_path, 'dumps')
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{slice_name}.{label}.csv")
    df_filtered_1[cols].to_csv(file_path)

def get_basic_stats(stats, df, df_all, info, cov_basis, slice_type, dataset, slice_name = None):
    als = ALScore()
    als_threshold = 0.5
    max_als = -1000
    max_info = {}
    stats['num_attacks'] = len(df)
    stats['slice_type'] = slice_type
    stats['dataset'] = dataset
    stats['avg_num_subsets'] = round(df['num_subsets'].mean(), 2)
    stats['avg_secret_percentage'] = round(df['secret_percentage'].mean(), 2)
    stats['avg_modal_percentage'] = round(df['modal_percentage'].mean(), 2)
    # All of these initial measures have coverage of 1.0
    # The base model prediction needs to be based on a model built from all of the columns,
    # not just the columns known to the attacker.
    stats['model_base_precision'] = round(df_all['model_base_answer'].sum() / len(df_all), 6)
    stats['model_base_coverage'] = 1.0
    stats['model_base_pcc'] = als.pcc(stats['model_base_precision'], stats['model_base_coverage'])
    # Same thing applies to the base meter prediction
    stats['meter_base_precision'] = round(df_all['base_meter_answer'].sum() / len(df_all), 6)
    stats['meter_base_coverage'] = 1.0
    stats['meter_base_pcc'] = als.pcc(stats['meter_base_precision'], stats['meter_base_coverage'])
    # We want to select the best precision-coverage-coefficient between the two baselines, the model
    # baseline and the anonymeter-style baseline (since in any event at this point the coverage is 1.0,
    # this is equivalent to selecting the best precision, but still....)
    stats['model_meter_best_pcc'] = max(stats['model_base_pcc'], stats['meter_base_pcc'])
    # Compute the ALS for the model attack on syndiffix. This is the attack based on the
    # groundhog method part 2
    stats['model_attack_precision'] = round(df['model_attack_answer'].sum() / len(df), 6)
    stats['model_attack_coverage'] = 1.0
    stats['model_attack_als'] = als.alscore(pcc_base=stats['model_meter_best_pcc'],
                                            p_attack=stats['model_attack_precision'],
                                            c_attack=stats['model_attack_coverage'])
    if stats['model_attack_als'] > als_threshold:
        stats['model_attack_problem'] = True
    else:
        stats['model_attack_problem'] = False

    # Compute the ALS for the model attack run over the original data. This has nothing
    # to do with syndiffix, rather we just want to show how ineffective the model attack is
    # We also ignore the meter_base when doing this measure
    stats['model_original_precision'] = round(df['model_original_answer'].sum() / len(df), 6)
    stats['model_original_coverage'] = 1.0
    stats['model_original_als'] = als.alscore(p_base=stats['model_base_precision'],
                                              c_base=stats['model_base_coverage'],
                                              p_attack=stats['model_original_precision'],
                                              c_attack=stats['model_original_coverage'])
    if stats['model_original_als'] > als_threshold:
        stats['model_original_problem'] = True
    else:
        stats['model_original_problem'] = False
    # Now measure the anonymeter-style attack on syndiffix.
    stats['meter_attack_precision'] = round(df['syn_meter_answer'].sum() / len(df), 6)
    stats['meter_attack_coverage'] = 1.0
    stats['meter_attack_als'] = als.alscore(pcc_base=stats['model_meter_best_pcc'],
                                            p_attack=stats['meter_attack_precision'],
                                            c_attack=stats['meter_attack_coverage'])
    if stats['meter_attack_als'] > als_threshold:
        stats['meter_attack_problem'] = True
    else:
        stats['meter_attack_problem'] = False
    max_als, max_info = update_max_als(max_als, max_info, 'meter_attack', stats)
    for v_label in variant_labels:
        for cc_label in col_comb_thresholds.keys():
            syn_meter_label = f"syn_meter_{v_label}_{cc_label}"
            syn_answer = f"{syn_meter_label}_answer"
            syn_precision = f"{syn_meter_label}_precision"
            syn_coverage = f"{syn_meter_label}_coverage"
            syn_pcc = f"{syn_meter_label}_pcc"
            syn_base_used = f"{syn_meter_label}_base_used"
            syn_base_precision = f"{syn_meter_label}_used_base_precision"
            syn_base_coverage = f"{syn_meter_label}_used_base_coverage"
            syn_base_pcc = f"{syn_meter_label}_used_base_pcc"
            syn_als = f"{syn_meter_label}_als"
            syn_problem = f"{syn_meter_label}_problem"
            base_meter_label = f"base_meter_{v_label}_{cc_label}"
            base_meter_answer = f"{base_meter_label}_answer"
            base_meter_precision = f"{base_meter_label}_precision"
            base_meter_coverage = f"{base_meter_label}_coverage"
            base_meter_pcc = f"{base_meter_label}_pcc"
            base_model_label = f"base_model_{v_label}_{cc_label}"
            base_model_precision = f"{base_model_label}_precision"
            base_model_coverage = f"{base_model_label}_coverage"
            base_model_pcc = f"{base_model_label}_pcc"
            # df_pred contains only the rows where predictions were made
            df_syn_pred = df[df[syn_answer] != -1]
            # target_coverage is the coverage we'd like to match from the base model
            target_coverage = len(df_syn_pred) / len(df)
            df_base_model_pred = get_base_pred(df_all, target_coverage)
            df_base_meter_pred = df_all[df_all[base_meter_answer] != -1]
            stats[base_meter_coverage] = 0
            stats[base_meter_precision] = 0
            stats[base_meter_pcc] = 0
            stats[base_model_coverage] = 0
            stats[base_model_precision] = 0
            stats[base_model_pcc] = 0
            stats[syn_base_coverage] = 0
            stats[syn_base_precision] = 0
            stats[syn_base_pcc] = 0
            stats[syn_coverage] = 0
            stats[syn_precision] = 0
            stats[syn_pcc] = 0
            stats[syn_als] = 0
            stats[syn_problem] = False
            if len(df_syn_pred) == 0:
                continue
            # Basing the base precision on the rows where the attack happened to make predictions
            # is not necessarily the right thing to do. What we really want is to find the best base
            # precision given a similar coverage
            if len(df_base_model_pred) > 0:
                stats[base_model_precision] = df_base_model_pred[base_meter_answer].sum() / len(df_base_model_pred)
                stats[base_model_coverage] = len(df_base_model_pred) / cov_basis
                stats[base_model_pcc] = als.pcc(stats[base_model_precision], stats[base_model_coverage])
            if len(df_base_meter_pred) > 0:
                stats[base_meter_precision] = df_base_meter_pred[base_meter_answer].sum() / len(df_base_meter_pred)
                stats[base_meter_coverage] = len(df_base_meter_pred) / cov_basis
                stats[base_meter_pcc] = als.pcc(stats[base_meter_precision], stats[base_meter_coverage])
            pcc_base_use = max(stats['model_meter_best_pcc'], stats[base_model_pcc], stats[base_meter_pcc])
            # We want to record how often each type of base measure was used
            pcc_index = [stats['model_base_pcc'], stats['meter_base_pcc'],
                       stats[base_model_pcc], stats[base_meter_pcc]].index(pcc_base_use)
            pcc_name = ['num_model_base', 'num_meter_base', 'num_subset_model_base', 'num_subset_meter_base'][pcc_index]
            info[pcc_name] += 1

            # We want to record the base precision and coverage associated with the best
            # base pcc (pcc_base_use)
            p_base_use = [stats['model_base_precision'], stats['meter_base_precision'],
                          stats[base_model_precision], stats[base_meter_precision]][pcc_index]
            c_base_use = [stats['model_base_coverage'], stats['meter_base_coverage'],
                          stats[base_model_coverage], stats[base_meter_coverage]][pcc_index]
            stats[syn_base_precision] = round(p_base_use, 4)
            stats[syn_base_coverage] = round(c_base_use, 6)
            stats[syn_base_pcc] = round(pcc_base_use, 4)

            stats[syn_base_used] = pcc_name
            stats[syn_precision] = df_syn_pred[syn_answer].sum() / len(df_syn_pred)
            stats[syn_coverage] = len(df_syn_pred) / cov_basis
            stats[syn_pcc] = als.pcc(stats[syn_precision], stats[syn_coverage])
            stats[syn_als] = als.alscore(pcc_base=pcc_base_use,
                                            p_attack=stats[syn_precision],
                                            c_attack=stats[syn_coverage])
            if stats[syn_als] > als_threshold:
                stats[syn_problem] = True
                add_to_dump(df_syn_pred, syn_meter_label, slice_name)
            else:
                stats[syn_problem] = False
            max_als, max_info = update_max_als(max_als, max_info, syn_meter_label, stats)
    stats['max_als_record'] = max_info
    stats['max_als'] = max_info['als']
    stats['max_precision'] = max_info['precision']	
    stats['max_coverage'] = max_info['coverage']	

def get_by_metric_from_by_slice(stats):
    problem_cases = {}
    num_problem_cases = 0
    for metric in stats['by_slice']['all_results'].keys():
        stats['by_metric'][metric] = {}	
        for slice_key, result in stats['by_slice'].items():
            if metric in result:
                stats['by_metric'][metric][slice_key] = result[metric]
            else:
                continue
            if metric[-3:] == 'als' and metric not in ['model_original_als', 'max_als']:
                if result[metric] > 0.5:
                    num_problem_cases += 1
                    if metric[:-3] not in problem_cases:
                        problem_cases[metric[:-3]] = []
                    problem_cases[metric[:-3]].append(str((slice_key, metric, result[metric])))
                    cov_metric = metric[:-3] + 'coverage'
                    coverage = stats['by_metric'][cov_metric][slice_key]
                    problem_cases[metric[:-3]].append(str((slice_key, cov_metric, coverage)))
                    prec_metric = metric[:-3] + 'precision'
                    precision = stats['by_metric'][prec_metric][slice_key]
                    problem_cases[metric[:-3]].append(str((slice_key, prec_metric, precision)))
                    num_predictions = int(round(coverage * stats['by_metric']['num_attacks'][slice_key]))
                    problem_cases[metric[:-3]].append(str(('num_predictions', num_predictions)))
    stats['problem_cases'] = problem_cases
    stats['info']['num_problem_cases'] = num_problem_cases


def df_compare(df1: pd.DataFrame, df2: pd.DataFrame) -> int:
    # Ensure the dataframes have the same columns and index
    df1, df2 = df1.align(df2)

    # Create a boolean mask where each row is True if the row in df1 differs from the row in df2
    mask = ~(df1 == df2).all(axis=1)

    # Count the number of True values in the mask
    num_differing_rows = mask.sum()

    return num_differing_rows

def set_model_base_predictions(df, thresh):
    df_copy = df.copy()
    num_pred_value_changes = 0
    num_answer_changes_to_1 = 0
    num_answer_changes_to_0 = 0
    for index, row in df_copy.iterrows():
        if row['modal_percentage'] > thresh:
            if row['model_base_pred_value'] != row['modal_value']:
                num_pred_value_changes += 1
                #print(row['model_base_pred_value'], row['modal_value'])
                df_copy.at[index, 'model_base_pred_value'] = row['modal_value']
            if row['modal_value'] == row['secret_value']:
                if row['model_base_answer'] != 1:
                    num_answer_changes_to_1 += 1
                df_copy.at[index, 'model_base_answer'] = 1
            else:
                if row['model_base_answer'] != 0:
                    num_answer_changes_to_0 += 1
                df_copy.at[index, 'model_base_answer'] = 0
    return df_copy

def run_stats_for_subsets(stats, df, df_all):
    stats['by_slice']['all_results'] = {}
    get_basic_stats(stats['by_slice']['all_results'], df, df_all, stats['info'], len(df), 'all', 'all')
    # make a new df that contains only rows where 'secret_col_type' is 'categorical'
    df_cat = df[df['secret_col_type'] == 'categorical']
    df_cat_copy = df_cat.copy()
    stats['by_slice']['categorical_results'] = {}
    get_basic_stats(stats['by_slice']['categorical_results'], df_cat_copy, df_all, stats['info'], len(df_cat_copy), 'all', 'all')
    #df_cat_copy['percentile_bin'] = pd.qcut(df_cat_copy['secret_percentage'], q=10, labels=False)
    df_cat_copy['percentile_bin'] = pd.cut(df_cat_copy['modal_percentage'], bins=10, labels=False)
    df_all['percentile_bin'] = pd.cut(df_all['modal_percentage'], bins=10, labels=False)
    for bin_value, df_bin in df_cat_copy.groupby('percentile_bin'):
        df_all_bin = df_all[df_all['percentile_bin'] == bin_value]
        average_modal_percentage = round(df_bin['modal_percentage'].mean(), 2)
        slice_name = f"cat_modal_percentage_{average_modal_percentage}"
        stats['by_slice'][slice_name] = {}
        get_basic_stats(stats['by_slice'][slice_name], df_bin, df_all_bin, stats['info'], len(df_cat_copy), 'modal_percentage', 'all', slice_name=slice_name)
    for bin_value, df_bin in df_cat_copy.groupby('dataset'):
        df_all_bin = df_all[df_all['dataset'] == bin_value]
        slice_name = f"cat_dataset_{bin_value}"
        stats['by_slice'][slice_name] = {}
        get_basic_stats(stats['by_slice'][slice_name], df_bin, df_all_bin, stats['info'], len(df_bin), 'dataset', bin_value, slice_name=slice_name)
    #pp.pprint(stats)
    get_by_metric_from_by_slice(stats)

def do_stats_dict(stats_path):
    df = gather(instances_path=os.path.join(base_path, 'instances'))

    print(f"df has shape {df.shape} and columns:")
    print(df.columns)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].astype(int)
            except ValueError:
                try:
                    df[col] = df[col].astype(float)
                except ValueError:
                    pass
    pp.pprint(df.dtypes)
    print(df.head())
    stats = {}
    # Maybe do some basic stats here???
    return stats

def make_df_from_stats(stats):
    collects = {
                'model_base': 'other',
                'meter_base': 'other',
                'model_attack': 'other',
                'model_original': 'other',
                'meter_attack': 'attack',
                'base_meter_vanilla_thresh_0': 'other',
                'syn_meter_vanilla_thresh_0': 'attack',
                'base_meter_vanilla_thresh_50': 'other',
                'syn_meter_vanilla_thresh_50': 'attack',
                'base_meter_vanilla_thresh_90': 'other',
                'syn_meter_vanilla_thresh_90': 'attack',
                'base_meter_modal_thresh_0': 'other',
                'syn_meter_modal_thresh_0': 'attack',
                'base_meter_modal_thresh_50': 'other',
                'syn_meter_modal_thresh_50': 'attack',
                'base_meter_modal_thresh_90': 'other',
                'syn_meter_modal_thresh_90': 'attack',
                'base_meter_modal_50_thresh_0': 'other',
                'syn_meter_modal_50_thresh_0': 'attack',
                'base_meter_modal_50_thresh_50': 'other',
                'syn_meter_modal_50_thresh_50': 'attack',
                'base_meter_modal_50_thresh_90': 'other',
                'syn_meter_modal_50_thresh_90': 'attack',
                'base_meter_modal_90_thresh_0': 'other',
                'syn_meter_modal_90_thresh_0': 'attack',
                'base_meter_modal_90_thresh_50': 'other',
                'syn_meter_modal_90_thresh_50': 'attack',
                'base_meter_modal_90_thresh_90': 'other',
                'syn_meter_modal_90_thresh_90': 'attack',
        }
    dat = []
    for num_known_key, stuff in stats.items():
        num_known = num_known_config[num_known_key]
        for slice_key, slice_results in stuff['by_slice'].items():
            for collect, info_type in collects.items():
                row = {
                        'num_known': num_known,
                        'info_type': info_type,
                        'num_attacks': slice_results['num_attacks'],
                        'slice_type': slice_results['slice_type'], 
                        'dataset': slice_results['dataset'], 
                        'avg_num_subsets': slice_results['avg_num_subsets'], 
                        'avg_secret_percentage': slice_results['avg_secret_percentage'], 
                        'avg_modal_percentage': slice_results['avg_modal_percentage'], 
                }
                row['metric'] = collect
                col_prec = f"{collect}_precision"
                col_als = f"{collect}_als"
                col_coverage = f"{collect}_coverage"
                col_use_prec = f"{collect}_base_precision"
                row['precision'] = slice_results[col_prec]
                row['als'] = 0.0
                row['coverage'] = 1.0
                if col_use_prec in slice_results:
                    row['base_precision'] = slice_results[col_use_prec]
                else:
                    row['base_precision'] = 0.0
                if col_als in slice_results:
                    row['als'] = slice_results[col_als]
                if col_coverage in slice_results:
                    row['coverage'] = slice_results[col_coverage]
                dat.append(row)
    df = pd.DataFrame(dat)
    return df

def make_prec(df, columns):
    # Group the DataFrame by the 'grouper' column
    grouped = df.groupby('grouper')
    
    # Initialize the new DataFrame with the specified columns
    df_prec = grouped[columns].first().reset_index()
    
    # Add the 'group_count' column
    df_prec['group_count'] = grouped.size().values
    
    # Add the precision columns
    df_prec['stadler_atk_prec'] = grouped['stadler_attack_answer'].sum().values / df_prec['group_count']
    df_prec['giomi_atk_prec'] = grouped['giomi_attack_answer'].sum().values / df_prec['group_count']
    df_prec['stadler_base_prec'] = grouped['stadler_base_answer'].sum().values / df_prec['group_count']
    df_prec['giomi_base_prec'] = grouped['giomi_base_answer'].sum().values / df_prec['group_count']
    df_prec['alc_base_prec'] = grouped['alc_base_answer'].sum().values / df_prec['group_count']
    
    # Add the 'stadler_alc' column
    df_prec['stadler_alc'] = (
        0.5 * (df_prec['stadler_atk_prec'] - df_prec['stadler_base_prec']) +
        0.5 * ((df_prec['stadler_atk_prec'] + 0.000001 - df_prec['stadler_base_prec']) / (1.000001 - df_prec['stadler_base_prec']))
    )
    
    # Add the 'stadler_our_alc' column
    df_prec['stadler_our_alc'] = (
        0.5 * (df_prec['stadler_atk_prec'] - df_prec['alc_base_prec']) +
        0.5 * ((df_prec['stadler_atk_prec'] + 0.000001 - df_prec['alc_base_prec']) / (1.000001 - df_prec['alc_base_prec']))
    )
    
    # Add the 'giomi_alc' column
    df_prec['giomi_alc'] = (
        0.5 * (df_prec['giomi_atk_prec'] - df_prec['giomi_base_prec']) +
        0.5 * ((df_prec['giomi_atk_prec'] + 0.000001 - df_prec['giomi_base_prec']) / (1.000001 - df_prec['giomi_base_prec']))
    )

    # Add the 'giomi_our_alc' column
    df_prec['giomi_our_alc'] = (
        0.5 * (df_prec['giomi_atk_prec'] - df_prec['alc_base_prec']) +
        0.5 * ((df_prec['giomi_atk_prec'] + 0.000001 - df_prec['alc_base_prec']) / (1.000001 - df_prec['alc_base_prec']))
    )
    
    # Add the improvement columns
    df_prec['alc_base_stadler_improve'] = df_prec['alc_base_prec'] - df_prec['stadler_base_prec']
    df_prec['alc_base_giomi_improve'] = df_prec['alc_base_prec'] - df_prec['giomi_base_prec']
    return df_prec

def grouper(df, columns):
    # Group the DataFrame by the specified columns
    grouped = df.groupby(columns)
    # Create a new column 'grouper' with distinct integers for each group
    df['grouper'] = grouped.ngroup()
    return make_prec(df, columns)


def plot_basic(df, name):
    # Define the columns to be plotted
    columns_to_plot = [
        'stadler_atk_prec',
        'stadler_alc',
        'stadler_our_alc',
        'giomi_atk_prec',
        'giomi_alc',
        'giomi_our_alc',
    ]

    # Create a new DataFrame for plotting
    plot_df = pd.melt(df[columns_to_plot], var_name='Metric', value_name='Value')
    
    # Create a mapping for the yticklabels
    label_mapping = {
        'stadler_atk_prec': 'Attack Precision\n(Stadler)',
        'stadler_alc': 'Prior ALC\n(Stadler)',
        'stadler_our_alc': 'Our ALC\n(Stadler)',
        'giomi_atk_prec': 'Attack Precision\n(Giomi)',
        'giomi_alc': 'Prior ALC\n(Giomi)',
        'giomi_our_alc': 'Our ALC\n(Giomi)'
    }
    
    # Map the yticklabels
    plot_df['Metric'] = plot_df['Metric'].map(label_mapping)
    
    # Create the boxplot
    plt.figure(figsize=(6, 3.5))
    ax = sns.boxplot(x='Value', y='Metric', data=plot_df, palette=['#FF9999', '#FF9999', '#FF9999', '#66B2FF', '#66B2FF', '#66B2FF'])

    plt.xlim(-1.05, 1.05)
    
    # Add a dashed vertical line at x=0
    plt.axvline(x=0, color='black', linestyle='--')

    ax.set_ylabel('')
    ax.set_xlabel('')
    
    # Add a legend for the two colors
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor='#FF9999', edgecolor='black', label='Stadler'),
                      Patch(facecolor='#66B2FF', edgecolor='black', label='Giomi')]
    plt.legend(handles=legend_handles, loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot as PNG and PDF
    plt.savefig(os.path.join(base_path, 'plots', f'basic_{name}.png'))
    plt.savefig(os.path.join(base_path, 'plots', f'basic_{name}.pdf'))

def make_custom_bins(df, column, bins):
    # Create bins based on the specified bin ranges
    df['bin_ranges'] = pd.cut(df[column], bins=bins, include_lowest=True)
    
    # Create a list of bin ranges as strings
    bin_ranges = [f'({bins[i]:.2f}, {bins[i+1]:.2f}]' for i in range(len(bins) - 1)]
    
    # Create a dictionary to map bin codes to bin ranges
    bin_ranges_dict = {i: bin_ranges[i] for i in range(len(bin_ranges))}
    
    # Map the bin ranges to the bin labels
    df['bin_ranges'] = df['bin_ranges'].cat.codes.map(bin_ranges_dict)
    
    # Add the 'grouper' column with the integer number of the group
    df['grouper'] = df['bin_ranges'].astype('category').cat.codes
    
    return df

def make_bins(df, column, num_bins=10):
    # Create bins of equal count
    df['bin_ranges'], bin_edges = pd.qcut(df[column], q=num_bins, retbins=True, duplicates='drop')
    bin_ranges = [f'({bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}]' for i in range(len(bin_edges) - 1)]
    bin_ranges_dict = {i: bin_ranges[i] for i in range(len(bin_ranges))}
    df['bin_ranges'] = df['bin_ranges'].cat.codes.map(bin_ranges_dict)
    df['grouper'] = df['bin_ranges'].astype('category').cat.codes
    return df

def plot_alc_improve_by_secret_percent(df, tag):
    # Define the columns to be plotted
    columns_to_plot = ['alc_base_stadler_improve', 'alc_base_giomi_improve']
    
    # Get the unique bin ranges in the order they appear in df
    bin_ranges = df['bin_ranges'].unique()
    
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(6, 3.5))
    
    # Define the bar width
    bar_width = 0.4
    
    # Define the positions of the bars
    y_pos = np.arange(len(bin_ranges))
    
    # Create a new DataFrame for plotting
    plot_df = pd.DataFrame({
        'bin_ranges': np.tile(bin_ranges, 2),
        'Improvement': np.concatenate([
            df.groupby('bin_ranges')['alc_base_stadler_improve'].mean().loc[bin_ranges].values,
            df.groupby('bin_ranges')['alc_base_giomi_improve'].mean().loc[bin_ranges].values
        ]),
        'Baseline': ['Stadler'] * len(bin_ranges) + ['Giomi'] * len(bin_ranges)
    })
    
    # Plot the bars using Seaborn
    sns.barplot(x='Improvement', y='bin_ranges', hue='Baseline', data=plot_df, palette=['#FF9999', '#66B2FF'], ax=ax)
    
    # Set the y-axis labels
    y_labels = [f"{bin_range}\n({group_count})" for bin_range, group_count in zip(bin_ranges, df['group_count'])]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    
    # Set the labels and title
    ax.set_xlabel('Improvement of our baseline over prior baselines')
    ax.set_ylabel('Fraction of rows with secret value\n(sample count)')
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Add a legend
    ax.legend()
    
    # Save the plot as PNG and PDF
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'plots', f'alc_improve_by_secret_percent_{tag}.png'))
    plt.savefig(os.path.join(base_path, 'plots', f'alc_improve_by_secret_percent_{tag}.pdf'))

def plot_alc_improve_by_secret_percent_old(df, tag):
    # Define the columns to be plotted
    columns_to_plot = ['alc_base_stadler_improve', 'alc_base_giomi_improve']
    
    # Get the unique bin ranges in the order they appear in df
    bin_ranges = df['bin_ranges'].unique()
    
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Define the bar width
    bar_width = 0.4
    
    # Define the positions of the bars
    y_pos = np.arange(len(bin_ranges))
    
    # Plot the bars for 'alc_base_stadler_improve'
    ax.barh(y_pos - bar_width/2, df.groupby('bin_ranges')['alc_base_stadler_improve'].mean().loc[bin_ranges], 
            height=bar_width, label='Stadler', color='#FF9999')
    
    # Plot the bars for 'alc_base_giomi_improve'
    ax.barh(y_pos + bar_width/2, df.groupby('bin_ranges')['alc_base_giomi_improve'].mean().loc[bin_ranges], 
            height=bar_width, label='Giomi', color='#66B2FF')
    
    y_labels = [f"{bin_range}\n({group_count})" for bin_range, group_count in zip(bin_ranges, df['group_count'])]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    
    # Set the labels and title
    ax.set_xlabel('Improvement of our baseline over prior baselines')
    ax.set_ylabel('Fraction of rows with secret value\n(sample count)')
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Add a legend
    ax.legend()
    
    # Save the plot as PNG and PDF
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'plots', f'alc_improve_by_secret_percent_{tag}.png'))
    plt.savefig(os.path.join(base_path, 'plots', f'alc_improve_by_secret_percent_{tag}.pdf'))

def do_plots():
    df = gather(instances_path=os.path.join(base_path, 'instances'))
    print(df.columns)

    min_value = df['secret_percentage'].min()
    max_value = df['secret_percentage'].max()
    print(f"Min value: {min_value}")
    print(f"Max value: {max_value}")

    bins = [0,10,20,50,100]
    df_temp = make_custom_bins(df, 'secret_percentage', bins)
    df_temp = make_prec(df_temp, ['bin_ranges'])
    print(df_temp.columns)
    print(df_temp[['bin_ranges', 'group_count', 'grouper']].to_string())
    plot_alc_improve_by_secret_percent(df_temp, 'custom')

    df_secret_percent = make_bins(df, 'secret_percentage', num_bins=5)
    df_secret_percent = make_prec(df_secret_percent, ['bin_ranges'])
    plot_alc_improve_by_secret_percent(df_secret_percent, 'equal')

    df_secret_col = grouper(df, ['dataset', 'secret_col'])
    plot_basic(df_secret_col, 'by_secret_col')

    df_temp = df[df['secret_percentage'] < 0.2].copy()
    df_temp = grouper(df_temp, ['dataset', 'secret_col'])
    plot_basic(df_temp, 'by_secret_col_low')

def do_tests():
    if find_most_frequent_value([1, 2, 2, 3, 3, 3], 0.5) != 3:
        print("failed 1")
    if find_most_frequent_value([1, 2, 2, 3, 3, 3], 0.6) is not None:
        print("failed 2")
    if find_most_frequent_value([], 0.5) is not None:
        print("failed 3")
    if find_most_frequent_value([1, 1, 1, 1, 1], 0.2) != 1:
        print("failed 4")

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    make_one_syn_parser = subparsers.add_parser('make_one_syn', help="Run make_one_syn with an integer")
    make_one_syn_parser.add_argument("job_num", type=int, help="An integer to pass to make_one_syn()")
    measure_parser = subparsers.add_parser('measure', help="Run measure with an integer")
    measure_parser.add_argument("job_num", type=int, help="An integer to pass to make_syn()")
    subparsers.add_parser('make_syn', help="Run make_syn")
    subparsers.add_parser('config', help="Run make_config")
    subparsers.add_parser('test', help="Run test")
    subparsers.add_parser('plots', help="Run plots")
    
    args = parser.parse_args()

    if args.command == 'measure':
        measure(args.job_num)
    elif args.command == 'make_one_syn':
        make_one_syn(args.job_num)
    elif args.command == 'make_syn':
        make_syn()
    elif args.command == 'config':
        make_config()
    elif args.command == 'test':
        do_tests()
    elif args.command == 'plots':
        do_plots()
    else:
        print(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()