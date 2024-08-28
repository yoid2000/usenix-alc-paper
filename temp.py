import pandas as pd

def convert_parquet_to_csv(parquet_file, csv_file):
    # Read the Parquet file
    df = pd.read_parquet(parquet_file)
    
    # Write the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)

# Specify the input and output file paths
parquet_file = 'national2019_partraw.parquet'
csv_file = 'national2019_partraw.csv'

# Convert the Parquet file to CSV
convert_parquet_to_csv(parquet_file, csv_file)

print(f"Converted {parquet_file} to {csv_file}")