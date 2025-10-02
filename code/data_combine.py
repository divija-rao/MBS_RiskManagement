# Import necessary libraries
import pandas as pd
import os

# Paths to the individual CSVs
files = [
    'data/raw/merged_loans_part_1.csv',
    'data/raw/merged_loans_part_2.csv',
    'data/raw/merged_loans_part_3.csv'
]

# Read all CSVs into a list of DataFrames
dfs = [pd.read_csv(f) for f in files]

# Combine them vertically (row-wise)
combined_df = pd.concat(dfs, ignore_index=True)

# Output path
output_path = 'data/raw/loans_combined.csv'

# Save combined CSV
combined_df.to_csv(output_path, index=False)

print(f"Combined CSV saved to {output_path}")
