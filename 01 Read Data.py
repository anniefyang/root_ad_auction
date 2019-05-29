# Parse Root Ad data

# Import libraries
import pandas as pd
import math as m
from os import listdir

# Directory containing project files
DIR = r'C:/Users/Annie/Documents/Python Bootcamp Final Project'

# Name of folder in DIR containing csv data
FOLDER_CSV = 'root_ad_auction_dataset_all_data'

# Get names of all files in directory
file_names = listdir(DIR + '/' + FOLDER_CSV)

# Columns to drop 
# month, year, and app_bundle have the same value for all observations
# creative_size encodes the same information as creative_type, so drop 
# creative_size
drop_cols = ['month', 'year', 'app_bundle', 'creative_size']

# Initialize the dataframe with the first file, consisting of just column names
df = pd.read_csv(DIR + '/' + FOLDER_CSV + '/2019-04-00.csv')

# Drop unnecessary columns
df = df.drop(drop_cols, axis=1)

# Drop the first file name (headers only) from file_names
file_names.remove('2019-04-00.csv')

# Read the remaining data and append to df
# This takes about 5 minutes
for i in file_names:
    # Read daily csv
    temp = pd.read_csv(DIR + '/' + FOLDER_CSV + '/' + i)
    
    # Optional: sample 1/100 of daily data
    # Useful for testing
    # temp = temp.sample(m.floor(temp.shape[0]/100), random_state = 0)
    
    # Drop unnecessary columns
    temp = temp.drop(drop_cols, axis=1)
    
    # Split daily csv into two dataframes: click and no click
    # Note: observations for which clicks = 0 & installs = 1 should actually
    # read clicks = 1 & installs = 1
    click = temp.loc[(temp.clicks == 1) | (temp.installs == 1)]
    click.loc[:, 'clicks'] = 1.0
    no_click = temp[(temp.clicks == 0) & (temp.installs == 0)]
    
    # Downsample no click to be the same size as click
    # For reproducibility, set random_state to 0
    df = pd.concat([
            df, 
            click,
            no_click.sample(n = click.shape[0], random_state = 0)
            ], axis=0)
    
    # Visually track progress by printing the file name
    print(i)

# Save dataset as pickle
df.to_pickle(DIR + '/raw_df.pickle')

# Test reading data back in
test = pd.read_pickle(DIR + '/raw_df.pickle')
