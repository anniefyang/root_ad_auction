# Clean Root Ad data
# 1. Adds two additional columns: 'local_hour' and 'state' to 'raw_df.pickle'
# and saves the new file as 'Time_Update.pickle
# 'local_hour' stores int which represents the hour in the local area
# 'state' stores string which is the abbreviation of states (eg. OH)

# Import libraries
import pandas as pd
import numpy as np

# Directory containing project files
DIR = r'C:/Users/Annie/Documents/Python Bootcamp Final Project'

# Read zipcode file and delete 3 or 4 digit zipcodes
ZIP = pd.read_csv(DIR + '/zipcode/zipcode.csv')
zipdf = ZIP[['zip','state','timezone','dst']]
zipdf = zipdf[zipdf.zip > 9999]

# Load data
df = pd.read_pickle(DIR + '/raw_df.pickle')

df.reset_index(inplace=True)

df = pd.merge(
        df, 
        zipdf, 
        left_on='geo_zip', 
        right_on='zip',
        how='left'
        )

# Initialize column 'local_hour' as int with the same value as 'hour'
hours = df['hour']
hours_list = hours.tolist()
hour_num = list(int(x[0:2]) for x in hours_list)
df['local_hour'] = hour_num

df = df.assign(local_hour=df.local_hour+df.timezone+df.dst)

df['day'] = np.where(df['local_hour'] < 0, df['day'] - 1, df['day'])
df['local_hour'] = np.where(df['local_hour'] < 0, df['local_hour'] + 24, df['local_hour'])

df = df.assign(diff_hour=df.timezone+df.dst)

# TODO: append day of week

# Identify columns with NA or NaN
df.columns[df.isna().any()]

# Replace NA/NaN with the string 'NA'
df = df.fillna(value = {'category': 'NA',
#                        'geo_zip': 'NA',
                        'platform_bandwidth': 'NA',
                        'platform_carrier': 'NA',
                        'platform_device_screen_size': 'NA',
                        'creative_type': 'NA'})
    
# Reformat the category column to be one category per column    
expand_category = df['category'].str.split(',', expand = True) 
expand_category = pd.concat([df.auction_id, expand_category], axis=1)

expand_category = pd.melt(expand_category, id_vars = ['auction_id']) 

expand_category = expand_category[expand_category['value'].notnull()]

expand_category = expand_category.pivot_table(
        index='auction_id',
        columns='value',
        aggfunc='size'
        )

expand_category.columns = ['category_' + str(col) for col in expand_category.columns]
expand_category = expand_category.fillna(0)

df = pd.merge(
        df, 
        expand_category, 
        left_on='auction_id', 
        right_index=True
        )

# Reformat the segment column to be one segment per column    
df['segments'] = df['segments'].str.replace(r"\[","")
df['segments'] = df['segments'].str.replace(r"\]","")
    
expand_segment = df['segments'].str.split(', ', expand = True)
expand_segment = pd.concat([df.auction_id, expand_segment], axis=1)

expand_segment = pd.melt(expand_segment, id_vars = ['auction_id']) 

expand_segment = expand_segment[expand_segment['value'].notnull()]

expand_segment = expand_segment.pivot_table(
        index='auction_id',
        columns='value',
        aggfunc='size'
        )

expand_segment.columns = ['segment_' + str(col) for col in expand_segment.columns]
expand_segment = expand_segment.fillna(0)

df = pd.merge(
        df, 
        expand_segment, 
        left_on='auction_id', 
        right_index=True
        )

# Save dataset as pickle
df.to_pickle(DIR + '/clean_df.pickle')
