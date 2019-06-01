# Clean Root Ad data
# 1. Adds two additional columns: 'local_hour' and 'state' to 'raw_df.pickle'
# and saves the new file as 'Time_Update.pickle
# 'local_hour' stores int which represents the hour in the local area
# 'state' stores string which is the abbreviation of states (eg. OH)

# Import libraries
import pandas as pd
import numpy as np
from dateutil import parser

# Directory containing project files
DIR = r'.'

# Read zipcode file and delete 3 or 4 digit zipcodes
ZIP = pd.read_csv(DIR + '/zipcode/zipcode.csv')
zipdf = ZIP[['zip','state','timezone','dst']]
zipdf = zipdf[zipdf.zip > 9999]

# Load data
df = pd.read_pickle(DIR + '/raw_df.pickle')

df = df[df.category != '-1'] # Removing Bad Data

df.reset_index(inplace=True)

df = pd.merge(
        df, 
        zipdf, 
        left_on='geo_zip', 
        right_on='zip',
        how='left'
        )

# If no corresponding zip is found, set it to be in central time
df.loc[df.loc[:,'timezone'].isnull(),'timezone'] = -6
df.loc[df.loc[:,'dst'].isnull(),'dst'] = 1
df.loc[df.loc[:,'state'].isnull(),'state'] = 'Unknown'

# Add column 'local_hour'
hours = df['hour']
hours_list = hours.tolist()
hour_num = list(int(x[0:2]) for x in hours_list)
df['local_hour'] = hour_num + df.timezone + df.dst

# Keep day and day_of_week consisent with hour
day_list = list(df.loc[df.loc[:,'local_hour']<0,'day'])
day_list = list(int(x)-1 for x in day_list)
df.loc[df.loc[:,'local_hour']<0,'day'] = day_list
# Update day
day_string = []
for x in df.loc[:,'day']:
   if x!=0:
       day_string.append('April'+ str(x) +',2019')
   else:
       day_string.append('March 31,2019')
dow = list(parser.parse(str).strftime("%A") for str in day_string)
df.loc[:,'day_of_week'] = dow
df.loc[df.loc[:,'local_hour']<0,'local_hour'] +=24


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

# Load data
df = pd.read_pickle(DIR + '/clean_df.pickle')

# Create modeling dataset
# Drop columns that will not be used in the model
df = df.drop(['geo_zip', 'bid_timestamp_utc', 'segments', 'category', 
              'index', 'Unnamed: 0', 'timezone', 'inventory_source'], axis=1)

df_train = df[df.day <= 21]
df_test = df[df.day >= 22]

# platform_device_model has thousands of values. use the 20 most frequent and
# group others into an 'Other' category
top_devices = df_train[
        'platform_device_model'].value_counts().head(20).index.values.tolist()

# group uncommon platform_device_model into one category
df_train['platform_device_model'] = np.where(
        df_train['platform_device_model'].isin(top_devices),
        df_train['platform_device_model'],
        'Other')

df_test['platform_device_model'] = np.where(
        df_test['platform_device_model'].isin(top_devices),
        df_test['platform_device_model'],
        'Other')

# platform_device_make has dozens of values, some very rare. use the 10 most 
# frequent and group others into an 'Other' category
top_makes = df_train[
        'platform_device_make'].value_counts().head(10).index.values.tolist()

# group uncommon platform_device_model into one category
df_train['platform_device_make'] = np.where(
        df_train['platform_device_make'].isin(top_makes),
        df_train['platform_device_make'],
        'Other')

df_test['platform_device_make'] = np.where(
        df_test['platform_device_make'].isin(top_makes),
        df_test['platform_device_make'],
        'Other')

# Some states are rare- group them into an 'Other' category
bottom_states = df[
        'state'].value_counts().tail(6).index.values.tolist()

# group uncommon platform_device_model into one category
df_train['state'] = np.where(
        ~df_train['state'].isin(bottom_states),
        df_train['state'],
        'Other')

df_test['state'] = np.where(
        ~df_test['state'].isin(bottom_states),
        df_test['state'],
        'Other')

# Some platform_device_screen_size are rare- group them into an 'Other' category
bottom_ss = df[
        'platform_device_screen_size'].value_counts().tail(3).index.values.tolist()

# group uncommon platform_device_model into one category
df_train['platform_device_screen_size'] = np.where(
        ~df_train['platform_device_screen_size'].isin(bottom_ss),
        df_train['platform_device_screen_size'],
        'Other')

df_test['platform_device_screen_size'] = np.where(
        ~df_test['platform_device_screen_size'].isin(bottom_ss),
        df_test['platform_device_screen_size'],
        'Other')

# Some platform_bandwidth are rare- group them into an 'Other' category
bottom_band = df[
        'platform_bandwidth'].value_counts().tail(4).index.values.tolist()

# group uncommon platform_device_model into one category
df_train['platform_bandwidth'] = np.where(
        ~df_train['platform_bandwidth'].isin(bottom_band),
        df_train['platform_bandwidth'],
        'Other')

df_test['platform_bandwidth'] = np.where(
        ~df_test['platform_bandwidth'].isin(bottom_band),
        df_test['platform_bandwidth'],
        'Other')

# Dummy-encode categorical columns
df_train = pd.get_dummies(
        df_train, 
        columns=set(df.select_dtypes(include=['object']).columns) - 
        set(['auction_id', 'zip', 'bid_floor']))

df_test = pd.get_dummies(
        df_test, 
        columns=set(df.select_dtypes(include=['object']).columns) - 
        set(['auction_id', 'zip', 'bid_floor']))

# Save test and train dataset as pickle
df_train.to_pickle(DIR + '/df_train.pickle')
df_test.to_pickle(DIR + '/df_test.pickle')
