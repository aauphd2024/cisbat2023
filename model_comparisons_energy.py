# -*- coding: utf-8 -*-

'''
kahean@build.aau.dk
KAMILLA HEIMAR ANDERSEN
'''

#%%
##############################################################################
### PACKAGES ###
##############################################################################

import pandas as pd
import json
import matplotlib.pyplot as plt
import datetime as dt
import time
import numpy as np
from datetime import date, datetime, timedelta  # Date and time stamp generation
from scipy.interpolate import interp1d
import plotly.express as px
import os
import matplotlib.dates as mdates
import plotly.express as px
import seaborn as sns
import statistics
from sklearn.cluster import DBSCAN
import plotly.express as px
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score 

#%%
##############################################################################
### FOLDER LOCATION ###
##############################################################################

# Initialization
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

os.chdir(dname)
current_working_folder = os.getcwd()
print(current_working_folder)

#%%
##############################################################################
### SOME DT ACTUAL ENERGY DATA ###
##############################################################################

# load data and convert object to datetime
actual = 'alldata_apartments_5min.csv'
df = pd.read_csv(actual)
df['DateTime'] = pd.to_datetime(df['DateTime'], format = '%d-%m-%y %H:%M')
print(df.dtypes)
df_describe = df.describe()

#%%

# simple stats
clean_data = df.copy()

# power
power_threshold_high = 9
power_threshold_low = 0

# energy
energy_threshold_high = 1
energy_threshold_low = 0

# Get the length of the df
clean_data_length = len(clean_data)

# columns to be replaced
columns_to_replace_power = ['heating_power_kw_sttv', 'heating_power_kw_stth', 'heating_power_kw_1tv', 'heating_power_kw_1th', 'heating_power_kw_2th', 'heating_power_kw_2tv']
columns_to_replace_energy = ['heating_energy_kwh_sttv', 'heating_energy_kwh_stth', 'heating_energy_kwh_1tv', 'heating_energy_kwh_1th', 'heating_energy_kwh_2th', 'heating_energy_kwh_2tv']

# Replace values above 9 and below 0 with NaNs POWER
clean_data.loc[0:clean_data_length-1, columns_to_replace_power] = np.where(
    (clean_data[columns_to_replace_power] > power_threshold_high) | (clean_data[columns_to_replace_power] < power_threshold_low), 
    np.nan, clean_data[columns_to_replace_power]
)

# Replace values above 1 and below 0 with NaNs ENERGY
clean_data.loc[0:clean_data_length-1, columns_to_replace_energy] = np.where(
    (clean_data[columns_to_replace_energy] > energy_threshold_high) | (clean_data[columns_to_replace_energy] < energy_threshold_low), 
    np.nan, clean_data[columns_to_replace_energy]
)

# simple stats
clean_data_describe = clean_data.describe
sum_energy_power = pd.DataFrame(clean_data.sum(), columns=['sum_energy_power'])

# filter columns in to power and energy
power_columns = clean_data.filter(regex='DateTime|power')
energy_columns = clean_data.filter(regex='DateTime|energy')

# sum power and energy
power_sum = pd.DataFrame(power_columns.sum(), columns=['sum_power'])

# Compute the sum of each row
row_sum_power = power_columns.sum(axis=1)

# Add the new column to the DataFrame
power_columns['sum'] = row_sum_power

# Compute the sum of each row
row_sum_energy = energy_columns.sum(axis=1)

# Add the new column to the DataFrame
energy_columns['sum'] = row_sum_energy

#%%
###############################################################################
###                 FOCUS ON THE ENERGY USE RESAMPLING                      ###
###############################################################################

'Sum the columns of all 6 apartments'
#energy_columns.plot(x = 'DateTime', y = 'sum', title = 'Heating energy use [kWh]')
#power_columns.plot(x = 'DateTime', y = 'sum', title = 'Heating power [kW]')

#energy_columns.plot(x = 'DateTime', y = ['heating_energy_kwh_sttv', 'heating_energy_kwh_stth', 'heating_energy_kwh_1tv', 'heating_energy_kwh_1th', 'heating_energy_kwh_2th', 'heating_energy_kwh_2tv'], title = 'Model 1 - all apartments', ylabel = 'Heating energy use [kWh]', xlabel = 'Date and Time (5-minute resolution)', grid = True)

"5-min"

energy_columns['month'] = energy_columns['DateTime'].dt.month

# Filter for data between January and May
df_jan_may = energy_columns[(energy_columns['month'] >= 1) & (energy_columns['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = energy_columns[(energy_columns['month'] >= 9) | (energy_columns['month'] <= 1)]

# Concatenate the two dataframes
energy_columns_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
energy_columns_filt = energy_columns_filt.reset_index(drop=True)

print(energy_columns_filt.sum())

'Hourly resample'
hour_energy_columns = energy_columns.resample('60T', on='DateTime', label='right', closed='right').sum()  # 2T for resampling every 2 minutes, DateTime is set as index in result df
hour_energy_columns.insert(0, "DateTime", hour_energy_columns.index)  # Insert back DateTime as column on first position
hour_energy_columns.index=range(len(hour_energy_columns))  # Reset index column from 0 to len df

#hour_energy_columns.plot(x = 'DateTime', y = ['heating_energy_kwh_sttv', 'heating_energy_kwh_stth', 'heating_energy_kwh_1tv', 'heating_energy_kwh_1th', 'heating_energy_kwh_2th', 'heating_energy_kwh_2tv'], title = 'Model 1 - all apartments', ylabel = 'Average hourlt heating energy use [kWh]', xlabel = 'Date and Time (60-minute resolution)', grid = True)
#hour_energy_columns.plot(x = 'DateTime', y = 'sum', title = 'Model 1 - avg. heating energy use', ylabel = 'Average hourly heating energy use [kWh/h]', xlabel = 'Date and Time (60-minute resolution)', grid = True)

hour_energy_columns['month'] = hour_energy_columns['DateTime'].dt.month

# Filter for data between January and May
df_jan_may = hour_energy_columns[(hour_energy_columns['month'] >= 1) & (hour_energy_columns['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = hour_energy_columns[(hour_energy_columns['month'] >= 9) | (hour_energy_columns['month'] <= 1)]

# Concatenate the two dataframes
hour_energy_columns_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
hour_energy_columns_filt = hour_energy_columns_filt.reset_index(drop=True)

print(hour_energy_columns_filt.sum())

'Daily resample'
day_energy_columns = energy_columns.resample('1D', on='DateTime', label='right', closed='right').sum()  # 2T for resampling every 2 minutes, DateTime is set as index in result df
day_energy_columns.insert(0, "DateTime", day_energy_columns.index)  # Insert back DateTime as column on first position
day_energy_columns.index=range(len(day_energy_columns))  # Reset index column from 0 to len df

#day_energy_columns.plot(x = 'DateTime', y = ['heating_energy_kwh_sttv', 'heating_energy_kwh_stth', 'heating_energy_kwh_1tv', 'heating_energy_kwh_1th', 'heating_energy_kwh_2th', 'heating_energy_kwh_2tv'], title = 'Model 1 - all apartments', ylabel = 'Heating energy use [kWh]', xlabel = 'Date and Time (Daily resolution)', grid = True)
#day_energy_columns.plot(x = 'DateTime', y = 'sum', title = 'Model 1 - avg. heating energy use', ylabel = 'Average daily heating energy use [kWh/day]', xlabel = 'Date and Time (24-hour resolution)', grid = True)

day_energy_columns['month'] = day_energy_columns['DateTime'].dt.month

# Filter for data between January and May
df_jan_may = day_energy_columns[(day_energy_columns['month'] >= 1) & (day_energy_columns['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = day_energy_columns[(day_energy_columns['month'] >= 9) | (day_energy_columns['month'] <= 1)]

# Concatenate the two dataframes
day_energy_columns_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
day_energy_columns_filt = day_energy_columns_filt.reset_index(drop=True)

print(day_energy_columns_filt.sum())

'Month resample'
month_energy_columns = energy_columns.resample('1M', on='DateTime', label='right', closed='right').sum()  # 1H for resampling every 2 minutes, DateTime is set as index in result df
month_energy_columns.insert(0, "DateTime", month_energy_columns.index)  # Insert back DateTime as column on first position
month_energy_columns.index=range(len(month_energy_columns))  # Reset index column from 0 to len df
#df_x_resampled_month.to_csv('df_x_resampled_month.csv')

month_energy_columns['month'] = month_energy_columns['DateTime'].dt.month

# Filter for data between January and May
df_jan_may = month_energy_columns[(month_energy_columns['month'] >= 1) & (month_energy_columns['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = month_energy_columns[(month_energy_columns['month'] >= 9) | (month_energy_columns['month'] <= 1)]

# Concatenate the two dataframes
month_energy_columns_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
month_energy_columns_filt = month_energy_columns_filt.reset_index(drop=True)

month_energy_columns_filt = month_energy_columns_filt.drop([5, 6]).reset_index(drop=True)

print(month_energy_columns_filt.sum())

#%%
###############################################################################
###                           CLEAN IDA ICE FILES                           ###
###############################################################################

"model A"
modela = 'modela.csv'
modela = pd.read_csv(modela)
modela['DateTimea'] = pd.to_datetime(modela['DateTime'], format = '%d-%m-%y %H:%M')
print(modela.dtypes)
modela_describe = modela.describe()

'5-min resample'
modela_resampled = modela.resample('5T', on='DateTimea', label='right', closed='right').mean()  # 1H for resampling every 2 minutes, DateTime is set as index in result df
modela_resampled.insert(0, "DateTimea", modela_resampled.index)  # Insert back DateTime as column on first position
modela_resampled.index=range(len(modela_resampled))  # Reset index column from 0 to len df
modela_resampled['sum kw_a'] = modela_resampled['sum kw_a'].interpolate()
#df_x_resampled.to_csv('model1.csv')
print(modela_resampled.sum())

'Convert to energy use'
# energy use kwh per 5 min
modela_resampled['energy_use_a'] = modela_resampled['sum kw_a'] * (5*60)/3600

modela_resampled['month'] = modela_resampled['DateTimea'].dt.month

# Filter for data between January and May
df_jan_may = modela_resampled[(modela_resampled['month'] >= 1) & (modela_resampled['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = modela_resampled[(modela_resampled['month'] >= 9) | (modela_resampled['month'] <= 1)]

# Concatenate the two dataframes
modela_resampled_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
modela_resampled_filt = modela_resampled_filt.reset_index(drop=True)

print(modela_resampled_filt.sum())

'1-hour resample'
# energy use kwh/h
modela_resampled_hour = modela_resampled.resample('1H', on='DateTimea', label='right', closed='right').sum()  # 1H for resampling every 2 minutes, DateTime is set as index in result df
modela_resampled_hour.insert(0, "DateTimea", modela_resampled_hour.index)  # Insert back DateTime as column on first position
modela_resampled_hour.index=range(len(modela_resampled_hour))  # Reset index column from 0 to len df
#df_x_resampled_hour.to_csv('dataset_resampled_hour.csv')

modela_resampled_hour['month'] = modela_resampled_hour['DateTimea'].dt.month

# Filter for data between January and May
df_jan_may = modela_resampled_hour[(modela_resampled_hour['month'] >= 1) & (modela_resampled_hour['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = modela_resampled_hour[(modela_resampled_hour['month'] >= 9) | (modela_resampled_hour['month'] <= 1)]

# Concatenate the two dataframes
modela_resampled_hour_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
modela_resampled_hour_filt = modela_resampled_hour_filt.reset_index(drop=True)

print(modela_resampled_hour_filt.sum())

'Day resample'
# enegy use kwh/day
modela_resampled_day = modela_resampled.resample('1D', on='DateTimea', label='right', closed='right').sum()  # 1H for resampling every 2 minutes, DateTime is set as index in result df
modela_resampled_day.insert(0, "DateTimea", modela_resampled_day.index)  # Insert back DateTime as column on first position
modela_resampled_day.index=range(len(modela_resampled_day))  # Reset index column from 0 to len df
#df_x_resampled_day.to_csv('df_x_resampled_day.csv')
print(modela_resampled_day.sum())

modela_resampled_day['month'] = modela_resampled_day['DateTimea'].dt.month

# Filter for data between January and May
df_jan_may = modela_resampled_day[(modela_resampled_day['month'] >= 1) & (modela_resampled_day['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = modela_resampled_day[(modela_resampled_day['month'] >= 9) | (modela_resampled_day['month'] <= 1)]

# Concatenate the two dataframes
modela_resampled_day_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
modela_resampled_day_filt = modela_resampled_day_filt.reset_index(drop=True)

print(modela_resampled_day_filt.sum())

'Month resample'
modela_resampled_month = modela_resampled.resample('1M', on='DateTimea', label='right', closed='right').sum()  # 1H for resampling every 2 minutes, DateTime is set as index in result df
modela_resampled_month.insert(0, "DateTimea", modela_resampled_month.index)  # Insert back DateTime as column on first position
modela_resampled_month.index=range(len(modela_resampled_month))  # Reset index column from 0 to len df
#df_x_resampled_month.to_csv('df_x_resampled_month.csv')
print(modela_resampled_month.sum())

modela_resampled_month['month'] = modela_resampled_month['DateTimea'].dt.month

# Filter for data between January and May
df_jan_may = modela_resampled_month[(modela_resampled_month['month'] >= 1) & (modela_resampled_month['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = modela_resampled_month[(modela_resampled_month['month'] >= 9) | (modela_resampled_month['month'] <= 1)]

# Concatenate the two dataframes
modela_resampled_month_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
modela_resampled_month_filt = modela_resampled_month_filt.reset_index(drop=True)

#drop index 5 and 6 which are duplicated
modela_resampled_month_filt = modela_resampled_month_filt.drop([5, 6]).reset_index(drop=True)

print(modela_resampled_month_filt.sum())

#%%

"model B"
modelb = 'modelb.csv'
modelb = pd.read_csv(modelb)
modelb['DateTimeb'] = pd.to_datetime(modelb['DateTime'], format = '%d-%m-%y %H:%M')
print(modelb.dtypes)
modelb_describe = modelb.describe()

'5-min resample'
modelb_resampled = modelb.resample('5T', on='DateTimeb', label='right', closed='right').mean()  # 1H for resampling every 2 minutes, DateTime is set as index in result df
modelb_resampled.insert(0, "DateTimeb", modelb_resampled.index)  # Insert back DateTime as column on first position
modelb_resampled.index=range(len(modelb_resampled))  # Reset index column from 0 to len df
modelb_resampled['sum kw_b'] = modelb_resampled['sum kw_b'].interpolate()
#df_x_resampled.to_csv('model1.csv')
print(modelb_resampled.sum())

'Convert to energy use'
# energy use kwh per 5 min
modelb_resampled['energy_use_b'] = modelb_resampled['sum kw_b'] * (5*60)/3600

modelb_resampled['month'] = modelb_resampled['DateTimeb'].dt.month

# Filter for data between January and May
df_jan_may = modelb_resampled[(modelb_resampled['month'] >= 1) & (modelb_resampled['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = modelb_resampled[(modelb_resampled['month'] >= 9) | (modelb_resampled['month'] <= 1)]

# Concatenate the two dataframes
modelb_resampled_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
modelb_resampled_filt = modelb_resampled_filt.reset_index(drop=True)

print(modelb_resampled_filt.sum())

'1-hour resample'
# energy use kwh/h
modelb_resampled_hour = modelb_resampled.resample('1H', on='DateTimeb', label='right', closed='right').sum()  # 1H for resampling every 2 minutes, DateTime is set as index in result df
modelb_resampled_hour.insert(0, "DateTimeb", modelb_resampled_hour.index)  # Insert back DateTime as column on first position
modelb_resampled_hour.index=range(len(modelb_resampled_hour))  # Reset index column from 0 to len df
#df_x_resampled_hour.to_csv('dataset_resampled_hour.csv')
print(modelb_resampled_hour.sum())

modelb_resampled_hour['month'] = modelb_resampled_hour['DateTimeb'].dt.month

# Filter for data between January and May
df_jan_may = modelb_resampled_hour[(modelb_resampled_hour['month'] >= 1) & (modelb_resampled_hour['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = modelb_resampled_hour[(modelb_resampled_hour['month'] >= 9) | (modelb_resampled_hour['month'] <= 1)]

# Concatenate the two dataframes
modelb_resampled_hour_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
modelb_resampled_hour_filt = modelb_resampled_hour_filt.reset_index(drop=True)

print(modelb_resampled_hour_filt.sum())

#modelb_resampled_hour_filt.to_csv('modelb_hour.csv')

'Day resample'
# enegy use kwh/day
modelb_resampled_day = modelb_resampled.resample('1D', on='DateTimeb', label='right', closed='right').sum()  # 1H for resampling every 2 minutes, DateTime is set as index in result df
modelb_resampled_day.insert(0, "DateTimeb", modelb_resampled_day.index)  # Insert back DateTime as column on first position
modelb_resampled_day.index=range(len(modelb_resampled_day))  # Reset index column from 0 to len df
#df_x_resampled_day.to_csv('df_x_resampled_day.csv')
print(modelb_resampled_day.sum())

modelb_resampled_day['month'] = modelb_resampled_day['DateTimeb'].dt.month

# Filter for data between January and May
df_jan_may = modelb_resampled_day[(modelb_resampled_day['month'] >= 1) & (modelb_resampled_day['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = modelb_resampled_day[(modelb_resampled_day['month'] >= 9) | (modelb_resampled_day['month'] <= 1)]

# Concatenate the two dataframes
modelb_resampled_day_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
modelb_resampled_day_filt = modelb_resampled_day_filt.reset_index(drop=True)

print(modelb_resampled_day_filt.sum())

'Month resample'
modelb_resampled_month = modelb_resampled.resample('1M', on='DateTimeb', label='right', closed='right').sum()  # 1H for resampling every 2 minutes, DateTime is set as index in result df
modelb_resampled_month.insert(0, "DateTimeb", modelb_resampled_month.index)  # Insert back DateTime as column on first position
modelb_resampled_month.index=range(len(modelb_resampled_month))  # Reset index column from 0 to len df
#df_x_resampled_month.to_csv('df_x_resampled_month.csv')
print(modelb_resampled_month.sum())

modelb_resampled_month['month'] = modelb_resampled_month['DateTimeb'].dt.month

# Filter for data between January and May
df_jan_may = modelb_resampled_month[(modelb_resampled_month['month'] >= 1) & (modelb_resampled_month['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = modelb_resampled_month[(modelb_resampled_month['month'] >= 9) | (modelb_resampled_month['month'] <= 1)]

# Concatenate the two dataframes
modelb_resampled_month_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
modelb_resampled_month_filt = modelb_resampled_month_filt.reset_index(drop=True)

#drop index 5 and 6 which are duplicated
modelb_resampled_month_filt = modelb_resampled_month_filt.drop([5, 6]).reset_index(drop=True)

print(modelb_resampled_month_filt.sum())

#%%

"model C"
modelc = 'modelc.csv'
modelc = pd.read_csv(modelc)
modelc['DateTimec'] = pd.to_datetime(modelc['DateTime'], format = '%d-%m-%y %H:%M')
print(modelc.dtypes)
modelc_describe = modelc.describe()

'5-min resample'
modelc_resampled = modelc.resample('5T', on='DateTimec', label='right', closed='right').mean()  # 1H for resampling every 2 minutes, DateTime is set as index in result df
modelc_resampled.insert(0, "DateTimec", modelc_resampled.index)  # Insert back DateTime as column on first position
modelc_resampled.index=range(len(modelc_resampled))  # Reset index column from 0 to len df
modelc_resampled['sum kw_c'] = modelc_resampled['sum kw_c'].interpolate()
#df_x_resampled.to_csv('model1.csv')
print(modelc_resampled.sum())

'Convert to energy use'
# energy use kwh per 5 min
modelc_resampled['energy_use_c'] = modelc_resampled['sum kw_c'] * (5*60)/3600

modelc_resampled['month'] = modelc_resampled['DateTimec'].dt.month

# Filter for data between January and May
df_jan_may = modelc_resampled[(modelc_resampled['month'] >= 1) & (modelc_resampled['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = modelc_resampled[(modelc_resampled['month'] >= 9) | (modelc_resampled['month'] <= 1)]

# Concatenate the two dataframes
modelc_resampled_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
modelc_resampled_filt = modelc_resampled_filt.reset_index(drop=True)

print(modelc_resampled_filt.sum())

'1-hour resample'
# energy use kwh/h
modelc_resampled_hour = modelc_resampled_filt.resample('1H', on='DateTimec', label='right', closed='right').sum()  # 1H for resampling every 2 minutes, DateTime is set as index in result df
modelc_resampled_hour.insert(0, "DateTimec", modelc_resampled_hour.index)  # Insert back DateTime as column on first position
modelc_resampled_hour.index=range(len(modelc_resampled_hour))  # Reset index column from 0 to len df
#df_x_resampled_hour.to_csv('dataset_resampled_hour.csv')
print(modelc_resampled_hour.sum())

modelc_resampled_hour['month'] = modelc_resampled_hour['DateTimec'].dt.month

# Filter for data between January and May
df_jan_may = modelc_resampled_hour[(modelc_resampled_hour['month'] >= 1) & (modelc_resampled_hour['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = modelc_resampled_hour[(modelc_resampled_hour['month'] >= 9) | (modelc_resampled_hour['month'] <= 1)]

# Concatenate the two dataframes
modelc_resampled_hour_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
modelc_resampled_hour_filt = modelc_resampled_hour_filt.reset_index(drop=True)

print(modelc_resampled_hour_filt.sum())

'Day resample'
# enegy use kwh/day
modelc_resampled_day = modelc_resampled_filt.resample('1D', on='DateTimec', label='right', closed='right').sum()  # 1H for resampling every 2 minutes, DateTime is set as index in result df
modelc_resampled_day.insert(0, "DateTimec", modelc_resampled_day.index)  # Insert back DateTime as column on first position
modelc_resampled_day.index=range(len(modelc_resampled_day))  # Reset index column from 0 to len df
#df_x_resampled_day.to_csv('df_x_resampled_day.csv')
print(modelc_resampled_day.sum())

modelc_resampled_day['month'] = modelc_resampled_day['DateTimec'].dt.month

# Filter for data between January and May
df_jan_may = modelc_resampled_day[(modelc_resampled_day['month'] >= 1) & (modelc_resampled_day['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = modelc_resampled_day[(modelc_resampled_day['month'] >= 9) | (modelc_resampled_day['month'] <= 1)]

# Concatenate the two dataframes
modelc_resampled_day_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
modelc_resampled_day_filt = modelc_resampled_day_filt.reset_index(drop=True)

print(modelc_resampled_day_filt.sum())

'Month resample'
modelc_resampled_month = modelc_resampled_filt.resample('1M', on='DateTimec', label='right', closed='right').sum()  # 1H for resampling every 2 minutes, DateTime is set as index in result df
modelc_resampled_month.insert(0, "DateTimec", modelc_resampled_month.index)  # Insert back DateTime as column on first position
modelc_resampled_month.index=range(len(modelc_resampled_month))  # Reset index column from 0 to len df
#df_x_resampled_month.to_csv('df_x_resampled_month.csv')
print(modelc_resampled_month.sum())

modelc_resampled_month['month'] = modelc_resampled_month['DateTimec'].dt.month

# Filter for data between January and May
df_jan_may = modelc_resampled_month[(modelc_resampled_month['month'] >= 1) & (modelc_resampled_month['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = modelc_resampled_month[(modelc_resampled_month['month'] >= 9) | (modelc_resampled_month['month'] <= 1)]

# Concatenate the two dataframes
modelc_resampled_month_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
modelc_resampled_month_filt = modelc_resampled_month_filt.reset_index(drop=True)

#drop index 5 and 6 which are duplicated
modelc_resampled_month_filt = modelc_resampled_month_filt.drop([5, 6]).reset_index(drop=True)

print(modelc_resampled_month_filt.sum())

#%%

"model D"
modeld = 'modeld.csv'
modeld = pd.read_csv(modeld)
modeld['DateTimed'] = pd.to_datetime(modeld['DateTime'], format = '%d-%m-%y %H:%M')
print(modeld.dtypes)
modeld_describe = modeld.describe()

'5-min resample'
modeld_resampled = modeld.resample('5T', on='DateTimed', label='right', closed='right').mean()  # 1H for resampling every 2 minutes, DateTime is set as index in result df
modeld_resampled.insert(0, "DateTimed", modeld_resampled.index)  # Insert back DateTime as column on first position
modeld_resampled.index=range(len(modeld_resampled))  # Reset index column from 0 to len df
modeld_resampled['sum kw_d'] = modeld_resampled['sum kw_d'].interpolate()
#df_x_resampled.to_csv('model1.csv')
print(modeld_resampled.sum())

'Convert to energy use'
# energy use kwh per 5 min
modeld_resampled['energy_use_d'] = modeld_resampled['sum kw_d'] * (5*60)/3600

modeld_resampled['month'] = modeld_resampled['DateTimed'].dt.month

# Filter for data between January and May
df_jan_may = modeld_resampled[(modeld_resampled['month'] >= 1) & (modeld_resampled['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = modeld_resampled[(modeld_resampled['month'] >= 9) | (modeld_resampled['month'] <= 1)]

# Concatenate the two dataframes
modeld_resampled_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
modeld_resampled_filt = modeld_resampled_filt.reset_index(drop=True)

print(modeld_resampled_filt.sum())

'1-hour resample'
# energy use kwh/h
modeld_resampled_hour = modeld_resampled.resample('1H', on='DateTimed', label='right', closed='right').sum()  # 1H for resampling every 2 minutes, DateTime is set as index in result df
modeld_resampled_hour.insert(0, "DateTimed", modeld_resampled_hour.index)  # Insert back DateTime as column on first position
modeld_resampled_hour.index=range(len(modeld_resampled_hour))  # Reset index column from 0 to len df
#df_x_resampled_hour.to_csv('dataset_resampled_hour.csv')
print(modeld_resampled_hour.sum())

modeld_resampled_hour['month'] = modeld_resampled_hour['DateTimed'].dt.month

# Filter for data between January and May
df_jan_may = modeld_resampled_hour[(modeld_resampled_hour['month'] >= 1) & (modeld_resampled_hour['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = modeld_resampled_hour[(modeld_resampled_hour['month'] >= 9) | (modeld_resampled_hour['month'] <= 1)]

# Concatenate the two dataframes
modeld_resampled_hour_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
modeld_resampled_hour_filt = modeld_resampled_hour_filt.reset_index(drop=True)

print(modeld_resampled_hour_filt.sum())

'Day resample'
# enegy use kwh/day
modeld_resampled_day = modeld_resampled.resample('1D', on='DateTimed', label='right', closed='right').sum()  # 1H for resampling every 2 minutes, DateTime is set as index in result df
modeld_resampled_day.insert(0, "DateTimed", modeld_resampled_day.index)  # Insert back DateTime as column on first position
modeld_resampled_day.index=range(len(modeld_resampled_day))  # Reset index column from 0 to len df
#df_x_resampled_day.to_csv('df_x_resampled_day.csv')
print(modeld_resampled_day.sum())

modeld_resampled_day['month'] = modeld_resampled_day['DateTimed'].dt.month

# Filter for data between January and May
df_jan_may = modeld_resampled_day[(modeld_resampled_day['month'] >= 1) & (modeld_resampled_day['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = modeld_resampled_day[(modeld_resampled_day['month'] >= 9) | (modeld_resampled_day['month'] <= 1)]

# Concatenate the two dataframes
modeld_resampled_day_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
modeld_resampled_day_filt = modeld_resampled_day_filt.reset_index(drop=True)

print(modeld_resampled_day_filt.sum())

'Month resample'
modeld_resampled_month = modeld_resampled.resample('1M', on='DateTimed', label='right', closed='right').sum()  # 1H for resampling every 2 minutes, DateTime is set as index in result df
modeld_resampled_month.insert(0, "DateTimed", modeld_resampled_month.index)  # Insert back DateTime as column on first position
modeld_resampled_month.index=range(len(modeld_resampled_month))  # Reset index column from 0 to len df
#df_x_resampled_month.to_csv('df_x_resampled_month.csv')
print(modeld_resampled_month.sum())

modeld_resampled_month['month'] = modeld_resampled_month['DateTimed'].dt.month

# Filter for data between January and May
df_jan_may = modeld_resampled_month[(modeld_resampled_month['month'] >= 1) & (modeld_resampled_month['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = modeld_resampled_month[(modeld_resampled_month['month'] >= 9) | (modeld_resampled_month['month'] <= 1)]

# Concatenate the two dataframes
modeld_resampled_month_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
modeld_resampled_month_filt = modeld_resampled_month_filt.reset_index(drop=True)

#drop index 5 and 6 which are duplicated
modeld_resampled_month_filt = modeld_resampled_month_filt.drop([5, 6]).reset_index(drop=True)

print(modeld_resampled_month_filt.sum())

#%%
###############################################################################
###                                 FIGURES                                 ###
###############################################################################

"5 min. res."
plot_comparison = pd.concat([energy_columns_filt, modela_resampled_filt, modelc_resampled_filt], axis = 1)
print(energy_columns_filt['sum'].sum())
print(modela_resampled_filt['energy_use_a'].sum())
print(modelc_resampled_filt['energy_use_c'].sum())

plot_comparison.plot(x= 'DateTimea', y= ['sum', 'energy_use_a', 'energy_use_c'],  title = 'Comparison of models a, Jan. 22 - Jan. 23', color = ['red', 'black', 'blue', 'green', 'yellow'], grid = True, legend = True, ylabel = 'Heating energy use [kWh/5-min.]', xlabel = 'Time [5 min. resolution]')
plt.legend(['Reference', 'Model A', 'Model C'])#, loc = 'top')

#%%

"1 hour res."
plot_comparison_hour = pd.concat([hour_energy_columns_filt, modela_resampled_hour_filt, modelb_resampled_hour_filt, modelc_resampled_hour_filt, modeld_resampled_hour_filt], axis = 1)
print(hour_energy_columns_filt['sum'].sum())
#print(modela_resampled_hour_filt['energy_use_a'].sum())
print(modelb_resampled_hour_filt['sum kw_b'].sum())
#print(modelc_resampled_hour_filt['energy_use_c'].sum())
print(modeld_resampled_hour_filt['sum kw_d'].sum())

plot_comparison_hour.plot(x= 'DateTimea', y= ['sum','energy_use_a', 'energy_use_b' , 'energy_use_c', 'energy_use_d'],  title = 'Comparison of models b, Jan. 22 - Jan. 23', color = ['red', 'black', 'blue', 'green', 'yellow'], grid = True, legend = True, ylabel = 'Heating energy use [kWh/h]', xlabel = 'Time [1 hour resolution]')
#plt.ylim([0, 1.6])
plt.legend(['Reference', 'Model A', 'Model B', 'Model C', 'Model D'])#, loc = 'top')

#%%

"1 day res."
plot_comparison_day = pd.concat([day_energy_columns_filt, modela_resampled_day_filt, modelb_resampled_day_filt, modelc_resampled_day_filt, modeld_resampled_day_filt], axis = 1)
print(day_energy_columns_filt['sum'].sum())
print(modela_resampled_day_filt['energy_use_a'].sum())
print(modelb_resampled_day_filt['energy_use_b'].sum())
print(modelc_resampled_day_filt['energy_use_c'].sum())
print(modeld_resampled_day_filt['energy_use_d'].sum())

plot_comparison_day.plot(x= 'DateTimea', y= ['sum','energy_use_a', 'energy_use_b' , 'energy_use_c', 'energy_use_d'],  title = 'Comparison of models b, Jan. 22 - Jan. 23', color = ['red', 'black', 'blue', 'green', 'yellow'], grid = True, legend = True, ylabel = 'Heating energy use [kWh/day]', xlabel = 'Time [Daily resolution]')
#plt.ylim([0, 1.6])
plt.legend(['Reference', 'Model A', 'Model B', 'Model C', 'Model D'])#, loc = 'top')

#%%

"1 month res."
plot_comparison_month = pd.concat([month_energy_columns_filt, modela_resampled_month_filt, modelb_resampled_month_filt, modelc_resampled_month_filt, modeld_resampled_month_filt], axis = 1)
print(month_energy_columns_filt['sum'].sum())
print(modela_resampled_month_filt['energy_use_a'].sum())
print(modelb_resampled_month_filt['energy_use_b'].sum())
print(modelc_resampled_month_filt['energy_use_c'].sum())
print(modeld_resampled_month_filt['energy_use_d'].sum())

plot_comparison_month.plot(x = 'DateTimea', y= ['sum','energy_use_a', 'energy_use_b' , 'energy_use_c', 'energy_use_d'],  title = 'Comparison of models 1c and 4, Feb. - Dec. 2022', color = ['red', 'black', 'blue', 'green', 'yellow'], grid = True, legend = True, ylabel = 'Sum of heating energy use [kWh/month]', xlabel = 'Time [Monthly resolution]')
plt.legend(['Reference', 'Model A', 'Model B', 'Model C', 'Model D'])#, loc = 'top')

#%%

"load duration curves"

"5-min. res."

df_load_duration_5min = plot_comparison[['sum' ,'energy_use_a', 'energy_use_c']]

# set the columns for which to create load duration curves
columns = ['sum' ,'energy_use_a', 'energy_use_c']

# sort the DataFrame by the first column in descending order
df_sorted = plot_comparison[columns].sort_values(columns[0], ascending=False)

# create a new column 'Rank' with the load duration rank
df_sorted['Rank'] = range(1, len(df_sorted) + 1)

# set the colors for each parameter
colors = ['royalblue', 'green', 'orange']

# plot the load duration curve for the first column with the specified color
plt.plot(df_sorted['Rank'], df_sorted[columns[0]], color=colors[0], label=columns[0])

# loop through each additional column
for i, col in enumerate(columns[1:]):
    # sort the DataFrame by the current column in descending order
    df_sorted = df_sorted.sort_values(col, ascending=False)

    # create a new column 'Rank' with the load duration rank
    df_sorted['Rank'] = range(1, len(df_sorted) + 1)

    # plot the load duration curve for the current column with the specified color
    plt.plot(df_sorted['Rank'], df_sorted[col], color=colors[i+1], label=col)

# set the plot labels and legend
plt.xlabel('Rank [Length of 5-min. resolution data]')
plt.ylabel('Energy use [kWh/5-min.]')
plt.title('Load Duration Curve 5 min. resolution')
plt.grid()
plt.legend(['Reference', 'Model A', 'Model C',])#, loc = 'top')

# display the plot
plt.show()

#%%

"1 hour res."

df_load_duration_60min = plot_comparison_hour[['sum' , 'energy_use_b' , 'energy_use_d']]

# set the columns for which to create load duration curves
columns = ['sum' , 'energy_use_b' , 'energy_use_d']

# sort the DataFrame by the first column in descending order
df_sorted = plot_comparison_hour[columns].sort_values(columns[0], ascending=False)

# create a new column 'Rank' with the load duration rank
df_sorted['Rank'] = range(1, len(df_sorted) + 1)

# set the colors for each parameter
colors = ['royalblue', 'violet', 'forestgreen']

# plot the load duration curve for the first column with the specified color
plt.plot(df_sorted['Rank'], df_sorted[columns[0]], color=colors[0], label=columns[0])

# loop through each additional column
for i, col in enumerate(columns[1:]):
    # sort the DataFrame by the current column in descending order
    df_sorted = df_sorted.sort_values(col, ascending=False)

    # create a new column 'Rank' with the load duration rank
    df_sorted['Rank'] = range(1, len(df_sorted) + 1)

    # plot the load duration curve for the current column with the specified color
    plt.plot(df_sorted['Rank'], df_sorted[col], color=colors[i+1], label=col)

# set the plot labels and legend
plt.xlabel('Time [Length of hourly resolution data]')
plt.ylabel('Energy use [kWh/h]')
plt.title('Load Duration Curve hourly resolution')
plt.grid()
plt.legend(['Reference', 'Model B', 'Model D'])#, loc = 'top')

# display the plot
plt.show()

#%%

"Daily res."

df_load_duration_60min = plot_comparison_day[['sum' , 'energy_use_b' , 'energy_use_d']]

# set the columns for which to create load duration curves
columns = ['sum' , 'energy_use_b' , 'energy_use_d']

# sort the DataFrame by the first column in descending order
df_sorted = plot_comparison_day[columns].sort_values(columns[0], ascending=False)

# create a new column 'Rank' with the load duration rank
df_sorted['Rank'] = range(1, len(df_sorted) + 1)

# set the colors for each parameter
colors = ['royalblue', 'violet', 'forestgreen']

# plot the load duration curve for the first column with the specified color
plt.plot(df_sorted['Rank'], df_sorted[columns[0]], color=colors[0], label=columns[0])

# loop through each additional column
for i, col in enumerate(columns[1:]):
    # sort the DataFrame by the current column in descending order
    df_sorted = df_sorted.sort_values(col, ascending=False)

    # create a new column 'Rank' with the load duration rank
    df_sorted['Rank'] = range(1, len(df_sorted) + 1)

    # plot the load duration curve for the current column with the specified color
    plt.plot(df_sorted['Rank'], df_sorted[col], color=colors[i+1], label=col)

# set the plot labels and legend
plt.xlabel('Rank [Length of daily resolution data]')
plt.ylabel('Energy use [kWh/day]')
plt.title('Load Duration Curve daily resolution')
plt.grid()
plt.legend(['Reference', 'Model B', 'Model D'])#, loc = 'top')

# display the plot
plt.show()

#%%

"Monthly res."

df_load_duration_month = plot_comparison_month[['sum' , 'energy_use_b' , 'energy_use_d']]

# set the columns for which to create load duration curves
columns = ['sum' , 'energy_use_b' , 'energy_use_d']

# sort the DataFrame by the first column in descending order
df_sorted = plot_comparison_month[columns].sort_values(columns[0], ascending=False)

# create a new column 'Rank' with the load duration rank
df_sorted['Rank'] = range(1, len(df_sorted) + 1)

# set the colors for each parameter
colors = ['royalblue', 'violet', 'forestgreen']

# plot the load duration curve for the first column with the specified color
plt.plot(df_sorted['Rank'], df_sorted[columns[0]], color=colors[0], label=columns[0])

# loop through each additional column
for i, col in enumerate(columns[1:]):
    # sort the DataFrame by the current column in descending order
    df_sorted = df_sorted.sort_values(col, ascending=False)

    # create a new column 'Rank' with the load duration rank
    df_sorted['Rank'] = range(1, len(df_sorted) + 1)

    # plot the load duration curve for the current column with the specified color
    plt.plot(df_sorted['Rank'], df_sorted[col], color=colors[i+1], label=col)

# set the plot labels and legend
plt.xlabel('Rank [Length of monthly resolution data]')
plt.ylabel('Energy use [kWh/month]')
plt.title('Load Duration Curve monthly resolution')
plt.grid()
plt.legend(['Reference', 'Model B', 'Model D'])#, loc = 'top')

# display the plot
plt.show()

#%%
###############################################################################
###                     TEMP AND SOLAR RAD. DATA                            ###
###############################################################################

# WEATHER DATA FULL YEAR IMPORT WITH NO SEASON 2022 - 2023 data #
weather = ('weatherdata_your_location.csv')
weather['Time'] = pd.to_datetime(weather['Time'], format='%Y-%m-%dT%H:%M:%S+00:00')
# Convert to the desired format
weather['Time'] = weather['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
weather['Time'] = pd.to_datetime(weather['Time'], format = '%Y-%m-%d %H:%M:%S')

weather_resampled = weather.resample('60T', on='Time', label='right', closed='right').mean() 
#data4_resampled.fillna(value='missing', inplace=True)  # fill NaN values with 'missing'
weather_resampled.insert(0, "Time", weather_resampled.index)
weather_resampled.index = range(len(weather_resampled))

# interpolate missing values
weather_resampled['mean_radiation'] = weather_resampled['mean_radiation'].interpolate()
weather_resampled['mean_temp'] = weather_resampled['mean_temp'].interpolate()
weather_resampled['mean_relative_hum'] = weather_resampled['mean_relative_hum'].interpolate()

weather_resampled['month'] = weather_resampled['Time'].dt.month

# Filter for data between January and May
df_jan_may = weather_resampled[(weather_resampled['month'] >= 1) & (weather_resampled['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = weather_resampled[(weather_resampled['month'] >= 9) | (weather_resampled['month'] <= 1)]

# Concatenate the two dataframes
weather_resampled_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
weather_resampled_filt = weather_resampled_filt.reset_index(drop=True)

all_data_weather_hour = pd.concat([hour_energy_columns_filt, modela_resampled_hour_filt, modelb_resampled_hour_filt, modelc_resampled_hour_filt, modeld_resampled_hour_filt, weather_resampled_filt], axis = 1)

#%%

'DAY'

# resample the hourly data to daily values
weather_resampled_day = weather_resampled.resample('1D', on='Time', label='right', closed='right').mean()  # DateTime is set as index in result df
weather_resampled_day.insert(0, "Time", weather_resampled_day.index)  # Insert back DateTime as column on first position
weather_resampled_day.index=range(len(weather_resampled_day))  # Reset index column from 0 to len df

weather_resampled_day['month'] = weather_resampled_day['Time'].dt.month

# Filter for data between January and May
df_jan_may = weather_resampled_day[(weather_resampled_day['month'] >= 1) & (weather_resampled_day['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = weather_resampled_day[(weather_resampled_day['month'] >= 9) | (weather_resampled_day['month'] <= 1)]

# Concatenate the two dataframes
weather_resampled_day_filt = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
weather_resampled_day_filt = weather_resampled_day_filt.reset_index(drop=True)

all_data_weather_day = pd.concat([day_energy_columns_filt, modela_resampled_day_filt, modelb_resampled_day_filt, modelc_resampled_day_filt, modeld_resampled_day_filt, weather_resampled_day_filt], axis = 1)

#%%

"weekly figures hourly"

# get the index and value of the highest value in the DataFrame
max_value_idx = all_data_weather_hour['sum'].values.argmax()
max_value = all_data_weather_hour['sum'].values.max()

print(f"Highest value: {max_value}, index: {max_value_idx}")

# Define the date range
start_date = '2022-12-12 00:00:00'
end_date = '2022-12-19 00:00:00'

# Sort each sensor
all_data_weather_hour_week = all_data_weather_hour[(all_data_weather_hour['DateTime'] >= start_date) & (all_data_weather_hour['DateTime'] <= end_date)]
all_data_weather_hour_week.reset_index(inplace=True, drop=True)

all_data_weather_hour_week = all_data_weather_hour_week.set_index('DateTime')

#%%

# create a figure with three subplots, sharing the x-axis
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(9,5))

#fix color on reference, model a and c

# plot the reference data on the top subplot
ax1.plot(all_data_weather_hour_week.index, all_data_weather_hour_week['sum'], label='Reference')
ax1.set_ylabel('kWh/h')
ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
ax1.set_title('Week 50 2022, hourly resolution data')

# plot the model data on the middle subplot
#colors = ['orange', 'darkviolet', 'magenta', 'deepskyblue']
#ax2.plot(all_data_weather_hour_week.index, all_data_weather_hour_week[['energy_use_a', 'energy_use_c']],  label=['Model A', 'Model C'])
ax2.plot(all_data_weather_hour_week.index, all_data_weather_hour_week['energy_use_a'], color='green', label='Model A')
ax2.plot(all_data_weather_hour_week.index, all_data_weather_hour_week['energy_use_c'], color='orange', label='Model C')
ax2.set_ylabel('kWh/h')
ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

# plot the radiation data on the bottom subplot
ax3.plot(all_data_weather_hour_week.index, all_data_weather_hour_week['mean_temp'], color='blue', label='Outdoor air temperature')
ax3.set_ylabel('°C')
ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

# plot the radiation data on the bottom subplot
ax4.plot(all_data_weather_hour_week.index, all_data_weather_hour_week['mean_radiation'], color='red', label='Global solar radiation')
ax4.set_ylabel('W/m²')
ax4.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

ax4.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))
ax4.xaxis.set_major_locator(mdates.HourLocator(interval=12))
ax4.set_xlim(pd.to_datetime('2022-12-12 00:00:00'), all_data_weather_hour_week.index[-1])
ax4.tick_params(axis='x', rotation=80)
fig.tight_layout()

# add gridlines to all subplots
ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()

# display the figure
plt.show()

#%%

"weekly figures daily"

# get the index and value of the highest value in the DataFrame
max_value_idx = all_data_weather_day['sum'].values.argmax()
max_value = all_data_weather_day['sum'].values.max()

print(f"Highest value: {max_value}, index: {max_value_idx}")

# Define the date range
start_date = '2022-12-12 00:00:00'
end_date = '2022-12-19 00:00:00'

# Sort each sensor
all_data_weather_day_week = all_data_weather_day[(all_data_weather_day['DateTime'] >= start_date) & (all_data_weather_day['DateTime'] <= end_date)]
all_data_weather_day_week.reset_index(inplace=True, drop=True)

all_data_weather_day_week = all_data_weather_day_week.set_index('DateTime')

#%%

# create a figure with three subplots, sharing the x-axis
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(9,5))

# plot the reference data on the top subplot
ax1.plot(all_data_weather_day_week.index, all_data_weather_day_week['sum'], color='green', label='Reference')
ax1.set_ylabel('kWh/h')
ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
ax1.set_title('Week 50 2022, day resolution data')

# plot the model data on the middle subplot
#colors = ['orange', 'darkviolet', 'magenta', 'deepskyblue']
ax2.plot(all_data_weather_day_week.index, all_data_weather_day_week[['energy_use_b', 'energy_use_d']], label=['Model B', 'Model D'])
ax2.set_ylabel('kWh/h')
ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

# plot the radiation data on the bottom subplot
ax3.plot(all_data_weather_day_week.index, all_data_weather_day_week['mean_temp'], color='blue', label='Outdoor air temperature')
ax3.set_ylabel('°C')
ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

# plot the radiation data on the bottom subplot
ax4.plot(all_data_weather_day_week.index, all_data_weather_day_week['mean_radiation'], color='red', label='Global solar radiation')
ax4.set_ylabel('W/m²')
ax4.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

ax4.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))
ax4.xaxis.set_major_locator(mdates.HourLocator(interval=24))
ax4.set_xlim(pd.to_datetime('2022-12-12 00:00:00'), all_data_weather_day_week.index[-1])
ax4.tick_params(axis='x', rotation=80)
fig.tight_layout()

# add gridlines to all subplots
ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()

# display the figure
plt.show()

#%%

"define pred"

pred = 'your location'

#%%
###############################################################################
###                             Calcualte CVRMSE, R2 and NMBE               ###
###############################################################################

# Convert actual and pred to DataFrames
actual = actual.to_frame()
pred = pred.to_frame()

'''Initialize the empty lists'''
list_cases = pd.Series.tolist(actual.columns) # Get list of names of columns of "actual" df and use it as names for output df

# Define function to calculate metrics
def calculate_metrics(actual, pred):
    list_cases = actual.columns.tolist() # Get list of names of columns of "actual" df and use it as names for output df
    list_CVRMSE =  [] # Empty list for the CVRMSE values
    list_R2 = [] # Empty list for the R2 values
    list_NMBE = [] # Empty list for the NMBE values
    
    for i_col, col_name in enumerate(list_cases):
        vect_measurements = actual.iloc[:,i_col].tolist() # Take i_th column with all rows and convert to list
        vect_simulation = pred.iloc[:,i_col].tolist() # Take i_th column with all rows and convert to list
        
        # Calculate metrics
        CVRMSE = sqrt(mean_squared_error(vect_simulation, vect_measurements))/(np.array(vect_measurements).mean())*100
        R2 = r2_score(vect_measurements, vect_simulation)
        NMBE = ((np.array(vect_measurements)-np.array(vect_simulation)).sum())/(np.array(vect_measurements).mean()*len(vect_measurements))*100
        
        # Append to lists
        list_CVRMSE.append(CVRMSE)
        list_R2.append(R2)
        list_NMBE.append(NMBE)
        
        print('Metrics calculated for', col_name)
        
    # Create output dataframes
    data = {'Cases':list_cases, 'CVRMSE [%]':list_CVRMSE} 
    CVRMSE_df = pd.DataFrame(data) 
    
    data = {'Cases':list_cases, 'R2 [-]':list_R2}
    R2_df = pd.DataFrame(data) 
    
    data = {'Cases':list_cases, 'NMBE [%]':list_NMBE}
    NMBE_df = pd.DataFrame(data)
    
    return CVRMSE_df, R2_df, NMBE_df
    
# Call function and print results
CVRMSE_df, R2_df, NMBE_df = calculate_metrics(actual, pred)
print('CVRMSE:')
print(CVRMSE_df)
print('R2:')
print(R2_df)
print('NMBE:')
print(NMBE_df)

#%%
###############################################################################
###                             END SCRIPT                                  ###
###############################################################################