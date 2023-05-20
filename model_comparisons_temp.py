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

"REFERENCE"

# load data and convert object to datetime
actual = 'alldata_5min_temp.csv'
df = pd.read_csv(actual)
df['DateTime'] = pd.to_datetime(df['DateTime'], format = '%d-%m-%y %H:%M')
print(df.dtypes)

df['month'] = df['DateTime'].dt.month

# Filter for data between January and May
df_jan_may = df[(df['month'] >= 1) & (df['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = df[(df['month'] >= 9) | (df['month'] <= 1)]

# Concatenate the two dataframes
df_concat = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
df_concat = df_concat.reset_index(drop=True)

'60-min resample'
df_resampled_hour = df.resample('60T', on='DateTime', label='right', closed='right').mean()  # 2T for resampling every 2 minutes, DateTime is set as index in result df
df_resampled_hour.insert(0, "DateTime", df_resampled_hour.index)  # Insert back DateTime as column on first position
df_resampled_hour.index=range(len(df_resampled_hour))  # Reset index column from 0 to len df

df_resampled_hour['month'] = df_resampled_hour['DateTime'].dt.month

# Filter for data between January and May
df_jan_may = df_resampled_hour[(df_resampled_hour['month'] >= 1) & (df_resampled_hour['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = df_resampled_hour[(df_resampled_hour['month'] >= 9) | (df_resampled_hour['month'] <= 1)]

# Concatenate the two dataframes
df_resampled_hour_concat = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
df_resampled_hour_concat = df_resampled_hour_concat.reset_index(drop=True)

#%%

"MODEL A"

# load data and convert object to datetime
modela = 'modela_temp.csv'
df_modela = pd.read_csv(modela)
df_modela['DateTimea'] = pd.to_datetime(df_modela['DateTime'], format = '%d-%m-%y %H:%M')
print(df_modela.dtypes)
del df_modela['DateTime']

'5-min resample'
df_modela_resampled = df_modela.resample('5T', on='DateTimea', label='right', closed='right').mean()  # 2T for resampling every 2 minutes, DateTime is set as index in result df
df_modela_resampled.insert(0, "DateTimea", df_modela_resampled.index)  # Insert back DateTime as column on first position
df_modela_resampled.index=range(len(df_modela_resampled))  # Reset index column from 0 to len df

df_modela_resampled['month'] = df_modela_resampled['DateTimea'].dt.month

# Filter for data between January and May
df_jan_may = df_modela_resampled[(df_modela_resampled['month'] >= 1) & (df_modela_resampled['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = df_modela_resampled[(df_modela_resampled['month'] >= 9) | (df_modela_resampled['month'] <= 1)]

# Concatenate the two dataframes
df_modela_resampled_concat = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
df_modela_resampled_concat = df_modela_resampled_concat.reset_index(drop=True)

'60-min resample'
df_modela_resampled_hour = df_modela.resample('60T', on='DateTimea', label='right', closed='right').mean()  # 2T for resampling every 2 minutes, DateTime is set as index in result df
df_modela_resampled_hour.insert(0, "DateTimea", df_modela_resampled_hour.index)  # Insert back DateTime as column on first position
df_modela_resampled_hour.index=range(len(df_modela_resampled_hour))  # Reset index column from 0 to len df

df_modela_resampled_hour['month'] = df_modela_resampled_hour['DateTimea'].dt.month

# Filter for data between January and May
df_jan_may = df_modela_resampled_hour[(df_modela_resampled_hour['month'] >= 1) & (df_modela_resampled_hour['month'] <= 5)]

# Filter for data between September and January
df_sept_jan = df_modela_resampled_hour[(df_modela_resampled_hour['month'] >= 9) | (df_modela_resampled_hour['month'] <= 1)]

# Concatenate the two dataframes
df_modela_resampled_concat_hour = pd.concat([df_jan_may, df_sept_jan])

# Reset the index of the concatenated dataframe
df_modela_resampled_concat_hour = df_modela_resampled_concat_hour.reset_index(drop=True)

#%%

"MODEL B"

# load data and convert object to datetime
modelb = 'modelb_temp.csv'
df_modelb = pd.read_csv(modelb)
df_modelb['DateTime'] = pd.to_datetime(df_modelb['DateTime'], format = '%d-%m-%y %H:%M')
print(df_modelb.dtypes)

'5-min resample'
df_modelb_resampled = df_modelb.resample('5T', on='DateTime', label='right', closed='right').mean()  # 2T for resampling every 2 minutes, DateTime is set as index in result df
df_modelb_resampled.insert(0, "DateTime", df_modelb_resampled.index)  # Insert back DateTime as column on first position
df_modelb_resampled.index=range(len(df_modelb_resampled))  # Reset index column from 0 to len df

'60-min resample'
df_modelb_resampled_hour = df_modelb.resample('1H', on='DateTime', label='right', closed='right').mean()  # 2T for resampling every 2 minutes, DateTime is set as index in result df
df_modelb_resampled_hour.insert(0, "DateTime", df_modelb_resampled_hour.index)  # Insert back DateTime as column on first position
df_modelb_resampled_hour.index=range(len(df_modelb_resampled_hour))  # Reset index column from 0 to len df

#%%

"MODEL C"

# load data and convert object to datetime
modelc = 'modelc_temp.csv'
df_modelc = pd.read_csv(modelc)
df_modelc['DateTime'] = pd.to_datetime(df_modelc['DateTime'], format = '%d-%m-%y %H:%M')
print(df_modelc.dtypes)

'5-min resample'
df_modelc_resampled = df_modelc.resample('5T', on='DateTime', label='right', closed='right').mean()  # 2T for resampling every 2 minutes, DateTime is set as index in result df
df_modelc_resampled.insert(0, "DateTime", df_modelc_resampled.index)  # Insert back DateTime as column on first position
df_modelc_resampled.index=range(len(df_modelc_resampled))  # Reset index column from 0 to len df

'60-min resample'
df_modelc_resampled_hour = df_modelc.resample('1H', on='DateTime', label='right', closed='right').mean()  # 2T for resampling every 2 minutes, DateTime is set as index in result df
df_modelc_resampled_hour.insert(0, "DateTime", df_modelc_resampled_hour.index)  # Insert back DateTime as column on first position
df_modelc_resampled_hour.index=range(len(df_modelc_resampled_hour))  # Reset index column from 0 to len df

#%%

"MODEL D"

# load data and convert object to datetime
modeld = 'modeld_temp.csv'
df_modeld = pd.read_csv(modeld)
df_modeld['DateTime'] = pd.to_datetime(df_modeld['DateTime'], format = '%d-%m-%y %H:%M')
print(df_modeld.dtypes)

'5-min resample'
df_modeld_resampled = df_modeld.resample('5T', on='DateTime', label='right', closed='right').mean()  # 2T for resampling every 2 minutes, DateTime is set as index in result df
df_modeld_resampled.insert(0, "DateTime", df_modeld_resampled.index)  # Insert back DateTime as column on first position
df_modeld_resampled.index=range(len(df_modeld_resampled))  # Reset index column from 0 to len df

'60-min resample'
df_modeld_resampled_hour = df_modeld.resample('1H', on='DateTime', label='right', closed='right').mean()  # 2T for resampling every 2 minutes, DateTime is set as index in result df
df_modeld_resampled_hour.insert(0, "DateTime", df_modeld_resampled_hour.index)  # Insert back DateTime as column on first position
df_modeld_resampled_hour.index=range(len(df_modeld_resampled_hour))  # Reset index column from 0 to len df

#%%
###############################################################################
###                                 FIGURE                                  ###
###############################################################################

"5-min"

ref_modela_concat = pd.concat([df_concat, df_modela_resampled_concat], axis = 1)
ref_modela_concat = ref_modela_concat.reset_index(drop=True)

df_load_duration_5min = ref_modela_concat[['stth_avg', 'stth_avg_a', 'sttv_avg', 'sttv_avg_a', '1tv_avg', '1tv_avg_a', '1th_avg', '1th_avg_a', '2tv_avg', '2tv_avg_a', '2th_avg', '2th_avg_a']]

# set the colors for each line
colors = ['r', 'r', 'g', 'g', 'm', 'm', 'k', 'k', 'b', 'b', 'y', 'y']

# set the columns for which to create load duration curves
columns = ['stth_avg', 'stth_avg_a', 'sttv_avg', 'sttv_avg_a', '1tv_avg', '1tv_avg_a', '1th_avg', '1th_avg_a', '2tv_avg', '2tv_avg_a', '2th_avg', '2th_avg_a']

# set the line styles for each line
line_styles = ['-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--']

# sort the DataFrame by the first column in descending order
df_sorted = ref_modela_concat[columns].sort_values(columns[0], ascending=False)

# create a new column 'Rank' with the load duration rank
df_sorted['Rank'] = range(1, len(df_sorted) + 1)

# plot the load duration curve for the first column with the first color and line style in the list
plt.plot(df_sorted['Rank'], df_sorted[columns[0]], color=colors[0], linestyle=line_styles[0], label=columns[0])

for i, col in enumerate(columns[1:]):
    df_sorted = df_sorted.sort_values(col, ascending=False)
    df_sorted['Rank'] = range(1, len(df_sorted) + 1)
    plt.plot(df_sorted['Rank'], df_sorted[col], color=colors[i+1], linestyle=line_styles[i+1], label=col)

# set the plot labels and legend
plt.xlabel('Rank [Length of 5-min. resolution data]')
plt.ylabel('Average indoor air temperature [°C]')
plt.title('Load Duration Curve 5-min. resolution')
#plt.legend(['GF right ref.', 'GF right model', 'GF left ref.', 'GF left model', '1st left ref', '1st left model', '1st right ref.', '1st right model', '2nd left ref.', '2nd left model', '2nd right ref.', '2nd right model'], loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()

# display the plot
plt.show()

#%%

"60-min"

ref_modela_concat_hour = pd.concat([df_resampled_hour_concat, df_modela_resampled_concat_hour], axis = 1)
ref_modela_concat_hour = ref_modela_concat_hour.reset_index(drop=True)

df_load_duration_60min = ref_modela_concat_hour[['stth_avg', 'stth_avg_a', 'sttv_avg', 'sttv_avg_a', '1tv_avg', '1tv_avg_a', '1th_avg', '1th_avg_a', '2tv_avg', '2tv_avg_a', '2th_avg', '2th_avg_a']]

# set the colors for each line
colors = ['r', 'r', 'g', 'g', 'm', 'm', 'k', 'k', 'b', 'b', 'y', 'y']

# set the columns for which to create load duration curves
columns = ['stth_avg', 'stth_avg_a', 'sttv_avg', 'sttv_avg_a', '1tv_avg', '1tv_avg_a', '1th_avg', '1th_avg_a', '2tv_avg', '2tv_avg_a', '2th_avg', '2th_avg_a']

# set the line styles for each line
line_styles = ['-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--']

# sort the DataFrame by the first column in descending order
df_sorted = ref_modela_concat_hour[columns].sort_values(columns[0], ascending=False)

# create a new column 'Rank' with the load duration rank
df_sorted['Rank'] = range(1, len(df_sorted) + 1)

# plot the load duration curve for the first column with the first color and line style in the list
plt.plot(df_sorted['Rank'], df_sorted[columns[0]], color=colors[0], linestyle=line_styles[0], label=columns[0])

for i, col in enumerate(columns[1:]):
    df_sorted = df_sorted.sort_values(col, ascending=False)
    df_sorted['Rank'] = range(1, len(df_sorted) + 1)
    plt.plot(df_sorted['Rank'], df_sorted[col], color=colors[i+1], linestyle=line_styles[i+1], label=col)

# set the plot labels and legend
plt.xlabel('Rank [Lenght of hourly resolution data]')
plt.ylabel('Average indoor air temperature [°C]')
plt.title('Load Duration Curve hourly resolution')
plt.legend(['GF right ref.', 'GF right model', 'GF left ref.', 'GF left model', '1st left ref', '1st left model', '1st right ref.', '1st right model', '2nd left ref.', '2nd left model', '2nd right ref.', '2nd right model'], loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()

# display the plot
plt.show()

#%%
###############################################################################
###                             END SCRIPT                                  ###
###############################################################################