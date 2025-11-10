import glob
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# ###################################################### HELPERS ######################################################
def days_in_month(year:int, month:int):
    days_reg = [
    -1,
    31,
    28,
    31,
    30,
    31,
    30,
    31,
    31,
    30,
    31,
    30,
    31,
    ]
    if year % 4 != 0 & month != 2:
        return days_reg[month]
    else:
        return 29

def get_next_point(year:int, month:int, day:int, hour:int) -> tuple[int,int,int,int]:
    if hour <= 19:
        return (year,month,day,hour+3)
    else:
        if day < days_in_month(year,month):
            return (year, month, day+1, (hour+3) % 24)
        else:
            if month < 12:
                return (year, month+1, 1, (hour+3) % 24)
            else:
                return (year+1, 1, 1, (hour+3) % 24)

# returns X_train, X_val, y_train, y_val
def load_data_and_preprocess(path:str):
    print('Reading in raw csv files')
    paths = glob.glob("*.csv", root_dir=path, recursive=True)
    merged_df = pd.read_csv(os.path.join('weather/', paths[0]))
    for p in paths[1:]:
        merged_df = pd.concat([merged_df, pd.read_csv(os.path.join('weather/', p))], ignore_index=True)

    print('Removing invalid data')
    merged_df = merged_df.sort_values(by=['Year', 'Month', 'Day', 'Time (LST)'], ascending=True)
    merged_df = merged_df.drop(
        labels=[
            'Longitude (x)',
            'Latitude (y)',
            'Station Name',
            'Climate ID',
            'Date/Time (LST)',
            'Flag',
            'Temp Flag',
            'Dew Point Temp Flag',
            'Rel Hum Flag',
            'Wind Dir Flag',
            'Wind Spd Flag',
            'Visibility Flag',
            'Hmdx Flag',
            'Wind Chill Flag',
            'Precip. Amount Flag',
            'Hmdx',
            'Stn Press Flag',
        ],
        axis=1
    )
    merged_df = merged_df.dropna(axis=1, how='all')
    merged_df = merged_df.dropna(axis=0)

    print('Transforming features')
    # dtHours: Delta time in hours since the last measurement, 0 for the first one
    # sinTime: Local time transformed into a sinusoidal function (p=24)
    # sinDay: Month and Day transformed into days 0-365 then into a sinusoidal function (p=365)
    # year: Keep as it is
    
    merged_df['DateTime'] = pd.to_datetime(
        merged_df['Year'].astype(str) + '-' +
        merged_df['Month'].astype(str) + '-' +
        merged_df['Day'].astype(str) + ' ' +
        merged_df['Time (LST)'].astype(str),
        errors='coerce'
    )
    merged_df['dtHours'] = merged_df['DateTime'].diff().dt.total_seconds().div(3600).fillna(0)
    hours = merged_df['DateTime'].dt.hour + merged_df['DateTime'].dt.minute / 60.0
    merged_df['sinTime'] = np.sin(2 * np.pi * hours / 24)

    days_of_year = merged_df['DateTime'].dt.dayofyear
    merged_df['sinDay'] = np.sin(2 * np.pi * days_of_year / 365)
    merged_df['year'] = merged_df['Year']

    print('Extracting relevant data for features and targets')
    labels = merged_df['Weather'].apply(lambda x: [l.strip() for l in str(x).split(',')])
    label_list:list[str] = json.load(open('docs/labels.json', 'r'))['separated']
    label_list.sort()
    for label in label_list:
        merged_df[label] = labels.apply(lambda wordList: int(label in wordList))


    # TODO: just use a vector of 1s and 0s for each label, forget about this complicated encoding
    
    # Encode weather labels:
        # Output Classes (see docs/labels.md):
            # Cloudiness        (Real, [0,1])
            # Precipitation     (Real, [0,1])
            # Intensity         (Real, [0,1])
            # Heat              (Real, [0,1])
            # Visibility        (Real, [0,1])
            # Wind              (Real, [0,1])
            # Pollution         (Real, [0,1])
            # Thunderstorms     (Integer, [0,1])
            # Smoke             (Integer, [0,1])
    
    
    merged_df.to_csv('processed_data/final.csv', index=False)

if __name__ == '__main__':
    load_data_and_preprocess('weather')