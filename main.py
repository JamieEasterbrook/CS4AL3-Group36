import glob
import os
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
def load_data_and_preprocess(path:str, match:str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    print('Reading in pre-merged data')
    merged_df = pd.read_csv('data/merged_raw.csv')

    print('Removing invalid data')
    merged_df = merged_df.sort_values(by=['Year', 'Month', 'Day', 'Time (LST)'], ascending=True)
    merged_df = merged_df.dropna(
        subset=[
            'Weather',
            'Temp (°C)',
            'Rel Hum (%)',
            'Stn Press (kPa)',
            'Visibility (km)',
            'Wind Spd (km/h)',
            'Dew Point Temp (°C)',
            'Wind Dir (10s deg)'
        ]
    )
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

    print('Transforming features')
    # TODO: Transform Year, Month, Date, Time, into features:
        # dtHours: Delta time in hours since the last measurement, 0 for the first one
        # sinTime: Local time transformed into a sinusoidal function (p=24)
        # sinDay: Month and Day transformed into days 0-365 then into a sinusoidal function (p=365)
        # year: Keep as it is
    
    merged_df['DateTime'] = pd.to_datetime(merged_df[['Year', 'Month', 'Day', 'Time (LST)']].agg('-'.join, axis=1))
    # TODO: Finish implementing

    print('Extracting relevant data for features and targets')
    x = merged_df.drop(columns=['Weather', 'Year', 'Month', 'Day', 'Time (LST)', 'DateTime']).to_numpy(dtype=np.float32)
    y = merged_df['Weather']
    # Encode weather labels:
        # Output Classes (see docs/labels.md):
            # Cloudiness(Real, [0,1]): [0,0.05)
    
    X_train, X_val, y_train, y_val = train_test_split(torch.tensor(x), torch.tensor(y), train_size=0.8)
    return X_train, X_val, y_train, y_val


# ###################################################### MODEL ######################################################

class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.initial_phase = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Sigmoid() if output_size == 1 else nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor):
        x = self.initial_phase(self.flatten(x))
        
        return x

# ###################################################### MAIN ######################################################

if __name__ == '__main__':
    X_train, X_val, y_train, y_val = load_data_and_preprocess('weather', '*.csv')
