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
    merged_df['dtHours'] = merged_df['DateTime'].diff()
    # TODO: Finish implementing

    print('Extracting relevant data for features and targets')
    x = merged_df.drop(columns=['Weather', 'Year', 'Month', 'Day', 'Time (LST)', 'DateTime']).to_numpy(dtype=np.float32)
    y = merged_df['Weather']
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
    
    X_train, X_val, y_train, y_val = train_test_split(torch.tensor(x), torch.tensor(y), train_size=0.8, shuffle=False)
    return X_train, X_val, y_train, y_val


# ###################################################### MODEL ######################################################

class RecurrentNeuralNetwork(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            output_size: int,
            reg_cutoff_idx_inclusive: int,
            device: str
        ):
        super(RecurrentNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.reg_cutoff_idx_inclusive = reg_cutoff_idx_inclusive
        self.device = device

        num_reg = reg_cutoff_idx_inclusive + 1
        num_class = output_size - num_reg

        self.rnn = nn.RNN(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=DROPOUT_RATE
        )
        
        self.fc_reg = nn.Linear(
            self.hidden_size,
            num_reg
        ) if (num_reg > 0) else nn.Identity()

        self.fc_class = nn.Linear(
            self.hidden_size,
            num_class
        ) if (num_class > 0) else nn.Identity()

        self.act_class = nn.Sigmoid()
        

    def forward(self, x: torch.Tensor):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.rnn(x, h0)
        h_n = out[:, -1, :]
        out1 = self.fc_reg(h_n)

        out2 = self.act_class(self.fc_class(h_n))

        return out1, out2

# ################################################## MAIN HELPERS ###################################################

def calculate_validation_loss(model: nn.Module, X_val: torch.Tensor, y_val: torch.Tensor, criterion_reg, criterion_class):
    model.eval()
    pred = model(X_val.unsqueeze(0))
    pred_reg, pred_class = pred[:REG_CUTOFF_IDX_INCLUSIVE+1], pred[REG_CUTOFF_IDX_INCLUSIVE+1:]
    targets_reg, targets_class = y_val[:REG_CUTOFF_IDX_INCLUSIVE+1], y_val[REG_CUTOFF_IDX_INCLUSIVE+1:]
    loss = criterion_reg(pred_reg, targets_reg) + criterion_class(pred_class, targets_class)

# ###################################################### MAIN ######################################################

# Hyperparameters
LEARNING_RATE = 0.005
DROPOUT_RATE = 0.0
BATCH_SIZE = 16
SEQUENCE_LENGTH = 30
NUM_EPOCHS = 100
CHECK_EVERY = 10

# Configuration
REG_CUTOFF_IDX_INCLUSIVE = 6


if __name__ == '__main__':
    X_train, X_val, y_train, y_val = load_data_and_preprocess('weather', '*.csv')
    
    model = RecurrentNeuralNetwork(
        input_size=12,
        hidden_size=42,
        num_layers=2,
        output_size=9,
        reg_cutoff_idx_inclusive=REG_CUTOFF_IDX_INCLUSIVE,
        device=torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
    )

    model.train()
    criterion_reg = nn.MSELoss()
    criterion_class = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        for i in range(SEQUENCE_LENGTH, X_train.size(0), 1): # slide by 1 with a window of SEQUENCE_LENGTH elements
            inputs = X_train[i-SEQUENCE_LENGTH:i].unsqueeze(0)
            targets = y_train[i].unsqueeze(0)

            optimizer.zero_grad()
            pred_reg, pred_class = model(inputs)

            targets_reg = targets[:, :REG_CUTOFF_IDX_INCLUSIVE+1]
            targets_class = targets[:, REG_CUTOFF_IDX_INCLUSIVE+1:]

            loss_reg = criterion_reg(pred_reg, targets_reg)
            loss_class = criterion_class(pred_class, targets_class)

            loss = loss_reg + loss_class
            loss.backward()
            optimizer.step()

        if (epoch+1) % CHECK_EVERY == 0:
            print(f"Epoch {epoch+1}, Loss: {calculate_validation_loss(model, X_val, y_val, criterion_reg, criterion_class)}")


