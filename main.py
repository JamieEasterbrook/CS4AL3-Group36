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
    with torch.no_grad():
        pred = model(X_val.unsqueeze(0))
        pred_reg, pred_class = pred[:REG_CUTOFF_IDX_INCLUSIVE+1], pred[REG_CUTOFF_IDX_INCLUSIVE+1:]
        targets_reg, targets_class = y_val[:REG_CUTOFF_IDX_INCLUSIVE+1], y_val[REG_CUTOFF_IDX_INCLUSIVE+1:]
        loss = criterion_reg(pred_reg, targets_reg) + criterion_class(pred_class, targets_class)
        model.train()
    return loss.item()

def load_data()->tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_columns = [
        'Cloudiness',
        'Precipitation',
        'Intensity'
        'Heat'
        'Visibility'
        'Wind'
        'Pollution'
        'Thunderstorms'
        'Smoke'
    ]

    feature_columns = [
        'Temp (°C)',
        'Dew Point Temp (°C)',
        'Rel Hum (%)',
        'Wind Dir (10s deg)',
        'Wind Spd (km/h)',
        'Visibility (km)',
        'Stn Press (kPa)',
        'Wind Chill',
        'dtHours',
        'sinTime',
        'sinDay',
        'year'
    ]
    df = pd.read_csv('processed_data/final.csv')
    X = torch.Tensor(df[feature_columns].to_numpy())
    y = torch.Tensor(df[target_columns].to_numpy())
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_val, y_train, y_val


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
    X_train, X_val, y_train, y_val = load_data()
    
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


