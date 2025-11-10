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
            device: str
        ):
        super(RecurrentNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.device = device

        self.rnn = nn.RNN(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=DROPOUT_RATE
        )

        self.fc = nn.Linear(
            self.hidden_size,
            self.output_size
        )
        self.act = []
        for i in range(self.output_size):
            self.act.append(nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.rnn(x, h0)
        h_n = out[:, -1, :]
        out = self.fc(h_n)
        predictions = []
        for i in range(self.output_size):
            predictions.append(self.act[i](out).item())
        return torch.Tensor(predictions)

# ################################################## MAIN HELPERS ###################################################

def calculate_validation_loss(model: nn.Module, X_val: torch.Tensor, y_val: torch.Tensor, criterion):
    model.eval()
    with torch.no_grad():
        pred: torch.Tensor = model(X_val.unsqueeze(0))
        loss = criterion(pred, y_val)
        model.train()
    return loss.item()

def load_data()->tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    df = pd.read_csv('processed_data/final.csv')
    X = torch.Tensor(df[FEATURE_COLUMNS].to_numpy())
    y = torch.Tensor(df[TARGET_COLUMNS].to_numpy())
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
TARGET_COLUMNS = [
        "Blowing Snow",
        "Clear",
        "Cloudy",
        "Drizzle",
        "Fog",
        "Freezing Drizzle",
        "Freezing Fog",
        "Freezing Rain",
        "Haze",
        "Heavy Rain",
        "Heavy Rain Showers",
        "Heavy Snow",
        "Ice Pellets",
        "Mainly Clear",
        "Moderate Hail",
        "Moderate Rain",
        "Moderate Rain Showers",
        "Moderate Snow",
        "Mostly Cloudy",
        "Rain",
        "Rain Showers",
        "Smoke",
        "Snow",
        "Snow Grains",
        "Snow Pellets",
        "Snow Showers",
        "Thunderstorms"
    ]

FEATURE_COLUMNS = [
        "Temp (°C)",
        "Dew Point Temp (°C)",
        "Rel Hum (%)",
        "Wind Dir (10s deg)",
        "Wind Spd (km/h)",
        "Visibility (km)",
        "Stn Press (kPa)",
        "Wind Chill",
        "dtHours",
        "sinTime",
        "sinDay",
        "year",
    ]

if __name__ == '__main__':
    print('loading data...')
    X_train, X_val, y_train, y_val = load_data()
    print(f"Training Data:\n\tX:\n\t\t{X_train}\n\ty:\n\t\t{y_train}")
    
    print('instantiating model...')
    model = RecurrentNeuralNetwork(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=42,
        num_layers=2,
        output_size=len(TARGET_COLUMNS),
        device=torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
    )

    model.train()
    criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print('starting training...')
    for epoch in range(NUM_EPOCHS):
        for i in range(SEQUENCE_LENGTH, X_train.size(0), 1): # slide by 1 with a window of SEQUENCE_LENGTH elements
            inputs = X_train[i-SEQUENCE_LENGTH:i].unsqueeze(0)
            targets = y_train[i].unsqueeze(0)

            optimizer.zero_grad()
            preds = model(inputs)

            loss = criterion(preds, targets)

            loss.backward()
            optimizer.step()

        if (epoch+1) % CHECK_EVERY == 0:
            print(f"Epoch {epoch+1}, Loss: {calculate_validation_loss(model, X_val, y_val, criterion)}")


