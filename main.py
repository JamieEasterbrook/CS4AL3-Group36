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
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.rnn(x, h0)
        h_n = out[:, -1, :]
        out = self.fc(h_n)
        return self.act(out)

class SequencedDataset(Dataset):
    def __init__(self, X:torch.Tensor, y:torch.Tensor, seq_len: int):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return self.X.size(0) - self.seq_len
    
    def __getitem__(self, idx):
        return self.X[idx:idx+self.seq_len], self.y[idx+self.seq_len]


# ################################################## MAIN HELPERS ###################################################

def calculate_validation_loss(model: nn.Module, val: SequencedDataset, criterion):
    model.eval()
    total_loss = 0.0
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            pred: torch.Tensor = model(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item()
        model.train()
    return total_loss / len(val_loader)

def calculate_accuracy(model: nn.Module, val: SequencedDataset) -> float:
    model.eval()
    correct = 0
    total = 0
    val_loader = DataLoader(val, batch_size=1, shuffle=False)
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            pred: torch.Tensor = model(X_batch)
            predicted = (pred >= 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.numel()
    return correct / total

def load_data()->tuple[SequencedDataset, SequencedDataset]:
    df = pd.read_csv('processed_data/final.csv')
    X = torch.Tensor(df[FEATURE_COLUMNS].to_numpy())
    y = torch.Tensor(df[TARGET_COLUMNS].to_numpy())
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    return SequencedDataset(X_train, y_train, SEQUENCE_LENGTH), SequencedDataset(X_val, y_val, SEQUENCE_LENGTH)


# ###################################################### MAIN ######################################################

# Hyperparameters
LEARNING_RATE = 0.005
DROPOUT_RATE = 0.0
BATCH_SIZE = 32
SEQUENCE_LENGTH = 30
NUM_EPOCHS = 1
HIDDEN_SIZE = 16
NUM_LAYERS = 2

# Evaluation
CHECK_EVERY = 1

# Configuration
EVALUATION_MODE = False

# Constants
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
    train_dataset, val_dataset = load_data()
    print(f"Training Data:\n\tX:\n\t{train_dataset.X}\n\ty:\n\t{train_dataset.y}")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print('instantiating model...')
    model = RecurrentNeuralNetwork(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=len(TARGET_COLUMNS),
        device=torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
    )
    if EVALUATION_MODE:
        model.load_state_dict(torch.load('model/rnn_model.pt'))
        accuracy = calculate_accuracy(model, val_dataset)
        print(f'Model accuracy on validation set: {accuracy*100:.2f}%')
        quit(0)

    model.train()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print('starting training...')
    print(f'Max number of iterations: {len(train_loader) * NUM_EPOCHS}')
    iteration = 0
    for epoch in range(NUM_EPOCHS):
        for X_batch, y_batch in train_loader:
                inputs = X_batch
                targets = y_batch

                optimizer.zero_grad()
                preds = model(inputs)

                loss = criterion(preds, targets)

                loss.backward()
                optimizer.step()

                if (iteration+1) % CHECK_EVERY == 0:
                    print(f"Iteration {iteration+1}, Loss: {calculate_validation_loss(model, val_dataset, criterion)}")
                iteration += 1

    print('training complete.')

    model_state = model.state_dict()
    torch.save(model_state, 'model/rnn_model.pt')