import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# ###################################################### MODEL ######################################################

class RecurrentNeuralNetwork(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            output_size: int,
            device
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
        self.act = nn.Identity()

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
            pred: torch.Tensor = model(X_batch.to(DEVICE))
            loss = criterion(pred, y_batch.to(DEVICE))
            total_loss += loss.item()
        model.train()
    return total_loss / len(val_loader)

def calculate_prediction(model: nn.Module, val_dataset: SequencedDataset) -> torch.Tensor:
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            preds.append(torch.sigmoid(model(X_batch.to(DEVICE)))) # manually apply sigmoid since loss function does it now, loss isnt calculated here

    model.train()

    return (torch.cat(preds) >= 0.5).float()

# acc, recall, tpr
def calculate_results(y_pred: torch.Tensor, y: torch.Tensor):
    print(f'calculating accuracy, y_pred shape: {y_pred.size()}, y size: {y.size()}')
    tp = ((y_pred==1) & (y==1)).sum().item()
    tn = ((y_pred==0) & (y==0)).sum().item()
    fp = ((y_pred==1) & (y==0)).sum().item()
    fn = ((y_pred==0) & (y==1)).sum().item()
    print(f'Counts: tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}')
    total = y_pred.numel()
    return (
        (tp+tn)/total if total else 0,
        tp/(tp+fn) if (tp+fn) else 0,
        tp/(tp+fp) if (tp+fp) else 0
    )

def print_evaluation_metrics(model:nn.Module, val:SequencedDataset):
    zeroed_y = torch.zeros(val.y.size()).to(DEVICE)
    zeroed_acc, zeroed_recall, zeroed_prec = calculate_results(zeroed_y[val.seq_len:], val.y[val.seq_len:])
    print(f'Baseline: Accuracy on all zero predictions: {zeroed_acc*100:.8f}%')
    print(f'Baseline: Recall on all zero predictions: {zeroed_recall*100:.8f}%')
    print(f'Baseline: Precision on all zero predictions: {zeroed_prec*100:.8f}%')
    print(f'Baseline: F1 score on all one predictions: {2 * (zeroed_prec * zeroed_recall) / (zeroed_prec + zeroed_recall + 1e-4) :.8f}')

    print()

    ones_y = torch.ones(val.y.size()).to(DEVICE)
    ones_acc, ones_recall, ones_prec = calculate_results(ones_y[val.seq_len:], val.y[val.seq_len:])
    print(f'Baseline: Accuracy on all one predictions: {ones_acc*100:.8f}%')
    print(f'Baseline: Recall on all one predictions: {ones_recall*100:.8f}%')
    print(f'Baseline: Precision on all one predictions: {ones_prec*100:.8f}%')
    print(f'Baseline: F1 score on all one predictions: {2 * (ones_prec*ones_recall)/(ones_prec+ones_recall + 1e-4) :.8f}')

    print()

    flattened = [tuple(t.flatten().tolist()) for t in val.y]
    from collections import Counter
    most_common_flat = Counter(flattened).most_common(1)[0][0]
    most_common_tensor = torch.tensor(most_common_flat).reshape(val.y[0].shape)
    mcv = [most_common_tensor.clone() for _ in range(len(val.y))]
    mcv = torch.stack(mcv).to(DEVICE)
    mcv_acc, mcv_recall, mcv_prec = calculate_results(mcv[val.seq_len:], val.y[val.seq_len:])
    print(mcv[0])
    print(f'Baseline: Accuracy on most common vote predictions: {mcv_acc * 100:.8f}%')
    print(f'Baseline: Recall on most common vote predictions: {mcv_recall * 100:.8f}%')
    print(f'Baseline: Precision on most common vote predictions: {mcv_prec * 100:.8f}%')
    print(f'Baseline: F1 score on most common vote predictions: {2 * (mcv_prec * mcv_recall) / (mcv_prec + mcv_recall + 1e-4) :.8f}')

    print()

    model_acc, model_recall,model_prec = calculate_results(calculate_prediction(model, val), val.y[val.seq_len:])
    print(f'Model accuracy: {model_acc*100:.8f}%')
    print(f'Model recall: {model_recall*100:.8f}%')
    print(f'Model precision: {model_prec*100:.8f}%')
    print(f'Model F1 score: {2 * (model_prec*model_recall)/(model_prec+model_recall + 1e-4) :.8f} ')
    


# ###################################################### MAIN ######################################################

# Hyperparameters
## Train
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
SEQUENCE_LENGTH = 300
NUM_EPOCHS = 9001
HIDDEN_SIZE = 256
NUM_LAYERS = 3
## Regularize
DROPOUT_RATE = 0.3
L2_LAMBDA = 1e-03

## Early Stop
PATIENCE = 20
THRESHOLD = 0.01

# Configuration
EVALUATION_MODE = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
### Used as a source path in evaluation mode and as a destination path in training mode, should be used to manage multiple models if needed
MODEL_PATH = 'model/rnn_model.pt'

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
    df = pd.read_csv('processed_data/final.csv')
    scalar = StandardScaler()
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df[TARGET_COLUMNS].to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train = scalar.fit_transform(X_train)
    X_val = scalar.transform(X_val)

    X_train = torch.Tensor(X_train).to(DEVICE)
    X_val = torch.Tensor(X_val).to(DEVICE)
    y_train = torch.Tensor(y_train).to(DEVICE)
    y_val = torch.Tensor(y_val).to(DEVICE)

    train_dataset, val_dataset = SequencedDataset(X_train, y_train, SEQUENCE_LENGTH), SequencedDataset(X_val, y_val, SEQUENCE_LENGTH)

    print(f"Training Data:\n\tX:\n\t{train_dataset.X}\n\ty:\n\t{train_dataset.y}")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print('instantiating model...')
    
    model = RecurrentNeuralNetwork(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=len(TARGET_COLUMNS),
        device=DEVICE
    ).to(DEVICE)
    if EVALUATION_MODE:
        model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device(DEVICE)))
        model = model.to(DEVICE)
        print_evaluation_metrics(model, val_dataset)
        quit(0)

    model.train()
    pos_weight = (y_train.size(0)) / (y_train.sum(dim=0) + 1e-6) # prefers positive predictions, should be inversely proportional to how rare the class is
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(pos_weight)).to(DEVICE) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_LAMBDA)
    
    print('starting training...')
    print(f'Max number of iterations: {len(train_loader) * NUM_EPOCHS}')
    iteration = 0
    best_loss = float('inf')
    best_model_state = None
    num_no_improve = 0
    for epoch in range(NUM_EPOCHS):
        for X_batch, y_batch in train_loader:
            inputs = X_batch.to(DEVICE)
            targets = y_batch.to(DEVICE)

            optimizer.zero_grad()
            preds = model(inputs)

            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            iteration += 1

        current_loss = calculate_validation_loss(model, val_dataset, criterion)
        if current_loss - best_loss >= THRESHOLD:
            num_no_improve += 1
            if num_no_improve >= PATIENCE:
                break
        else:
            num_no_improve = 0
            if best_loss > current_loss:
                best_loss = current_loss
                best_model_state = copy.deepcopy(model.state_dict())
        print(f"Iteration {iteration}, Loss: {current_loss}, No Improvement Count: {num_no_improve}")
            

    print(f'training complete {'due to early stop' if num_no_improve >= PATIENCE else ''}')

    torch.save(best_model_state, MODEL_PATH)
