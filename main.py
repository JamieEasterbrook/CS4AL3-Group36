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

# transformer functions
from torch.optim.lr_scheduler import LambdaLR
import time

# ###################################################### MODEL ######################################################

# RNN model implementation
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



# Transformer model implementation
class Transformer(nn.Module):
    def __init__(
            self,
            input_dim,
            model_dim,
            num_heads,
            num_layers,
            output_dim,
            dropout=0.25    ):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = model_dim,
            nhead = num_heads,
            dim_feedforward = 4*model_dim,  # can be changed, but 4x was recommended
            dropout = dropout,
            batch_first=True,   # silences warning
            activation="relu")
        self.transformer = nn.TransformerEncoder(encoder_layer = encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


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

def calculate_prediction(model: nn.Module, val_dataset: SequencedDataset, threshold = 0.5) -> torch.Tensor:
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    labels = []
    with torch.no_grad():
        for X_batch,_ in val_loader:
            preds.append(torch.sigmoid(model(X_batch.to(DEVICE)))) # manually apply sigmoid since loss function does it now, loss isnt calculated here
    model.train()

    return (torch.cat(preds) >= threshold).float()

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


def multilabel_metrics(y_pred, y_true):
    acc = 0
    for pred_row, true_row in zip(y_pred, y_true):
        if torch.equal(pred_row, true_row):
            acc += 1
    return acc/len(y_pred)

def batch_decode(preds: torch.Tensor, columns: list[str]):
    decoded = []
    for row in preds:
        decode = []
        labels = [col for col, bit in zip(columns, row.tolist()) if bit == 1]
        for label in labels:
            c = SIMPLE_MAPPING[label]
            if c not in decode:
                decode.append(c)
        decoded.append(decode)
    return decoded


# allow for variable learning rate
def lr_lambda(current_step: int):
    total_steps = NUM_EPOCHS * TRAIN_LENGTH
    warmup_steps = int(total_steps * 0.1)
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(
        0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
    )

def simplify_data (y):
    new_y = np.zeros((len(y), len(SIMPLE_COLUMNS)))

    for i in range(len(y)):
        for j in range(len(y[i])):
            if y[i, j] == 1:
                new_col = SIMPLE_MAPPING[TARGET_COLUMNS[j]]
                q = SIMPLE_COLUMNS.index(new_col)
                new_y[i, q] = 1
    return new_y


# ###################################################### MAIN ######################################################

# Configuration
EVALUATION_MODE = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_RNN = False
TRAIN_TRANSFORMER = not TRAIN_RNN
SIMPLE = True
### Used as a source path in evaluation mode and as a destination path in training mode, should be used to manage multiple models if needed
MODEL_PATH = 'model/rnn_model.pt'
TRANSFORMER_PATH = 'model/transformer_model.pt'

SIMPLE_MODEL_PATH = 'model/simple_rnn_model.pt'
SIMPLE_TRANSFORMER_PATH = 'model/simple_transformer_model.pt'

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


# these columns remove redundant feature names.
SIMPLE_COLUMNS = [
    "Snow",
    "Clear",
    "Cloudy",
    "Hail",
    "Drizzle",
    "Fog",
    "Rain",
    "Haze",
    "Thunder"
]

SIMPLE_MAPPING = {
    "Blowing Snow": "Snow",
    "Clear": "Clear",
    "Cloudy": "Cloudy",
    "Drizzle": "Drizzle",
    "Fog": "Fog",
    "Freezing Drizzle": "Drizzle",
    "Freezing Fog": "Fog",
    "Freezing Rain": "Rain",
    "Haze": "Haze",
    "Heavy Rain": "Rain",
    "Heavy Rain Showers": "Rain",
    "Heavy Snow": "Snow",
    "Ice Pellets": "Hail",
    "Mainly Clear": "Clear",
    "Moderate Hail": "Hail",
    "Moderate Rain": "Rain",
    "Moderate Rain Showers": "Rain",
    "Moderate Snow": "Snow",
    "Mostly Cloudy": "Cloudy",
    "Rain": "Rain",
    "Rain Showers": "Rain",
    "Smoke": "Haze",
    "Snow": "Snow",
    "Snow Grains": "Snow",
    "Snow Pellets": "Snow",
    "Snow Showers": "Snow",
    "Thunderstorms": "Thunder"
}

if __name__ == '__main__':

    # Set relative hyperparams
    if TRAIN_RNN:
        ## RNN HYPERPARAMS
        LEARNING_RATE = 1e-04
        BATCH_SIZE = 256
        SEQUENCE_LENGTH = 300
        NUM_EPOCHS = 100
        HIDDEN_SIZE = 256
        NUM_LAYERS = 3

        NUM_HEADS = 16
        ## Regularize
        DROPOUT_RATE = 0.1
        L2_LAMBDA = 1e-03

        ## Early Stop
        PATIENCE = 10
        THRESHOLD = 1e-04
    else:
        ## TRANSFORMER HYPERPARAMS
        LEARNING_RATE = 1e-04
        BATCH_SIZE = 256
        SEQUENCE_LENGTH = 300
        NUM_EPOCHS = 50
        HIDDEN_SIZE = 128
        NUM_LAYERS = 3

        NUM_HEADS = 4
        ## Regularize
        DROPOUT_RATE = 0.1
        L2_LAMBDA = 1e-03

        ## Early Stop
        PATIENCE = 3
        THRESHOLD = 1e-04

    print('loading data...')
    df = pd.read_csv('processed_data/final.csv')
    scalar = StandardScaler()
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df[TARGET_COLUMNS].to_numpy()

    # converts output to 9-class vector
    if SIMPLE:
        y = simplify_data(y)

    X_main, X_test, y_main, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=0.1, shuffle=False)
    X_train = scalar.fit_transform(X_train)
    X_val = scalar.transform(X_val)

    X_train = torch.Tensor(X_train).to(DEVICE)
    X_val = torch.Tensor(X_val).to(DEVICE)
    X_test = torch.Tensor(X_test).to(DEVICE)
    y_train = torch.Tensor(y_train).to(DEVICE)
    y_val = torch.Tensor(y_val).to(DEVICE)
    y_test = torch.Tensor(y_test).to(DEVICE)

    train_dataset = SequencedDataset(X_train, y_train, SEQUENCE_LENGTH)
    val_dataset = SequencedDataset(X_val, y_val, SEQUENCE_LENGTH)
    test_dataset = SequencedDataset(X_test, y_test, SEQUENCE_LENGTH)

    print(f"training Data:\n\tX:\n\t{train_dataset.X}\n\ty:\n\t{train_dataset.y}")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    TRAIN_LENGTH = len(train_loader)

    if TRAIN_RNN:
        print('instantiating rnn model...')
        model = RecurrentNeuralNetwork(
            input_size=len(FEATURE_COLUMNS),
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            output_size=len(y[0]),
            device=DEVICE
        ).to(DEVICE)
    elif TRAIN_TRANSFORMER:
        print('instantiating transformer model...')
        model = Transformer(
            input_dim = len(FEATURE_COLUMNS),
            model_dim = HIDDEN_SIZE,
            num_heads = NUM_HEADS,
            num_layers = NUM_LAYERS,
            output_dim = len(y[0])    )



    if EVALUATION_MODE:
        if TRAIN_RNN:
            if not SIMPLE:
                print('loading rnn model...')
                model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device(DEVICE)))
            else:
                print('loading simple rnn model...')
                model.load_state_dict(torch.load(SIMPLE_MODEL_PATH, map_location=torch.device(DEVICE)))
        else:
            if not SIMPLE:
                print('loading transformer model...')
                model.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=torch.device(DEVICE)))
            else:
                print('loading simple transformer model...')
                model.load_state_dict(torch.load(SIMPLE_TRANSFORMER_PATH, map_location=torch.device(DEVICE)))
        model = model.to(DEVICE)
        model.eval()
        print_evaluation_metrics(model, val_dataset)

        print('calculating accuracy...')
        #print(calculate_prediction(model, X_val))
        #print(y_train[3])

        preds = calculate_prediction(model, test_dataset, 0.50)
        #print(preds[3])


        pred_simple = batch_decode(preds, TARGET_COLUMNS)
        class_simple = batch_decode(y_test, TARGET_COLUMNS)
        total = 0
        for pred, label in zip(pred_simple, class_simple):
            for c in pred:
                if c in label:
                    total += 1/len(label)
        total = total/len(pred_simple)
        print(f"Class Accuracy: {total * 100.:2f}%")

        #print(f"Total Identical Classifications: {multilabel_metrics(preds, y_test) * 100.:2f}%")
        print(y_test)
        #print(pred_simple)
        #print(class_simple)

        quit(0)
    if TRAIN_RNN:
        model.train()
        pos_weight = (y_train.size(0)) / (y_train.sum(dim=0) + 1e-6) # prefers positive predictions, should be inversely proportional to how rare the class is
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(pos_weight)).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_LAMBDA)

        print('starting rnn model training...')
        print(f'Max number of iterations: {len(train_loader) * NUM_EPOCHS}')
        iteration = 0
        best_loss = float('inf')
        best_model_state = None
        num_no_improve = 0
        losses = []
        for epoch in range(NUM_EPOCHS):
            start_time = time.time()
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
            losses.append(current_loss)
            if current_loss - best_loss >= THRESHOLD:
                num_no_improve += 1
                if num_no_improve >= PATIENCE:
                    break
            else:
                num_no_improve = 0
                if best_loss > current_loss:
                    best_loss = current_loss
                    best_model_state = copy.deepcopy(model.state_dict())
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Epoch {epoch}, Loss: {current_loss}, No Improvement Count: {num_no_improve}, Time Elapsed: {elapsed_time}")
            
        early_stop = 'due to early stop'
        finished = ''
        print(f'training complete {early_stop if num_no_improve >= PATIENCE else finished}')
        if SIMPLE:
            torch.save(best_model_state, SIMPLE_MODEL_PATH)
        else:
            torch.save(best_model_state, MODEL_PATH)
        print(losses)
    if TRAIN_TRANSFORMER:
        model.train()
        pos_weight = (y_train.size(0)) / (y_train.sum(
            dim=0) + 1e-6)  # prefers positive predictions, should be inversely proportional to how rare the class is
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(pos_weight)).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_LAMBDA)   # uses decoupled weight decay, better for regularizaion
        print('starting transformer model training...')

        scheduler = LambdaLR(optimizer, lr_lambda)

        iteration = 0
        best_loss = float('inf')
        best_model_state = None
        num_no_improve = 0
        losses = []

        for epoch in range(NUM_EPOCHS):
            start_time = time.time()
            for X_batch, y_batch in train_loader:
                inputs = X_batch.to(DEVICE)
                targets = y_batch.to(DEVICE)

                optimizer.zero_grad()
                preds = model(inputs)

                loss = criterion(preds, targets)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    # stabilizes training gradients
                optimizer.step()
                scheduler.step()    # updates learning rate dynamically

                iteration += 1

            current_loss = calculate_validation_loss(model, val_dataset, criterion)
            train_loss = calculate_validation_loss(model, train_dataset, criterion)
            losses.append(current_loss)
            if current_loss - best_loss >= THRESHOLD:
                num_no_improve += 1
                if num_no_improve >= PATIENCE:
                    break
            else:
                num_no_improve = 0
                if best_loss > current_loss:
                    best_loss = current_loss
                    best_model_state = copy.deepcopy(model.state_dict())
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Epoch {epoch+1:3} | Val Loss: {current_loss:.4f}, Train Loss: {train_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}, No Improvement Count: {num_no_improve}, Time Elapsed: {elapsed_time:.3f}s")

        early_stop = 'due to early stop'
        finished = ''
        print(f'training complete {early_stop if num_no_improve >= PATIENCE else finished}')
        print("saving transformer model...")

        if SIMPLE:
            torch.save(best_model_state, SIMPLE_TRANSFORMER_PATH)
        else:
            torch.save(best_model_state, TRANSFORMER_PATH)
        print(losses)

