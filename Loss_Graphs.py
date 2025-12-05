import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fontTools.misc.cython import returns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from main import RecurrentNeuralNetwork
import datetime as dt
from collections import Counter

def display_regularization_graphs():
    # These are hard-coded lists of past validation loss functions, involving experiements with disabling different features
    rnn_loss = [1.053846816221873, 0.9378917614618937, 0.8742596904436747, 0.865005095799764, 0.8460609118143717, 0.8556694587071737, 0.856731136639913, 0.8680587609608968, 0.8774961233139038, 0.8659790754318237, 0.8535849650700887, 0.8840540846188863, 0.8670772512753805, 0.8943798542022705, 0.8784773548444113]

    simple_rnn_loss = [1.0501125852266948, 0.9466719627380371, 0.8895710706710815, 0.8665611743927002, 0.8677223126093546, 0.8539268374443054, 0.8236783544222513, 0.8377448916435242, 0.8314360976219177, 0.8542728424072266, 0.8607694506645203, 0.8555105129877726, 0.8483902414639791, 0.8720333178838094, 0.8671643932660421, 0.8745713432629904, 0.8758355180422465]


    simple_transformer_loss = [1.1659323771794636, 1.0118072628974915, 0.8677447239557902, 0.7750234603881836,
                               0.7504841685295105, 0.7456353505452474, 0.7471621433893839, 0.7382955352465311,
                               0.7598686019579569, 0.7876739899317423, 0.76479172706604, 0.8037663896878561,
                               0.7982682188351949, 0.8469982345898946, 0.8572281797726949, 0.8576135635375977,
                               0.9629906415939331, 0.8613306085268656]

    transformer_loss = [
        1.1641,
        1.0118,
        0.8819,
        0.8919,
        0.9101,
        0.8902,
        0.8999,
        0.9485,
        0.8924,
        0.9363,
        0.8902,
        1.0288
    ]

    chop = 1

    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3)
    plt.plot(range(len(rnn_loss)),rnn_loss,label='RNN model', linestyle='-', color='green', marker='o')
    plt.plot(range(len(simple_rnn_loss)),simple_rnn_loss,label='simplified RNN model', linestyle='-', color='purple', marker='o')
    plt.plot(range(len(transformer_loss)), transformer_loss,label='transformer model', linestyle='-', color='blue', marker='o')
    plt.plot(range(len(simple_transformer_loss)),simple_transformer_loss,label='simplified transformer model', linestyle='-', color='red', marker='o')
    plt.axvline(rnn_loss.index(min(rnn_loss)), color='green', linestyle='-', linewidth=2)
    plt.axvline(simple_rnn_loss.index(min(simple_rnn_loss)), color='purple', linestyle='-', linewidth=2)
    plt.axvline(transformer_loss.index(min(transformer_loss)), color='blue', linestyle='-', linewidth=2)
    plt.axvline(simple_transformer_loss.index(min(simple_transformer_loss)), color='red', linestyle='-', linewidth=2)
    plt.title('Comparison of Validation Loss for RNN and Transformer Implementations')
    plt.legend()
    plt.xlim(0,10)
    plt.savefig("report/Loss_Analysis.png", bbox_inches='tight')
    #plt.show()

display_regularization_graphs()


















"""
SIMPLE RNN
Baseline: Accuracy on all zero predictions: 88.48848849%
Baseline: Recall on all zero predictions: 0.00000000%
Baseline: Precision on all zero predictions: 0.00000000%
Baseline: F1 score on all one predictions: 0.00000000

calculating accuracy, y_pred shape: torch.Size([666, 9]), y size: torch.Size([666, 9])
Counts: tp: 690, tn: 0, fp: 5304, fn: 0
Baseline: Accuracy on all one predictions: 11.51151151%
Baseline: Recall on all one predictions: 100.00000000%
Baseline: Precision on all one predictions: 11.51151151%
Baseline: F1 score on all one predictions: 0.20644468

calculating accuracy, y_pred shape: torch.Size([666, 9]), y size: torch.Size([666, 9])
Counts: tp: 237, tn: 4875, fp: 429, fn: 453
tensor([0., 0., 1., 0., 0., 0., 0., 0., 0.])
Baseline: Accuracy on most common vote predictions: 85.28528529%
Baseline: Recall on most common vote predictions: 34.34782609%
Baseline: Precision on most common vote predictions: 35.58558559%
Baseline: F1 score on most common vote predictions: 0.34950754

calculating accuracy, y_pred shape: torch.Size([666, 9]), y size: torch.Size([666, 9])
Counts: tp: 597, tn: 4089, fp: 1215, fn: 93
Model accuracy: 78.17817818%
Model recall: 86.52173913%
Model precision: 32.94701987%
Model F1 score: 0.47717828 
"""

"""
SIMPLE TRANSFORMER
Baseline: Accuracy on all zero predictions: 88.48848849%
Baseline: Recall on all zero predictions: 0.00000000%
Baseline: Precision on all zero predictions: 0.00000000%
Baseline: F1 score on all one predictions: 0.00000000

calculating accuracy, y_pred shape: torch.Size([666, 9]), y size: torch.Size([666, 9])
Counts: tp: 690, tn: 0, fp: 5304, fn: 0
Baseline: Accuracy on all one predictions: 11.51151151%
Baseline: Recall on all one predictions: 100.00000000%
Baseline: Precision on all one predictions: 11.51151151%
Baseline: F1 score on all one predictions: 0.20644468

calculating accuracy, y_pred shape: torch.Size([666, 9]), y size: torch.Size([666, 9])
Counts: tp: 237, tn: 4875, fp: 429, fn: 453
tensor([0., 0., 1., 0., 0., 0., 0., 0., 0.])
Baseline: Accuracy on most common vote predictions: 85.28528529%
Baseline: Recall on most common vote predictions: 34.34782609%
Baseline: Precision on most common vote predictions: 35.58558559%
Baseline: F1 score on most common vote predictions: 0.34950754

calculating accuracy, y_pred shape: torch.Size([666, 9]), y size: torch.Size([666, 9])
Counts: tp: 582, tn: 4250, fp: 1054, fn: 108
Model accuracy: 80.61394728%
Model recall: 84.34782609%
Model precision: 35.57457213%
Model F1 score: 0.50038820 
"""