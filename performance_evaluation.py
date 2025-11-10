import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from main import RecurrentNeuralNetwork
import datetime as dt


def validation (x,y,test_size = 0.25, feedback = False):
    """
    While these metrics are from a different dataset, the approach should still be the determining factor, so we will compare.
    Our metrics to compare, from https://doi.org/10.1016/j.buildenv.2021.107601 are as follows:
    Base airport data: MSE = 19.66
    We want to beat this score to prove that our model surpassed a physics-based approach.

    Feed-forward neural network: MSE = 31.28
    This was a basic non-recurrent implementation, and should be our absolute base case to determine if the model is good or not.

    Recurrent neural network: MSE = 4.72
    This is the RNN that the paper implemented, and our target goal for this project. Approaching a MSE similar to this will show
    whether the model can accurately predict weather values.
    """

    AIRPORT_MSE = 19.66
    FNN_MSE = 31.28
    RNN_MSE = 4.72

    total_months = 147

    # takes the most recent 100*n% of the dataset for testing
    split_index = int(len(x) * (1 - test_size))
    x_train = x[:split_index]; y_train = y[:split_index]
    x_val = x[split_index:]; y_val = y[split_index:]

    # TODO make fully compatible with model
    model = RecurrentNeuralNetwork() #add hyperparams
    model.train(x_train)

    y_pred = [model.predict(t) for t in x_val]

    MSE = np.square(y_val - y_pred)

    #dates = [dt.datetime(row['Year'],row['Month'],row['Day']) for row in x_val]

    if feedback:
        plt.plot(x_val['DateTime'], MSE)
        plt.title(f'Per-point MSE over the last {total_months*test_size:.0f} months')
        plt.xlabel('Date')
        plt.ylabel('Mean Squared Error')
        plt.show()

        print("-----MSE Comparison-----")
        print(f"Our RNN MSE: {MSE}")
        print(f"Airport MSE: {AIRPORT_MSE}      Improvement: {AIRPORT_MSE/MSE*100}%")
        print(f"FNN MSE: {FNN_MSE}      Improvement: {FNN_MSE/MSE*100}%")
        print(f"Baseline RNN MSE: {RNN_MSE}      Improvement: {RNN_MSE/MSE*100}%")
    return