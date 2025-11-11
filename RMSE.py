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


def RMS_val (model,x,y,test_size = 0.25, feedback = False):
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

    Gated Recurrent neural network: MSE = 2.96
    This model uses a gated recurrent unit to help determine which information to forget, helping work in tomporal information better.
    This model should be considered an upgrade to our current iteration, and while we are not expecting to surpass it, we
    should try to aim close to this score anyways.
    """

    AIRPORT_MSE = 19.66
    FNN_MSE = 31.28
    RNN_MSE = 4.72
    GRNN_MSE = 2.96

    total_months = 147

    # takes the most recent 100*n% of the dataset for testing
    split_index = int(len(x) * (1 - test_size))
    x_train = x[:split_index]; y_train = y[:split_index]
    x_val = x[split_index:]; y_val = y[split_index:]

    y_pred = [model(t.unsqueeze(0)) for t in x_val]

    y_pred = np.array(y_pred).reshape(np.shape(y_val))

    MSE_list = np.square(np.array(y_val) - np.array(y_pred))
    MSE = np.sum(MSE_list)/len(x_val)

    if feedback:
        plt.plot(range(len(x_val)), MSE_list)
        plt.title(f'Per-point MSE over the last {total_months*test_size:.0f} months')
        plt.xlabel('Date')
        plt.ylabel('Mean Squared Error')
        plt.show(block=False)
        plt.pause(5)

        print(f"{'-' * 16} MSE Comparison {'-' * 16}")
        print(f"{'Model':<20} {'MSE':>10} {'Improvement':>16}")
        print(f"{'-' * 48}")
        print(f"{'Our RNN':<20} {MSE:>10.2f} {'-':>15}")
        print(f"{'Airport':<20} {AIRPORT_MSE:>10.2f} {AIRPORT_MSE / MSE * 100:>14.2f}%")
        print(f"{'FNN':<20} {FNN_MSE:>10.2f} {FNN_MSE / MSE * 100:>14.2f}%")
        print(f"{'Baseline RNN':<20} {RNN_MSE:>10.2f} {RNN_MSE / MSE * 100:>14.2f}%")
        print(f"{'Gated RNN':<20} {GRNN_MSE:>10.2f} {GRNN_MSE / MSE * 100:>14.2f}%")
    return
