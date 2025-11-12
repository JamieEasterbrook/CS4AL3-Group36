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


# Helper function to clean up data
def clean_data_split(x,y,model,threshold=0.5,test_size=1, scalar=1):

    # takes the most recent 100*n% of the dataset for testing
    split_index = int(len(x) * (1 - test_size))
    x_train = x[:split_index]; y_train = y[:split_index]
    x_val = x[split_index:]; y_val = y[split_index:]

    y_pred = [model(t.unsqueeze(0)) for t in x_val]
    for t in y_pred:
        t[t > threshold] = scalar;t[t < threshold] = 0
    y_val = [t * scalar for t in y_val]

    y_pred = np.array(y_pred).reshape(np.shape(y_val))
    return x_val, y_val, x_train,y_train, y_pred

# Helper function to find majority vote
def most_common_row(array):
    # Example array
    array = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0]
    ])
    tupled_array = [tuple(row) for row in array]
    most_common, count = Counter(tupled_array).most_common(1)[0]

    return most_common

def RMS_val (model,x,y,columns, feedback = False):
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
    x_val, y_val, x_train, y_train, y_pred = clean_data_split(x,y,model, threshold=0.33, test_size=1, scalar=1)

    MSE_list = np.square(np.array(y_val) - np.array(y_pred))
    MSE = np.sqrt( np.sum(MSE_list)/len(x_val))

    MSE_ZEROS = np.sqrt( np.sum(np.square(np.array(y_val) - np.zeros(np.shape(y_val))))/len(x_val))
    MSE_ONES = np.sqrt( np.sum(np.square(np.array(y_val) - np.ones(np.shape(y_val))))/len(x_val))

    MSE_MV = [np.array(most_common_row(y_val)).copy() for _ in range(5)]
    if feedback:
        if False:
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
        print(f"{'Zeros baseline':<20} {MSE_ZEROS:>10.2f} {MSE_ZEROS / MSE * 100:>14.2f}%")
        print(f"{'Ones baseline':<20} {MSE_ONES:>10.2f} {MSE_ONES / MSE * 100:>14.2f}%")
        print(f"{'Airport':<20} {AIRPORT_MSE:>10.2f} {AIRPORT_MSE / MSE * 100:>14.2f}%")
        print(f"{'FNN':<20} {FNN_MSE:>10.2f} {FNN_MSE / MSE * 100:>14.2f}%")
        print(f"{'Baseline RNN':<20} {RNN_MSE:>10.2f} {RNN_MSE / MSE * 100:>14.2f}%")
        print(f"{'Gated RNN':<20} {GRNN_MSE:>10.2f} {GRNN_MSE / MSE * 100:>14.2f}%")
    return MSE

def display_regularization_graphs():
    # These are hard-coded lists of past validation loss functions, involving experiements with disabling different features
    base_i = [259, 518, 777, 1036, 1295, 1554, 1813, 2072, 2331, 2590, 2849, 3108, 3367, 3626, 3885, 4144, 4403, 4662, 4921, 5180, 5439, 5698, 5957, 6216, 6475, 6734, 6993, 7252, 7511, 7770, 8029, 8288, 8547, 8806, 9065]
    base_l = [0.8224111923883701, 0.7607846116197521, 0.7235216107861749, 0.7033979558739168, 0.7140400425113481, 0.7171712635919966, 0.7033634709900823, 0.7223344378430268, 0.7568681301741764, 0.7362939417362213, 0.7261025597309244, 0.7358390545022899, 0.7067187541517718, 0.7436778663561262, 0.6898356229066849, 0.7599375556255209, 0.7580448579171608, 0.7462144745834942, 0.7498617824809305, 0.817752114657698, 0.8248936811397816, 0.8114052990387226, 0.7927848357578804, 0.9521028438004954, 0.9482944972556213, 0.9662410239207333, 0.9536401526681308, 0.9955887152203198, 0.9341389115514427, 1.1328056593393456, 0.9501691434917778, 1.0619981509858165, 1.095580369234085, 1.0307816503376797, 1.0063344638409286]
    base_min = base_i[base_l.index(min(base_l))]

    nodrop_i = [259, 518, 777, 1036, 1295, 1554, 1813, 2072, 2331, 2590, 2849, 3108, 3367, 3626, 3885, 4144, 4403, 4662, 4921, 5180, 5439, 5698, 5957, 6216, 6475, 6734, 6993, 7252]
    nodrop_l = [0.8008562269909628, 0.7750300641717582, 0.7218175802765221, 0.7411371705860927, 0.7053549192075071, 0.709787091304516, 0.6869584352805697, 0.6793997822136715, 0.7214385779767201, 0.7177866445533161, 0.7316570970518835, 0.7589235593532694, 0.7284658607737772, 0.8203662042987758, 0.7951704053015545, 0.7994675723643139, 0.7415394197250235, 0.7886455557469664, 0.7725729377105318, 0.7987078965224069, 0.8756770745947443, 0.8357863719093388, 0.9734390940645645, 0.9079331031133389, 0.8315142909514492, 0.94678845256567, 0.9105629884991152, 1.0756002613182725]
    nodrop_min = nodrop_i[nodrop_l.index(min(nodrop_l))]

    nol2_i =  [259, 518, 777, 1036, 1295, 1554, 1813, 2072, 2331, 2590, 2849, 3108, 3367, 3626, 3885, 4144, 4403, 4662, 4921, 5180, 5439, 5698, 5957, 6216, 6475, 6734, 6993, 7252]
    nol2_l =  [0.8500936406439749, 0.7288552538074297, 0.7302758524130131, 0.7139199360691267, 0.7260159007434187, 0.7193152750360554, 0.6972454683534031, 0.6799354866660875, 0.7187511962035606, 0.7931359466807596, 0.7898478081514095, 0.830856090475773, 0.7859551659945784, 0.7514553779158099, 0.8350463494144637, 0.7983631239882831, 0.8715473449435728, 0.9326345722736984, 0.8353322245951357, 0.8741458867644442, 0.8976207249637308, 0.9211722835898399, 0.8185988495062138, 1.0169079974293709, 1.1746697292245667, 0.9863868653774261, 1.1248631605814243, 1.191721010567813]
    nol2_min = nol2_i[nol2_l.index(min(nol2_l))]


    noreg_i = [259, 518, 777, 1036, 1295, 1554, 1813, 2072, 2331, 2590, 2849, 3108, 3367, 3626, 3885, 4144, 4403, 4662, 4921, 5180, 5439, 5698, 5957, 6216, 6475, 6734, 6993, 7252, 7511, 7770, 8029]
    noreg_l = [0.8423401749339597, 0.7618973481244055, 0.7295128713394033, 0.7162640706218523, 0.7256219741599313, 0.736338006011371, 0.6856191199401329, 0.6969360668083717, 0.7610355055537718, 0.7495100683179395, 0.6928497259986812, 0.7840738291370457, 0.8560626835658632, 0.882247546623493, 0.83258292592805, 0.8609962643220507, 0.8805847250182053, 0.9868240145773723, 1.5275654258399174, 1.089889288719358, 1.0612542557305302, 1.0601208883112874, 1.095444596276201, 1.081957690674683, 1.1849492310449994, 1.0647290690705693, 1.1179841487058277, 1.3307801230714238, 1.177698676956111, 1.4519095014909218, 1.4804649527730613]
    noreg_min = noreg_i[noreg_l.index(min(noreg_l))]

    chop = 1

    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3)
    plt.plot(base_i[:int(len(base_i)*chop)],base_l[:int(len(base_i)*chop)],label='l2 regularization and dropout layer', linestyle='-', color='green', marker='o')
    plt.plot(nodrop_i[:int(len(nodrop_i)*chop)],nodrop_l[:int(len(nodrop_l)*chop)],label='l2 regularization', linestyle='-', color='purple', marker='o')
    plt.plot(nol2_i[:int(len(nol2_i)*chop)],nol2_l[:int(len(nol2_l)*chop)],label='dropout layer', linestyle='-', color='red', marker='o')
    plt.plot(noreg_i[:int(len(noreg_i)*chop)],noreg_l[:int(len(noreg_l)*chop)],label='no regularization', linestyle='-', color='blue', marker='o')
    plt.axvline(base_min, color='green', linestyle='-', linewidth=2)
    plt.axvline(nodrop_min+25, color='purple', linestyle='-', linewidth=2)
    plt.axvline(nol2_min, color='red', linestyle='-', linewidth=2)
    plt.axvline(noreg_min, color='blue', linestyle='-', linewidth=2)
    plt.xlim(right=7000)
    plt.title('Comparizon of Different Regularization Methods on RNN Validation Loss')
    plt.legend()
    plt.savefig("report/Reg_Analysis.png", bbox_inches='tight')
    #plt.show()

display_regularization_graphs()
