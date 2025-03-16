print("--------------------------------------")
print("DEPRESSION CLASSIFICATION MODEL // V.0")
print("--------------------------------------")
print("Loading libraries...")

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from data import load_dataframe_labels, preprocessed_dataloaders, extract_features_from_window, kfolds_dataframes
from engine import train_test
from eval import eval_model
from model import ZeroR, ConvNN, LSTM
from util import print_model_performance_table

print("Loading data...")

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load dataframe
actigraph_data, actigraph_labels = load_dataframe_labels(dir_names = ["data/control", "data/condition"],
                                                                    class_names = ["control", "condition"],
                                                                    time = "12:00:00")

extract_features_from_window(actigraph_data.iloc[0])

# split data into folds
NUM_FOLDS = 5
kf_dataframes = kfolds_dataframes(actigraph_data, actigraph_labels, numfolds=NUM_FOLDS, shuffle=True, random_state=42, batch_size=32)
kf_dataloaders = []
for i in range(NUM_FOLDS):
      (X_train, X_test, y_train, y_test) = kf_dataframes[i]
      kf_dataloaders.append(
            preprocessed_dataloaders(X_train, X_test, y_train, y_test,
                                     shuffle = True, batch_size = 32)
      )

# model hyperparameters
IN_SHAPE = 1
OUT_SHAPE = 1
HIDDEN_SHAPE = 32
FLATTEN_FACTOR = 720

# construct CNN model
model_0 = ConvNN(IN_SHAPE, OUT_SHAPE, HIDDEN_SHAPE, FLATTEN_FACTOR).to(device)
init_params = model_0.state_dict()

# define criterion
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(params = model_0.parameters(), lr=0.001)

results = []
print("Training...")
print("----------------")

# zeroR baseline
for i, (train_dataloader, test_dataloader) in enumerate(kf_dataloaders):
      model_0R = ZeroR(device)
      results.append(
            eval_model(model = model_0R,
                       note = f"fold_{i}",
                       dataloader = test_dataloader,
                       criterion = criterion,
                       device = device)
      )
results.append({})

# convNN training
for i, (train_dataloader, test_dataloader) in enumerate(kf_dataloaders):
      # reset model
      model_0 = ConvNN(IN_SHAPE, OUT_SHAPE, HIDDEN_SHAPE, FLATTEN_FACTOR).to(device)
      optimizer = torch.optim.Adam(params = model_0.parameters(), lr=0.001)
      # train model
      print(f"fold_{i+1}...")
      train_test(model_0, train_dataloader, test_dataloader, epochs = 10, optimizer=optimizer, 
            criterion=criterion, device=device, verbose=True)
      results.append(
            eval_model(model = model_0,
                       note = f"fold_{i}",
                       dataloader = test_dataloader,
                       criterion = criterion,
                       device = device)
      )
print("----------------")

print("Evaluating...")
print_model_performance_table(results)

print("Done.")