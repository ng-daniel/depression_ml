print("--------------------------------------")
print("DEPRESSION CLASSIFICATION MODEL // V.0")
print("--------------------------------------")
print("Loading libraries...")

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from data import load_preprocess_dataframe_labels, train_test_dataloaders, kfolds_dataloaders
from engine import train_test
from eval import eval_model
from model import ConvNN, LSTM
from util import print_model_performance_table

print("Loading data...")

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load dataframe
actigraph_data, actigraph_labels = load_preprocess_dataframe_labels(dir_names = ["data/control", "data/condition"],
                                                                    class_names = ["control", "condition"],
                                                                    time = "12:00:00")
# load dataloaders
kf_dataloaders = kfolds_dataloaders(actigraph_data, actigraph_labels, numfolds=4, shuffle=True, random_state=42, batch_size=32)

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
for i, (train_dataloader, test_dataloader) in enumerate(kf_dataloaders):
      # reset model
      model_0 = ConvNN(IN_SHAPE, OUT_SHAPE, HIDDEN_SHAPE, FLATTEN_FACTOR).to(device)
      optimizer = torch.optim.Adam(params = model_0.parameters(), lr=0.001)

      # train model
      print(f"fold_{i+1}...")
      train_test(model_0, train_dataloader, test_dataloader, epochs = 10, optimizer=optimizer, 
            criterion=criterion, device=device, verbose=True)
      results.append(eval_model(model = model_0, 
                                note = f"fold_{i}",
                                dataloader = test_dataloader,
                                criterion = criterion,
                                device = device))
print("----------------")

print("Evaluating...")
print_model_performance_table(results)

print("Done.")