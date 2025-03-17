print("--------------------------------------")
print("DEPRESSION CLASSIFICATION MODEL // V.0")
print("--------------------------------------")
print("Loading libraries...")

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from data import load_dataframe_labels, preprocess_train_test_dataframes, create_feature_dataframe, create_dataloaders, kfolds_dataframes
from engine import train_test
from eval import eval_model
from model import ZeroR, ConvNN, LSTM, FeatureMLP
from util import print_model_performance_table

print("Loading raw dataframe...")

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load dataframe
actigraph_data, actigraph_labels = load_dataframe_labels(dir_names = ["data/control", "data/condition"],
                                                                    class_names = ["control", "condition"],
                                                                    time = "12:00:00")

print("Loading folds...")

NUM_FOLDS = 5
kf_dfs = kfolds_dataframes(actigraph_data, actigraph_labels, numfolds=NUM_FOLDS, shuffle=True, random_state=42, batch_size=32)
kf_dataloaders = []
kf_feature_dfs = []
kf_feature_dataloaders = []
for i in range(NUM_FOLDS):
      (X_train, X_test, y_train, y_test) = kf_dfs[i]
      (X_train_p, X_test_p) = preprocess_train_test_dataframes(X_train, X_test)
      
      # load actigraph data folds

      kf_dataloaders.append(
            create_dataloaders(X_train_p, X_test_p, y_train, y_test,
                                     shuffle = True, batch_size = 32)
      )

      # load feature folds

      X_train_features = create_feature_dataframe(data = X_train_p, raw_data = X_train)
      print(X_train_features.shape)
      X_test_features = create_feature_dataframe(data = X_test_p, raw_data = X_test)
      kf_feature_dfs.append((X_train_features, X_test_features, ))
      kf_feature_dataloaders.append(
            create_dataloaders(X_train_features, X_test_features, y_train, y_test,
                               shuffle = True, batch_size = 32)
      )

# model hyperparameters
IN_0 = 1
OUT_0 = 1
HIDDEN_0 = 32
FLATTEN_0 = 720
# construct CNN model
model_0 = ConvNN(IN_0, OUT_0, HIDDEN_0, FLATTEN_0).to(device)
init_params = model_0.state_dict()

#IN_1 = 

#model_1 = FeatureMLP()

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
      model_0 = ConvNN(IN_0, OUT_0, HIDDEN_0, FLATTEN_0).to(device)
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