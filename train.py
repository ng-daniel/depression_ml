print("--------------------------------------")
print("DEPRESSION CLASSIFICATION MODEL // V.0")
print("--------------------------------------")
print("PROCESS: TRAINING AND EVALUATION")
print("Loading libraries...")

import os

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

from data import create_dataloaders
from engine import train_test
from eval import eval_model, eval_forest_model
from model import ZeroR, ConvNN, LSTM, FeatureMLP

print("Loading raw dataframe...")

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading dataframes from files...")

kf_actigraphy_dfs = []
kf_feature_dfs = []

kf_actigraphy_dataloaders = []
kf_feature_dataloaders = []

A_DIR = "data/processed_dataframes/actigraphy"
num_a = len(os.listdir(A_DIR))
for i in range(int(num_a / 2)):
      
      # load train data

      X_train = pd.read_csv(os.path.join(A_DIR, f"a{i}t.csv"), index_col=0)
      y_train = list(X_train['label'])
      
      # load test data

      X_test = pd.read_csv(os.path.join(A_DIR, f"a{i}e.csv"), index_col=0)
      y_test = list(X_test['label'])

      # drop label column

      X_train.drop('label', axis=1, inplace=True)
      X_test.drop('label', axis=1, inplace=True)

      print(X_train)
      print(y_train)

      # load dataframes

      kf_actigraphy_dfs.append((X_train, X_test, y_train, y_test))
      kf_actigraphy_dataloaders.append(
            create_dataloaders(X_train, X_test, y_train, y_test, shuffle=True, batch_size=32)
      )

F_DIR = "data/processed_dataframes/feature"
num_f = len(os.listdir(F_DIR))
for i in range(int(num_f / 2)):
      
      # load train data

      X_train = pd.read_csv(os.path.join(F_DIR, f"f{i}t.csv"), index_col=0)
      y_train = list(X_train['label'])
      
      # load test data

      X_test = pd.read_csv(os.path.join(F_DIR, f"f{i}e.csv"), index_col=0)
      y_test = list(X_test['label'])
      
      # drop label column

      X_train.drop('label', axis=1, inplace=True)
      X_test.drop('label', axis=1, inplace=True)

      # load dataframes

      kf_feature_dfs.append((X_train, X_test, y_train, y_test))
      kf_feature_dataloaders.append(
            create_dataloaders(X_train, X_test, y_train, y_test, shuffle=True, batch_size=32)
      )

print("Training...")
print("----------------")

RESULTS_DIR = "results"

# define criterion
criterion = nn.BCEWithLogitsLoss().to(device)

# zeroR baseline

zeroR_results = []
print(len(kf_actigraphy_dataloaders))
for i, (train_dataloader, test_dataloader) in enumerate(kf_actigraphy_dataloaders):
      model_0R = ZeroR(device)
      zeroR_results.append(
            eval_model(model = model_0R,
                       note = f"fold_{i}",
                       dataloader = test_dataloader,
                       criterion = criterion,
                       device = device)
      )
print(zeroR_results)
zeroR_results = pd.concat(zeroR_results, axis=1).transpose()
zeroR_results.to_csv(os.path.join(RESULTS_DIR, "zeroR"))

# convNN training

cnn_results = []
IN_0 = 1
OUT_0 = 1
HIDDEN_0 = 32
FLATTEN_0 = 720
for i, (train_dataloader, test_dataloader) in enumerate(kf_actigraphy_dataloaders):
      # reset model
      model_0 = ConvNN(IN_0, OUT_0, HIDDEN_0, FLATTEN_0).to(device)
      optimizer = torch.optim.Adam(params = model_0.parameters(), lr=0.001)
      # train model
      print(f"fold_{i+1}...")
      train_test(model_0, train_dataloader, test_dataloader, epochs = 10, optimizer=optimizer, 
            criterion=criterion, device=device, verbose=False)
      cnn_results.append(
            eval_model(model = model_0,
                       note = f"fold_{i}",
                       dataloader = test_dataloader,
                       criterion = criterion,
                       device = device)
      )
cnn_results = pd.concat(cnn_results, axis=1).transpose()
cnn_results.to_csv(os.path.join(RESULTS_DIR, "cnn"))

# MLP training

mlp_results = []
IN_1 = 756
OUT_1 = 1
HIDDEN_1 = 128
for i, (train_dataloader, test_dataloader) in enumerate(kf_feature_dataloaders):
      # reset model
      model_1 = FeatureMLP(IN_1, OUT_1, HIDDEN_1).to(device)
      optimizer = torch.optim.Adam(params = model_1.parameters(), lr=0.01)
      # train model
      print(f"fold_{i+1}...")
      train_test(model_1, train_dataloader, test_dataloader, epochs = 10, optimizer=optimizer, 
            criterion=criterion, device=device, verbose=False)
      mlp_results.append(
            eval_model(model = model_1,
                       note = f"fold_{i}",
                       dataloader = test_dataloader,
                       criterion = criterion,
                       device = device)
      )
mlp_results = pd.concat(mlp_results, axis=1).transpose()
mlp_results.to_csv(os.path.join(RESULTS_DIR, "mlp"))

# random forest training

forest_results = []
for i, (X_train, X_test, y_train, y_test) in enumerate(kf_feature_dfs):
      # reset model
      model_2 = RandomForestClassifier()
      # train model
      print(f"fold_{i+1}...")
      model_2.fit(X_train, y_train)
      forest_results.append(
            eval_forest_model(model = model_2,
                              note = f"fold_{i}",
                              X_test=X_test,
                              y_test=y_test,
                              criterion = criterion)
      )
forest_results = pd.concat(forest_results, axis=1).transpose()
forest_results.to_csv(os.path.join(RESULTS_DIR, "random_forest"))

print("Done.")