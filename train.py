print("--------------------------------")
print("DEPRESSION CLASSIFICATION // V.0")
print("--------------------------------")
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
from eval import eval_model, eval_forest_model, append_weighted_average, create_metrics_table
from model import ZeroR, ConvNN, LSTM, FeatureMLP

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

print("Training models...")

RESULTS_DIR = "results"

# define criterion
criterion = nn.BCEWithLogitsLoss().to(device)

#
# zeroR baseline
#

print("ZeroR Baseline:")

zeroR_results = []
for i, (train_dataloader, test_dataloader) in enumerate(tqdm(kf_actigraphy_dataloaders, ncols=50)):
      model_0R = ZeroR()
      zeroR_results.append(
            eval_model(model = model_0R,
                       note = f"{i}",
                       dataloader = test_dataloader,
                       criterion = criterion,
                       device = device)
      )
zeroR_results = pd.concat(zeroR_results, axis=1).transpose()
zeroR_results = append_weighted_average(zeroR_results)
zeroR_results.to_csv(os.path.join(RESULTS_DIR, "zeroR.csv"))

#
# LSTM training
#

print("LSTM:")

lstm_results = []
IN_2 = 60
OUT_2 = 1
HIDDEN_2 = 16
LSTM_LAYERS = 1
for i, (train_dataloader, test_dataloader) in enumerate(tqdm(kf_actigraphy_dataloaders, ncols=50)):
      # reset model
      model_2 = LSTM(IN_2, OUT_2, HIDDEN_2, LSTM_LAYERS).to(device)
      
      optimizer = torch.optim.RMSprop(params = model_2.parameters(), lr=0.0005)
      # train model
      train_test(model_2, train_dataloader, test_dataloader, epochs = 10, optimizer=optimizer, 
            criterion=criterion, device=device, verbose=True)
      lstm_results.append(
            eval_model(model = model_2,
                       note = f"{i}",
                       dataloader = test_dataloader,
                       criterion = criterion,
                       device = device)
      )
lstm_results = pd.concat(lstm_results, axis=1).transpose()
lstm_results = append_weighted_average(lstm_results)
lstm_results.to_csv(os.path.join(RESULTS_DIR, "lstm.csv"))

#
# convNN training
#

print("1D Convolutional NN:")

cnn_results = []
IN_0 = 1
OUT_0 = 1
HIDDEN_0 = 32
FLATTEN_0 = 720
for i, (train_dataloader, test_dataloader) in enumerate(tqdm(kf_actigraphy_dataloaders, ncols=50)):
      # reset model
      model_0 = ConvNN(IN_0, OUT_0, HIDDEN_0, FLATTEN_0).to(device)
      optimizer = torch.optim.Adam(params = model_0.parameters(), lr=0.001)
      # train model
      train_test(model_0, train_dataloader, test_dataloader, epochs = 10, optimizer=optimizer, 
            criterion=criterion, device=device, verbose=False)
      cnn_results.append(
            eval_model(model = model_0,
                       note = f"{i}",
                       dataloader = test_dataloader,
                       criterion = criterion,
                       device = device)
      )
cnn_results = pd.concat(cnn_results, axis=1).transpose()
cnn_results = append_weighted_average(cnn_results)
cnn_results.to_csv(os.path.join(RESULTS_DIR, "cnn.csv"))

#
# MLP training
#

print("Extracted Features MLP:")

mlp_results = []
IN_1 = len(kf_feature_dfs[0][0].columns)
OUT_1 = 1
HIDDEN_1 = 128
for i, (train_dataloader, test_dataloader) in enumerate(tqdm(kf_feature_dataloaders, ncols=50)):
      # reset model
      model_1 = FeatureMLP(IN_1, OUT_1, HIDDEN_1).to(device)
      optimizer = torch.optim.Adam(params = model_1.parameters(), lr=0.005)
      # train model
      train_test(model_1, train_dataloader, test_dataloader, epochs = 30, optimizer=optimizer, 
            criterion=criterion, device=device, verbose=False)
      mlp_results.append(
            eval_model(model = model_1,
                       note = f"{i}",
                       dataloader = test_dataloader,
                       criterion = criterion,
                       device = device)
      )
mlp_results = pd.concat(mlp_results, axis=1).transpose()
mlp_results = append_weighted_average(mlp_results)
mlp_results.to_csv(os.path.join(RESULTS_DIR, "mlp.csv"))

#
# random forest training
#

print("Extracted Features Random Forest:")

forest_results = []
for i, (X_train, X_test, y_train, y_test) in enumerate(tqdm(kf_feature_dfs, ncols=50)):
      # reset model
      model_2 = RandomForestClassifier()
      # train model
      model_2.fit(X_train, y_train)
      forest_results.append(
            eval_forest_model(model = model_2,
                              note = f"{i}",
                              X_test=X_test,
                              y_test=y_test,
                              criterion = criterion)
      )
forest_results = pd.concat(forest_results, axis=1).transpose()
forest_results = append_weighted_average(forest_results)
forest_results.to_csv(os.path.join(RESULTS_DIR, "random_forest.csv"))

print("Aggregating metrics...")

metrics = create_metrics_table([zeroR_results, cnn_results, mlp_results, forest_results, lstm_results])
metrics.to_csv(os.path.join(RESULTS_DIR, "results.csv"))
print(metrics)

print("Done.")