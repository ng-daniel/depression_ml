print("--------------------------------------")
print("DEPRESSION CLASSIFICATION MODEL // V.0")
print("--------------------------------------")
print("Loading libraries...")

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

from data import load_dataframe_labels, preprocess_train_test_dataframes, create_feature_dataframe, create_dataloaders, kfolds_dataframes
from engine import train_test
from eval import eval_model, eval_forest_model
from model import ZeroR, ConvNN, LSTM, FeatureMLP
from util import print_model_performance_table

print("Loading raw dataframe...")

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load dataframe
actigraph_data, actigraph_labels = load_dataframe_labels(dir_names = ["data/control", "data/condition"],
                                                                    class_names = ["control", "condition"],
                                                                    time = "12:00:00", undersample=True)

print("Loading folds and extracting features...")

NUM_FOLDS = 4
NUM_DATAFRAMES = 4
kf_dfs = kfolds_dataframes(actigraph_data, actigraph_labels, numfolds=NUM_FOLDS, shuffle=True, random_state=42, batch_size=32)
kf_dataloaders = []
kf_feature_dfs = []
kf_feature_dataloaders = []
for i in tqdm(range(NUM_DATAFRAMES), ncols=50, leave=True):
      (X_train, X_test, y_train, y_test) = kf_dfs[i]
      (X_train_p, X_test_p) = preprocess_train_test_dataframes(X_train, X_test)
      
      # load actigraph data folds

      kf_dataloaders.append(
            create_dataloaders(X_train_p, X_test_p, y_train, y_test,
                                     shuffle = True, batch_size = 32)
      )

      # load feature folds

      X_train_features = create_feature_dataframe(data = X_train_p, raw_data = X_train)
      X_test_features = create_feature_dataframe(data = X_test_p, raw_data = X_test)
      kf_feature_dfs.append((X_train_features, X_test_features, y_train, y_test))
      kf_feature_dataloaders.append(
            create_dataloaders(X_train_features, X_test_features, y_train, y_test,
                               shuffle = True, batch_size = 32)
      )
      #print(f"{i+1}/{NUM_FOLDS}")

# define criterion
criterion = nn.BCEWithLogitsLoss().to(device)

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

# convNN training

IN_0 = 1
OUT_0 = 1
HIDDEN_0 = 32
FLATTEN_0 = 720
for i, (train_dataloader, test_dataloader) in enumerate(kf_dataloaders):
      # reset model
      model_0 = ConvNN(IN_0, OUT_0, HIDDEN_0, FLATTEN_0).to(device)
      optimizer = torch.optim.Adam(params = model_0.parameters(), lr=0.001)
      # train model
      print(f"fold_{i+1}...")
      train_test(model_0, train_dataloader, test_dataloader, epochs = 10, optimizer=optimizer, 
            criterion=criterion, device=device, verbose=False)
      results.append(
            eval_model(model = model_0,
                       note = f"fold_{i}",
                       dataloader = test_dataloader,
                       criterion = criterion,
                       device = device)
      )
print("----------------")

# MLP training

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
      results.append(
            eval_model(model = model_1,
                       note = f"fold_{i}",
                       dataloader = test_dataloader,
                       criterion = criterion,
                       device = device)
      )
print("----------------")

# random forest training

for i, (X_train, X_test, y_train, y_test) in enumerate(kf_feature_dfs):
      # reset model
      model_2 = RandomForestClassifier()
      # train model
      print(f"fold_{i+1}...")
      model_2.fit(X_train, y_train)
      results.append(
            eval_forest_model(model = model_2,
                              note = f"fold_{i}",
                              X_test=X_test,
                              y_test=y_test,
                              criterion = criterion)
      )
print("----------------")

print("Results:")

results_df = pd.concat(results, axis=1).transpose()
results_df.set_index(['model_name'])
print(results_df)

print("Done.")