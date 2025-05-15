print("--------------------------------")
print("DEPRESSION CLASSIFICATION // V.0")
print("--------------------------------")
print("PROCESS: TRAINING AND EVALUATION")
print("Loading libraries...")

import os
import pandas as pd
import torch
import torch.nn as nn
from data import create_dataloaders
from eval import create_metrics_table
from training_loops import (run_linear_svc, run_decision_tree, run_cnn, run_lstm, run_lstm_feature, 
                            run_mlp, run_random_forest, run_zeroR_baseline)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# read actigraphy data and extracted features data to respective lists from 
# the appropriate directories and create dataloaders for both types of data
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

# read feature series data for LSTM v2 into 
# respective lists and create dataloaders for them
kf_series_tensors = []
kf_series_dataloaders = []
FS_DIR = "data/processed_dataframes/feature_series"
num_fs = len(os.listdir(os.path.join(FS_DIR, "kfolds")))
for i in range(int(num_fs / 2)):
      
      # test data csv loading + to tensor
      test_tensors = []
      test_labels = []
      test_directory = os.path.join(FS_DIR, "kfolds", f"fs{i}e.txt")
      with open(test_directory, "r") as file:
            for sample_name in file:
                  df = pd.read_csv(os.path.join(FS_DIR, "csv_files", sample_name.strip() + ".csv"))
                  test_tensor = torch.tensor(df.to_numpy())
                  test_label = int(sample_name[0])
                  test_tensors.append(test_tensor)

      # train data csv loading + to tensor
      train_tensors = []
      train_labels = []
      train_directory = os.path.join(FS_DIR, "kfolds", f"fs{i}t.txt")
      with open(train_directory, "r") as file:
            for sample_name in file:
                  df = pd.read_csv(os.path.join(FS_DIR, "csv_files", sample_name.strip() + ".csv"))
                  train_tensor = torch.tensor(df.to_numpy())
                  train_label = int(sample_name[0])
                  train_tensors.append(train_tensor)

      # convert train and test tensors into single 3D tensors
      test_tensor = torch.stack(test_tensors, dim = 0)
      train_tensor = torch.stack(train_tensors, dim = 0)
      test_labels = torch.tensor(test_labels)
      train_labels = torch.tensor(train_labels)

      # create and store raw tensors + dataloaders
      kf_series_tensors.append((train_tensors, test_tensor, train_labels, test_label))
      kf_series_dataloaders.append(
            create_dataloaders(X_train, X_test, y_train, y_test, shuffle=True, batch_size=64)
      )


# setup output directory, class weights, and loss function, 
# aka criterion, for model training and evaluation
RESULTS_DIR = "results"
class_weights = torch.tensor([1.1]).to(device)
class_weights_dict = {
      0 : 1,
      1 : class_weights.item()
}
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device)
NUM_FEATURES = len(kf_feature_dfs[0][0].columns)


# LINEAR SUPPORT VECTOR MACHINE CLASSIFIER
linear_svc_results = run_linear_svc(data=kf_feature_dfs, 
                                    criterion=criterion,
                                    device=device,
                                    weights=class_weights_dict)
linear_svc_results.to_csv(os.path.join(RESULTS_DIR, "linear_svc.csv"))


# DECISION TREE
decision_tree_results = run_decision_tree(data=kf_feature_dfs,
                                          criterion=criterion,
                                          device=device,
                                          weights=class_weights_dict)
decision_tree_results.to_csv(os.path.join(RESULTS_DIR, "decision_tree.csv"))


# RANDOM FOREST
forest_results = run_random_forest(data=kf_feature_dfs,
                                   criterion=criterion,
                                   device=device,
                                   weights=class_weights_dict,
                                   n_estimators=300)
forest_results.to_csv(os.path.join(RESULTS_DIR, "random_forest.csv"))


# ZERO BASELINE
zeroR_results = run_zeroR_baseline(data=kf_actigraphy_dataloaders,
                                   criterion=criterion,
                                   device=device)
zeroR_results.to_csv(os.path.join(RESULTS_DIR, "zeroR.csv"))


# LONG SHORT TERM MEMORY NEURAL NETWORK V1
lstm_results = run_lstm(data=kf_actigraphy_dataloaders,
                        criterion=criterion,
                        device=device,
                        learning_rate=0.005,
                        epochs=20,
                        in_shape=60,
                        out_shape=1,
                        hidden_shape=16,
                        lstm_layers=1)
lstm_results.to_csv(os.path.join(RESULTS_DIR, "lstm.csv"))


# CONVOLUTIONAL NEURAL NETWORK
cnn_results = run_cnn(data=kf_actigraphy_dataloaders,
                      criterion=criterion,
                      device=device,
                      learning_rate=0.001,
                      epochs=20,
                      in_shape=1,
                      out_shape=1,
                      hidden_shape=32,
                      flatten_factor=720)
cnn_results.to_csv(os.path.join(RESULTS_DIR, "cnn.csv"))


# MULTILAYER PERCEPTRON NEURAL NETWORK
mlp_results = run_mlp(data=kf_feature_dataloaders,
                      criterion=criterion,
                      device=device,
                      learning_rate=0.005,
                      epochs=30,
                      in_shape=NUM_FEATURES,
                      out_shape=1,
                      hidden_shape=128)
mlp_results.to_csv(os.path.join(RESULTS_DIR, "mlp.csv"))


# LONG SHORT TERM MEMORY NEURAL NETWORK V2
lstm_series_results = run_lstm_feature(data=kf_series_dataloaders,
                                       criterion=criterion,
                                       device=device,
                                       learning_rate=0.0004,
                                       epochs=200,
                                       in_shape=NUM_FEATURES,
                                       out_shape=1,
                                       hidden_shape=16,
                                       lstm_layers=8)
lstm_series_results.to_csv(os.path.join(RESULTS_DIR, "lstm_series.csv"))


# aggregate all metrics and output the summary table
metrics = create_metrics_table([zeroR_results, cnn_results, mlp_results, forest_results, lstm_results, 
                              lstm_series_results, linear_svc_results, decision_tree_results])
metrics.to_csv(os.path.join(RESULTS_DIR, "results.csv"))
print(metrics)

print("Done.")