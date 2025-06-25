print("--------------------------------")
print("DEPRESSION CLASSIFICATION // V.0")
print("--------------------------------")
print("PROCESS: CONV NN TRAINING")
print("Importing libraries...")

import os
from tqdm import tqdm
import random
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

from core.data import create_dataloaders, process_data_folds
from core.training_loops import run_cnn
from core.eval import combine_several_weighted_averages

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_DIR = "data/processed_dataframes/"
RAW_PTH = DATA_DIR + "/data_raw.csv"
SMOTE_PTH = DATA_DIR + "/data_resampled.csv"
KFOLD_DIR = DATA_DIR + "/kfolds"
KFOLD_SMOTE_DIR = DATA_DIR + "/kfolds_smote"
NUM_FOLDS = len(os.listdir(KFOLD_DIR))//2
RESULTS_DIR = "results"

preprocessing_settings = {
    'resample' : True,
    'log_base' : None,
    'scale_range' : None,
    'use_standard' : True,
    'use_gaussian' : 30,
    'subtract_mean' : True,
    'adjust_seasonality' : True,
}
feature_settings = {
    'use_feature' : False,
    'long_feature' : True,
    'window_size' : 30,
    'quarter_diff' : False,
    'simple' : True
}
hyperparameter_settings = {
    'learning_rate' : 0.000005,
    'weight_decay' : 1e-4,
    'epochs' : 20,
    'in_shape' : 1,
    'out_shape' : 1,
    'hidden_shape' : 32,
    'flatten_factor' : 720
}

print("Loading data...")

# load data and cross validation folds
data_directory = RAW_PTH
kfolds_directory = KFOLD_DIR
data = pd.read_csv(data_directory, index_col=0)
kfolds = []
for i in range(NUM_FOLDS):
    train_index = []
    test_index = []
    with open(kfolds_directory + f"/fold{i}t.txt", "r") as tfile:
        for sample_name in tfile:
            train_index.append(sample_name.strip())
    with open(kfolds_directory + f"/fold{i}e.txt", "r") as efile:
        for sample_name in efile:
            test_index.append(sample_name.strip())
    kfolds.append((train_index, test_index))
dataframes = process_data_folds(data, kfolds, preprocessing_settings, feature_settings)

# augment data for cnn and wrap into dataloaders
dataloaders = []
for (X_train, X_test, y_train, y_test) in dataframes:
    (train_dataloader, test_dataloader) = create_dataloaders(X_train, X_test, y_train, y_test, shuffle=True, batch_size=32)
    dataloaders.append((train_dataloader, test_dataloader))

# setup output directory, class weights, and loss function, 
# aka criterion, for model training and evaluation
class_weights = torch.tensor([1]).to(device)
class_weights_dict = {
      0 : 1,
      1 : class_weights.item()
}
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device)

print("Evaluating model...")

cnn_results_list = []
for i in tqdm(range(30), ncols=50):
    cnn_results = run_cnn(
        data=dataloaders,
        criterion=criterion,
        device=device,
        learning_rate=hyperparameter_settings['learning_rate'],
        epochs=hyperparameter_settings['epochs'],
        in_shape=hyperparameter_settings['in_shape'],
        out_shape=hyperparameter_settings['out_shape'],
        hidden_shape=hyperparameter_settings['hidden_shape'],
        flatten_factor=hyperparameter_settings['flatten_factor']
    )
    cnn_results_list.append(cnn_results)
    print(cnn_results)
# print(cnn_results)
# print(hyperparameter_settings)
# cnn_results.to_csv(os.path.join(RESULTS_DIR, "cnn.csv"))
metrics = combine_several_weighted_averages(cnn_results_list)
metrics.to_csv(os.path.join(RESULTS_DIR, "cnn_30_metrics.csv"))
print(metrics)

print("Done.")