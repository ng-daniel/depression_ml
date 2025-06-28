print("--------------------------------")
print("DEPRESSION CLASSIFICATION // V.0")
print("--------------------------------")
print("PROCESS: MLP TRAINING")
print("Importing libraries...")

import os
from tqdm import tqdm
import random
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

from core.data import create_dataloaders, process_data_folds
from core.training_loops import run_mlp
from core.eval import combine_several_weighted_averages

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_DIR = "data/processed_dataframes/"
RAW_PTH = DATA_DIR + "/data_raw.csv"
SMOTE_PTH = DATA_DIR + "/data_resampled.csv"
KFOLD_DIR = DATA_DIR + "/kfolds"
KFOLD_SMOTE_DIR = DATA_DIR + "/kfolds_smote"
NUM_FOLDS = len(os.listdir(KFOLD_DIR))//2
RESULTS_DIR = "results"
LONG_FEATURE = False

preprocessing_settings = {
    'resample' : True,
    'log_base' : None,
    'scale_range' : None,
    'use_standard' : True,
    'use_gaussian' : 30,
    'adjust_seasonality' : True,
}
feature_settings = {
    'use_feature' : True,
    'long_feature' : True,
    'window_size' : 60,
    'quarter_diff' : False,
    'simple' : True
}
hyperparameter_settings = {
    'learning_rate' : 0.0005,
    'weight_decay' : 1e-4,
    'epochs' : 10,
    'in_shape' : 1,
    'out_shape' : 1,
    'hidden_shape' : 32
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

# augment data for mlp and wrap into dataloaders
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
NUM_FEATURES = len(dataframes[0][0].columns)

print("Evaluating model...")

mlp_results_list = []
for i in tqdm(range(30), ncols=50):
    mlp_results = run_mlp(
        data=dataloaders,
        criterion=criterion,
        device=device,
        learning_rate=hyperparameter_settings['learning_rate'],
        epochs=hyperparameter_settings['epochs'],
        in_shape=NUM_FEATURES,
        out_shape=hyperparameter_settings['out_shape'],
        hidden_shape=hyperparameter_settings['hidden_shape'],
    )
    mlp_results_list.append(mlp_results)
    print(mlp_results)
# print(mlp_results)
# print(hyperparameter_settings)
# mlp_results.to_csv(os.path.join(RESULTS_DIR, "mlp.csv"))
metrics = combine_several_weighted_averages(mlp_results_list)
metrics.to_csv(os.path.join(RESULTS_DIR, "mlp_30_metrics.csv"))
print(metrics)

print("Done.")