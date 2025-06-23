print("--------------------------------")
print("DEPRESSION CLASSIFICATION // V.0")
print("--------------------------------")
print("PROCESS: XGBOOST TRAINING")
print("Importing libraries...")

import os
from tqdm import tqdm
import random
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from core.data import process_data_folds
from core.training_loops import run_XGBoost
from core.eval import create_metrics_table

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_DIR = "data/processed_dataframes/"
RAW_PTH = DATA_DIR + "/data_raw.csv"
SMOTE_PTH = DATA_DIR + "/data_resampled.csv"
KFOLD_DIR = DATA_DIR + "/kfolds"
KFOLD_SMOTE_DIR = DATA_DIR + "/kfolds_smote"
NUM_FOLDS = len(os.listdir(KFOLD_DIR))//2
RESULTS_DIR = "results"

hyperparameter_grid = {
    'min_child_weight' : [1, 5, 10],
    'subsample' : [0.6, 0.8, 1],
    'colsample_bytree' : [0.6, 0.8, 1],
}
use_grid_search = True

preprocessing_settings = {
    'resample' : True,
    'log_base' : 10,
    'scale_range' : (-1,1),
    'use_standard' : False,
    'use_gaussian' : 100
}
feature_settings = {
    'long_feature' : True,
    'window_size' : 30,
    'quarter_diff' : False,
    'simple' : True
}
hyperparameter_settings = {
    'max_depth' : 6,
    'min_child_weight' : 1,
    'subsample' : 1,
    'colsample_bytree' : 1,
    'learning_rate' : 0.1
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

# setup output directory, class weights, and loss function, 
# aka criterion, for model training and evaluation
class_weights = torch.tensor([1]).to(device)
class_weights_dict = {
      0 : 1,
      1 : class_weights.item()
}
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device)

if use_grid_search:
    
    print("Performing gridsearch...")
    # select a random training dataframe to optimize
    index = random.randint(0, len(dataframes) - 1)
    (X_train, _, y_train, _) = dataframes[index]
    # gridsearch for optimal hyperparameters
    gridsearch = RandomizedSearchCV(estimator=XGBClassifier(),
                              param_distributions=hyperparameter_grid,
                              n_iter=135,
                              cv=2,
                              verbose=True)
    gridsearch.fit(X_train, y_train)
    print(f"absolute cinema of params: {gridsearch.best_params_}")
    # set settings to the optimal parameters
    for param in gridsearch.best_params_:
        hyperparameter_settings[param] = gridsearch.best_params_[param]

print("Evaluating model...")

xgb_results = run_XGBoost(
    data=dataframes,
    learning_rate=hyperparameter_settings['learning_rate'],
    criterion=criterion,
    device=device,
    weights=class_weights_dict,
    max_depth=hyperparameter_settings['max_depth'],
    min_child_weight = hyperparameter_settings['min_child_weight'],
    subsample = hyperparameter_settings['subsample'],
    colsample_bytree = hyperparameter_settings['colsample_bytree']
)
print(xgb_results)
print(hyperparameter_settings)
xgb_results.to_csv(os.path.join(RESULTS_DIR, "xgb.csv"))
metrics = create_metrics_table([xgb_results])
print(metrics)

print("Done.")