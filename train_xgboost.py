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
from data import preprocess_train_test_dataframes, create_feature_dataframe, create_long_feature_dataframe
from training_loops import run_XGBoost
from eval import create_metrics_table

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_DIR = "data/processed_dataframes/"
RAW_PTH = DATA_DIR + "/data_raw.csv"
SMOTE_PTH = DATA_DIR + "/data_resampled.csv"
KFOLD_DIR = DATA_DIR + "/kfolds"
KFOLD_SMOTE_DIR = DATA_DIR + "/kfolds_smote"
NUM_FOLDS = len(os.listdir(KFOLD_DIR))//2
RESULTS_DIR = "results"
GRID_SEARCH = True
LONG_FEATURE = True

preprocessing_grid = {
    'resample' : [False, True],
    'log_base' : [None, 10, 2],
    'scale_range' : [None, (0,1), (-1,1)]
}
hyperparameter_grid = {
    'min_child_weight' : [1, 5, 10],
    'subsample' : [0.6, 0.8, 1],
    'colsample_bytree' : [0.6, 0.8, 1],
}

preprocessing_settings = {
    'resample' : True,
    'log_base' : 10,
    'scale_range' : None
}
hyperparameter_settings = {
    'max_depth' : 6,
    'min_child_weight' : 1,
    'subsample' : 1,
    'colsample_bytree' : 1,
    'learning_rate' : 0.3
}

print("Loading data...")

# load data and cross validation folds
data_directory = SMOTE_PTH if preprocessing_settings['resample'] else RAW_PTH
kfolds_directory = KFOLD_SMOTE_DIR if preprocessing_settings['resample'] else KFOLD_DIR
data = pd.read_csv(data_directory, index_col=0)
dataframes = []
for i in tqdm(range(NUM_FOLDS),ncols=50):
    # extract train/test index names
    train_index = []
    test_index = []
    with open(kfolds_directory + f"/fold{i}t.txt", "r") as tfile:
        for sample_name in tfile:
            train_index.append(sample_name.strip())
    with open(kfolds_directory + f"/fold{i}e.txt", "r") as efile:
        for sample_name in efile:
            test_index.append(sample_name.strip())
    # split data based on the extracted train/test split
    X = data.copy()
    X_train = X.drop(labels=test_index, axis=0)
    X_test = X.drop(labels=train_index, axis=0)
    y_train = list(X_train.pop('label'))
    y_test = list(X_test.pop('label'))
    # preprocess data accordingly for the model
    (X_train, X_test) = preprocess_train_test_dataframes(
                            X_train=X_train,
                            X_test=X_test,
                            log_base=preprocessing_settings['log_base'],
                            scale_range=preprocessing_settings['scale_range']
                        )
    # extract features
    if LONG_FEATURE:
        X_train = create_long_feature_dataframe(X_train, window_size=60, include_quarter_diff=False)
        X_test = create_long_feature_dataframe(X_test, window_size=60, include_quarter_diff=False)
        print(X_train)
    else:
        X_train = create_feature_dataframe(X_train, True)
        X_test = create_feature_dataframe(X_test, True)
    dataframes.append((X_train, X_test, y_train, y_test))

# setup output directory, class weights, and loss function, 
# aka criterion, for model training and evaluation
class_weights = torch.tensor([1]).to(device)
class_weights_dict = {
      0 : 1,
      1 : class_weights.item()
}
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device)

if GRID_SEARCH:
    
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