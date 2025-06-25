print("--------------------------------")
print("DEPRESSION CLASSIFICATION // V.2")
print("--------------------------------")
print("PROCESS: RANDOM FOREST TRAINING")
print("Importing libraries...")

import os
import random
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from core.data import create_dataloaders, process_data_folds
from core.training_loops import run_random_forest
from core.eval import create_metrics_table

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_DIR = "data/processed_dataframes/"
RAW_PTH = DATA_DIR + "/data_raw.csv"
SMOTE_PTH = DATA_DIR + "/data_resampled.csv"
KFOLD_DIR = DATA_DIR + "/kfolds"
KFOLD_SMOTE_DIR = DATA_DIR + "/kfolds_smote"
NUM_FOLDS = len(os.listdir(KFOLD_DIR))//2
RESULTS_DIR = "results"
LONG_FEATURE = True

hyperparameter_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}
use_grid_search = False

preprocessing_settings = {
    'resample' : False,
    'log_base' : None,
    'scale_range' : None,
    'use_standard' : True,
    'use_gaussian' : None,
    'subtract_mean' : False,
    'adjust_seasonality' : False,
    'window_size' : 30
}
feature_settings = {
    'use_feature' : True,
    'long_feature' : True,
    'window_size' : 60,
    'quarter_diff' : False,
    'simple' : True
}
hyperparameter_settings = {
    'n_estimators': 100,
    'max_depth': None,
    'max_features': 'sqrt',
    'min_samples_split': 2,
    'min_samples_leaf': 2,
    'bootstrap': False
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
    # CHANGE TO RANDOM SEARCH CV
    print("Performing gridsearch...")
    
    # select a random training dataframe to optimize
    index = random.randint(0, len(dataframes) - 1)
    (X_train, _, y_train, _) = dataframes[index]
    # gridsearch for optimal hyperparameters
    gridsearch = RandomizedSearchCV(estimator=RandomForestClassifier(),
                              param_distributions=hyperparameter_grid,
                              cv=5,
                              verbose=True)
    gridsearch.fit(X_train, y_train)
    print(f"absolute cinema of params: {gridsearch.best_params_}")
    # set settings to the optimal parameters
    for param in gridsearch.best_params_:
        hyperparameter_settings[param] = gridsearch.best_params_[param]

print("Evaluating model...")

forest_results = run_random_forest(
    data=dataframes,
    criterion=criterion,
    device=device,
    weights=class_weights_dict,
    n_estimators=hyperparameter_settings['n_estimators'],
    max_depth=hyperparameter_settings['max_depth'],
    max_features=hyperparameter_settings['max_features'],
    min_samples_split=hyperparameter_settings['min_samples_split'],
    min_samples_leaf=hyperparameter_settings['min_samples_leaf'],
    bootstrap=hyperparameter_settings['bootstrap']
)
print(forest_results)
print(hyperparameter_settings)
forest_results.to_csv(os.path.join(RESULTS_DIR, "random_forest.csv"))
metrics = create_metrics_table([forest_results])
print(metrics)

print("Done.")