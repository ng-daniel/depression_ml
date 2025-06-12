print("--------------------------------")
print("DEPRESSION CLASSIFICATION // V.0")
print("--------------------------------")
print("PROCESS: LINEAR SVM TRAINING")
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
from data import apply_smote, preprocess_train_test_dataframes, create_feature_dataframe, create_long_feature_dataframe
from training_loops import run_linear_svc
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
LONG_FEATURE = False

preprocessing_grid = {
    'resample' : [False, True],
    'log_base' : [None, 10, 2],
    'scale_range' : [None, (0,1), (-1,1)]
}

preprocessing_settings = {
    'resample' : True,
    'log_base' : 10,
    'scale_range' : (0,1),
    'use_standard' : False,
    'use_gaussian' : None
}

print("Loading data...")

# load data and cross validation folds
data_directory = RAW_PTH
kfolds_directory = KFOLD_DIR
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
    if preprocessing_settings['resample']:
        # apply smote to training set
        X_train, y_train = apply_smote(X_train, y_train, 0.1)
    # preprocess data accordingly for the model
    (X_train, X_test) = preprocess_train_test_dataframes(
                            X_train=X_train,
                            X_test=X_test,
                            log_base=preprocessing_settings['log_base'],
                            scale_range=preprocessing_settings['scale_range'],
                            use_standard=preprocessing_settings['use_standard'],
                            use_gaussian=preprocessing_settings['use_gaussian']
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

# if GRID_SEARCH:
    
#     print("Performing gridsearch...")
#     # select a random training dataframe to optimize
#     index = random.randint(0, len(dataframes) - 1)
#     (X_train, _, y_train, _) = dataframes[index]
#     # gridsearch for optimal hyperparameters
#     gridsearch = RandomizedSearchCV(estimator=XGBClassifier(),
#                               param_distributions=hyperparameter_grid,
#                               n_iter=135,
#                               cv=2,
#                               verbose=True)
#     gridsearch.fit(X_train, y_train)
#     print(f"absolute cinema of params: {gridsearch.best_params_}")
#     # set settings to the optimal parameters
#     for param in gridsearch.best_params_:
#         hyperparameter_settings[param] = gridsearch.best_params_[param]

print("Evaluating model...")

linear_svm_results = run_linear_svc(
    data=dataframes,
    criterion=criterion,
    device=device,
    weights=class_weights_dict,
)
print(linear_svm_results)
linear_svm_results.to_csv(os.path.join(RESULTS_DIR, "linear_svm.csv"))
metrics = create_metrics_table([linear_svm_results])
print(metrics)

print("Done.")