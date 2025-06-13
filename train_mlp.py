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

from data import apply_smote, preprocess_train_test_dataframes, create_dataloaders, create_feature_dataframe, create_long_feature_dataframe
from training_loops import run_mlp
from eval import create_metrics_table

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_DIR = "data/processed_dataframes/"
RAW_PTH = DATA_DIR + "/data_raw.csv"
SMOTE_PTH = DATA_DIR + "/data_resampled.csv"
KFOLD_DIR = DATA_DIR + "/kfolds"
KFOLD_SMOTE_DIR = DATA_DIR + "/kfolds_smote"
NUM_FOLDS = len(os.listdir(KFOLD_DIR))//2
RESULTS_DIR = "results"
LONG_FEATURE = False

preprocessing_grid = {
    'resample' : [False, True],
    'log_base' : [None, 10, 2],
    'scale_range' : [None, (0,1), (-1,1)]
}

preprocessing_settings = {
    'resample' : False,
    'log_base' : None,
    'scale_range' : None,
    'use_standard' : False,
    'use_gaussian' : None,
    'window_size' : 30,
    'batch_size' : 32
}
hyperparameter_settings = {
    'learning_rate' : 0.000005,
    'weight_decay' : 1e-4,
    'epochs' : 10,
    'in_shape' : 1,
    'out_shape' : 1,
    'hidden_shape' : 16,
    'flatten_factor' : 720
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
        X_train = create_long_feature_dataframe(X_train, 
                                                window_size=preprocessing_settings['window_size'],
                                                include_quarter_diff=False)
        X_test = create_long_feature_dataframe(X_test, 
                                               window_size=preprocessing_settings['window_size'],
                                               include_quarter_diff=False)
    else:
        X_train = create_feature_dataframe(X_train, False)
        X_test = create_feature_dataframe(X_test, False)
    dataframes.append((X_train, X_test, y_train, y_test))

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

mlp_results_list = []
for i in tqdm(range(20), ncols=50):
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
# print(mlp_results)
# print(hyperparameter_settings)
# mlp_results.to_csv(os.path.join(RESULTS_DIR, "mlp.csv"))
metrics = create_metrics_table(mlp_results_list)
metrics.to_csv(os.path.join(RESULTS_DIR, "mlp_20_runs_metrics.csv"))
print(metrics)
print(metrics['acc'].mean())
print(metrics['acc'].std())
print(metrics['f1sc'].mean())
print(metrics['f1sc'].std())

print("Done.")