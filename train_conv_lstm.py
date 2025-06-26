print("--------------------------------")
print("DEPRESSION CLASSIFICATION // V.0")
print("--------------------------------")
print("PROCESS: CONV LSTM TRAINING")
print("Importing libraries...")

import os
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn

from core.data import create_dataloaders, process_data_folds
from core.training_loops import run_conv_lstm
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
    'use_gaussian' : 50,
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
    'learning_rate' : 0.00005,
    'weight_decay' : 1e-4,
    'epochs' : 75,
    'out_shape' : 1,
    'hidden_shape' : 64,
    'lstm_layers' : 8
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

# augment data for LSTM v2 and wrap into dataloaders
dataloaders = []
for (X_train, X_test, y_train, y_test) in dataframes:
    (train_dataloader, test_dataloader) = create_dataloaders(X_train, X_test, y_train, y_test, shuffle=True, batch_size=32)
    dataloaders.append((train_dataloader, test_dataloader))

# setup output directory, class weights, and loss function, 
# aka criterion, for model training and evaluation
class_weights = torch.tensor([1.2]).to(device)
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
conv_lstm_results_list = []
for i in tqdm(range(20), ncols=50):
    conv_lstm_results = run_conv_lstm(
        data=dataloaders,
        criterion=criterion,
        device=device,
        learning_rate=hyperparameter_settings['learning_rate'],
        weight_decay=hyperparameter_settings['weight_decay'],
        epochs=hyperparameter_settings['epochs'],
        in_shape=NUM_FEATURES,
        out_shape=hyperparameter_settings['out_shape'],
        hidden_shape=hyperparameter_settings['hidden_shape'],
        lstm_layers=hyperparameter_settings['lstm_layers'],
    )
    conv_lstm_results_list.append(conv_lstm_results)
    print(conv_lstm_results)
metrics = combine_several_weighted_averages(conv_lstm_results_list)
metrics.to_csv(os.path.join(RESULTS_DIR, "conv_lstm_20_metrics.csv"))
print(metrics)

print("Done.")