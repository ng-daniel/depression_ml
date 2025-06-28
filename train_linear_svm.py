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
from core.data import process_data_folds
from core.training_loops import run_linear_svc
from core.eval import create_metrics_table

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
    'subtract_mean' : True,
    'adjust_seasonality' : True
}
feature_settings = {
    'use_feature' : True,
    'long_feature' : False,
    'window_size' : 30,
    'quarter_diff' : False,
    'simple' : True
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