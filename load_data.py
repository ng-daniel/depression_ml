print("--------------------------------")
print("DEPRESSION CLASSIFICATION // V.0")
print("--------------------------------")
print("PROCESS: DATA LOADING")

import os
import pandas as pd
import random
from tqdm import tqdm
from data import (load_dataframe_labels, export_kfolds_split_indices)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold

EXPORT_DIR = "data/processed_dataframes"
DIR_NAMES = ["data/control", "data/condition"]
CLASS_NAMES = ["control", "condition"]
START_TIME = "12:00:00"
NUM_FOLDS = 10
SHUFFLE = True
RANDOM_STATE = 42

# load raw dataset into dataframes and labels
actigraph_data, actigraph_labels = load_dataframe_labels(dir_names = DIR_NAMES,
                                                         class_names = CLASS_NAMES,
                                                         time = START_TIME)
actigraph_labels = list(actigraph_labels)
# export raw dataset to csv
actigraph_csv = actigraph_data.copy()
actigraph_csv['label'] = actigraph_labels
actigraph_csv.to_csv(os.path.join(EXPORT_DIR, 'data_raw.csv'))

# undersample control class by a set amount
print(f"initial amount: {pd.Series(actigraph_labels).value_counts()}")
undersample_amount = 0.10
undersample_indices = random.sample(range(actigraph_labels.count(0)), round(actigraph_labels.count(0) * undersample_amount))
resampled_data = actigraph_data.copy()
resampled_data = resampled_data.drop(list(actigraph_data.index[undersample_indices]), axis=0)
resampled_labels = [actigraph_labels[i] for i in range(len(actigraph_labels)) if i not in undersample_indices]
print(f"undersampled amount: {pd.Series(resampled_labels).value_counts()}")

# oversample the condition class to match the control class
oversample = SMOTE(sampling_strategy='minority')
resampled_index = list(resampled_data.index)
resampled_data, resampled_labels = oversample.fit_resample(resampled_data, resampled_labels)
# create new index names for the generated samples
new_data_count = len(resampled_labels) - len(resampled_index)
new_index = [f'1_N_{i}' for i in range(new_data_count)]
resampled_data.index = resampled_index + new_index
print(f"SMOTE amount: {pd.Series(resampled_labels).value_counts()}")
# export resampled data to csv
resampled_csv = resampled_data.copy()
resampled_csv['label'] = list(resampled_labels)
resampled_csv.to_csv(os.path.join(EXPORT_DIR, 'data_resampled.csv'))

# write cross validation train/test split indices to txt files
export_kfolds_split_indices(data=actigraph_data,
                            labels=actigraph_labels,
                            export_dir=os.path.join(EXPORT_DIR, "kfolds"),
                            n_splits=NUM_FOLDS,
                            shuffle=SHUFFLE,
                            random_state=RANDOM_STATE)
export_kfolds_split_indices(data=resampled_data,
                            labels=resampled_labels,
                            export_dir=os.path.join(EXPORT_DIR, "kfolds_smote"),
                            n_splits=NUM_FOLDS,
                            shuffle=SHUFFLE,
                            random_state=RANDOM_STATE)

print("Done.")