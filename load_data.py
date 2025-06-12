print("--------------------------------")
print("DEPRESSION CLASSIFICATION // V.0")
print("--------------------------------")
print("PROCESS: DATA LOADING")

import os
import pandas as pd
import random
from data import (load_dataframe_labels, export_kfolds_split_indices)

EXPORT_DIR = "data/processed_dataframes"
DIR_NAMES = ["data/control", "data/condition"]
CLASS_NAMES = ["control", "condition"]
START_TIME = "12:00:00"
NUM_FOLDS = 5
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

# write cross validation train/test split indices to txt files
export_kfolds_split_indices(data=actigraph_data,
                            labels=actigraph_labels,
                            export_dir=os.path.join(EXPORT_DIR, "kfolds"),
                            n_splits=NUM_FOLDS,
                            shuffle=SHUFFLE,
                            random_state=RANDOM_STATE)

print("Done.")