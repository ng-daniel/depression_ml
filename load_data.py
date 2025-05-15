print("--------------------------------")
print("DEPRESSION CLASSIFICATION // V.0")
print("--------------------------------")
print("PROCESS: DATA LOADING")
print("Loading libraries...")

from pathlib import Path
import pandas as pd
from tqdm import tqdm
from data import (load_dataframe_labels, preprocess_train_test_dataframes, create_feature_dataframe, 
                  kfolds_dataframes, reset_feature_series, load_feature_series_data)
from imblearn.over_sampling import SMOTE

EXPORT_DIR = Path("data/processed_dataframes")

print("Loading raw dataframe...")

# load dataframe
actigraph_data, actigraph_labels = load_dataframe_labels(dir_names = ["data/control", "data/condition"],
                                                                    class_names = ["control", "condition"],
                                                                    time = "12:00:00", undersample=False)
print("Oversampling minority class...")

# # oversample the condition class to match the control class
# oversample = SMOTE(sampling_strategy='minority')
# resampled_data, resampled_labels = oversample.fit_resample(actigraph_data, actigraph_labels)

# # create new index names for the generated samples
# new_data_count = len(resampled_labels) - len(actigraph_labels)
# new_index = [f'1_N_{i}' for i in range(new_data_count)]
# resampled_data.index = list(actigraph_data.index) + new_index

# actigraph_data = resampled_data
# actigraph_labels = resampled_labels

print("Loading folds and extracting features...")

NUM_FOLDS = 5
kf_dfs = kfolds_dataframes(actigraph_data, actigraph_labels, numfolds=NUM_FOLDS, shuffle=True, batch_size=32, random_state=42)
kf_actigraphy_dfs = []
kf_feature_dfs = []
for i in tqdm(range(NUM_FOLDS), ncols=50, leave=True):
    (X_train, X_test, y_train, y_test) = kf_dfs[i]

    # load actigraphy folds
    (X_train_p, X_test_p) = preprocess_train_test_dataframes(X_train, X_test)
    kf_actigraphy_dfs.append((X_train_p, X_test_p, y_train, y_test))

    # load feature folds
    X_train_features = create_feature_dataframe(data = X_train_p, raw_data = X_train)
    X_test_features = create_feature_dataframe(data = X_test_p, raw_data = X_test)
    kf_feature_dfs.append((X_train_features, X_test_features, y_train, y_test))

print(f"Loading feature series csv files and sample names...")

# feature series data
EXPORT_DIR_FEATURE_SERIES = Path("data/processed_dataframes/feature_series")
load_feature_series_data(actigraph_data, EXPORT_DIR_FEATURE_SERIES.joinpath("csv_files"))
for i, (X_train, X_test, y_train, y_test) in enumerate(tqdm(kf_actigraphy_dfs, ncols=50, leave=True)):
    
    # extract sample names from compacted dataframe
    train_names = list(X_train.index)
    test_names = list(X_test.index)
    train_filename = f"fs{i}t.txt"
    test_filename = f"fs{i}e.txt"

    # write names to files in the kfolds folder
    with open(EXPORT_DIR_FEATURE_SERIES.joinpath("kfolds", train_filename), "w") as file:
        for name in train_names:
            file.write(name + "\n")
    with open(EXPORT_DIR_FEATURE_SERIES.joinpath("kfolds", test_filename), "w") as file:
        for name in test_names:
            file.write(name + "\n")

print(f"Writing csv files to: {EXPORT_DIR}...")

# actigraphy data
EXPORT_DIR_ACTIGRAPHY = Path("data/processed_dataframes/actigraphy")
for i, (X_train, X_test, y_train, y_test) in enumerate(kf_actigraphy_dfs):
    X_train['label'] = y_train
    X_test['label'] = y_test
    X_train.to_csv(EXPORT_DIR_ACTIGRAPHY.joinpath(f"a{i}t.csv"), index=True)
    X_test.to_csv(EXPORT_DIR_ACTIGRAPHY.joinpath(f"a{i}e.csv"), index=True)

# extracted feature data
EXPORT_DIR_FEATURE = Path("data/processed_dataframes/feature")
for i, (X_train, X_test, y_train, y_test) in enumerate(kf_feature_dfs):
    X_train['label'] = y_train
    X_test['label'] = y_test
    X_train.to_csv(EXPORT_DIR_FEATURE.joinpath(f"f{i}t.csv"), index=True)
    X_test.to_csv(EXPORT_DIR_FEATURE.joinpath(f"f{i}e.csv"), index=True)

print("Done.")