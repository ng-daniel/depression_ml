print("--------------------------------------")
print("DEPRESSION CLASSIFICATION MODEL // V.0")
print("--------------------------------------")
print("PROCESS: DATA LOADING")
print("Loading libraries...")

from pathlib import Path
import pandas as pd
from tqdm import tqdm
from data import load_dataframe_labels, preprocess_train_test_dataframes, create_feature_dataframe, kfolds_dataframes

EXPORT_DIR = Path("data/processed_dataframes")

print("Loading raw dataframe...")

# load dataframe
actigraph_data, actigraph_labels = load_dataframe_labels(dir_names = ["data/control", "data/condition"],
                                                                    class_names = ["control", "condition"],
                                                                    time = "12:00:00", undersample=False)
print("Loading folds and extracting features...")

NUM_FOLDS = 10
kf_dfs = kfolds_dataframes(actigraph_data, actigraph_labels, numfolds=NUM_FOLDS, shuffle=True, random_state=42, batch_size=32)

kf_actigraphy_dfs = []
kf_feature_dfs = []
for i in tqdm(range(NUM_FOLDS), ncols=50, leave=False):
    (X_train, X_test, y_train, y_test) = kf_dfs[i]

    # load actigraphy folds

    (X_train_p, X_test_p) = preprocess_train_test_dataframes(X_train, X_test)
    kf_actigraphy_dfs.append((X_train_p, X_test_p, y_train, y_test))

    # load feature folds

    X_train_features = create_feature_dataframe(data = X_train_p, raw_data = X_train)
    X_test_features = create_feature_dataframe(data = X_test_p, raw_data = X_test)
    kf_feature_dfs.append((X_train_features, X_test_features, y_train, y_test))

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