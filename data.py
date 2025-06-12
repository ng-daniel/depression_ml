import os
import shutil
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.fft import rfft
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader

from util import log_skip_zeroes

CONDITION_SIZE = 23
CONTROL_SIZE = 32
DIR_PATH = "data/all"
scores = pd.read_csv("data/scores.csv", index_col='number')

class ActigraphDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return X, y

def _load_data_from_folder(dir_name: str, class_type: str, class_label: int, output_df: pd.DataFrame, scores_df: pd.DataFrame, start_time: str = None):
    '''
    Processes and groups all individual data files into a single dataframe, and then concatenates them to an existing dataframe.
    
    Args
    - dir_name - the directory to process the dataframe from
    - class_type - the class name of data to load ('control', 'condition')
    - class_label - the class number of data to load (0 / 1)
    - output_df - the dataframe to concatenate final compliation to
    - scores_df - dataframe containing the 'scores' which includes number of recorded days per subject
    - start_tile - arbitrary time selected to start counting data to ensure uniformity across subjects (ie. 12:00:00)
    '''

    data = pd.DataFrame()
    day_dfs = []
    
    dir_size = len(os.listdir(dir_name))
    for i in range(1, dir_size + 1):
        
        # read CSV into truncated dataframe
        
        filename = f"{class_type}_{i}"
        datapath = os.path.join(dir_name, filename + ".csv")
        file_df = pd.read_csv(datapath)

        # find all occurences of start time

        sr = file_df['timestamp'].map(lambda x: x.split()[1])
        indexes_of_time = []
        if start_time:
            indexes_of_time = list(sr[sr==start_time].index)
        else:
            indexes_of_time = [idx for idx in range(0, len(sr), 1440)]

        # split file_df into intervals of days

        num_days = scores_df.loc[filename, "days"]
        for j in range(num_days):
            interval = [indexes_of_time[j], indexes_of_time[j+1]]
            day_df = file_df.iloc[interval[0]:interval[1]]
            day_df = day_df['activity'].rename(f"{class_label}_{i}_{j}").reset_index(drop=True)
            
            # add data column to list of data columns (ignoring incomplete days)
            if len(day_df) == 1440:
                day_dfs.append(day_df)
    
    # concatenate all columns into single dataframe
    data = pd.concat(day_dfs, axis=1)
    return data

def load_dataframe_labels(dir_names: list, class_names: list, time: str = None):
    '''
    Aggregates all data into a single dataframe, starting from a uniform time value.
    
    Args:
    - dir_names - a list of the directories to search for data
    - class_names - a list of the relevant class name corresponding to each directory
    - time - arbitrary time selected to start counting data to ensure uniformity across subjects (ie. 12:00:00)

    Returns a dataframe containing all the data
    '''
    # load scores dataframe (information about each datafile)
    scores = pd.read_csv("data/scores.csv", index_col='number')
    
    # fill control and condition dataframes
    data = pd.DataFrame()
    dfs = []
    for CLASS in range(len(dir_names)):
        dfs.append(
            _load_data_from_folder(dir_names[CLASS], class_names[CLASS], 
                                    CLASS, data, scores, time)
        )
    
    # combine control and condition dfs
    data = pd.concat(dfs, axis=1)
    # transpose data so columns are time and rows are subjects
    data = data.transpose()
    # set labels
    labels = data.index.map(lambda x: int(x[0]))
    return data, labels

def kfolds_dataframes(data: pd.DataFrame, labels: list, numfolds: int, shuffle: bool, random_state: int = None):
    if shuffle:
        kf = StratifiedKFold(n_splits=numfolds, shuffle=shuffle, random_state=random_state)
    else:
        kf = StratifiedKFold(n_splits=numfolds, shuffle=shuffle)
    kf.get_n_splits(data, labels)

    folds = []
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        
        X_train = data.iloc[train_index]
        X_test = data.iloc[test_index]
        y_train = [labels[i] for i in train_index]
        y_test = [labels[i] for i in test_index]

        folds.append((X_train, X_test, y_train, y_test))
    
    return folds

def apply_smote(actigraph_data:pd.DataFrame, actigraph_labels:list, undersample_amount:float):
    # undersample control class by a set amount
    print(f"initial amount: {pd.Series(actigraph_labels).value_counts()}")
    resampled_data = actigraph_data.copy()
    if undersample_amount > 0:
        undersample_indices = random.sample(range(actigraph_labels.count(0)), round(actigraph_labels.count(0) * undersample_amount))
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

    return (resampled_data, list(resampled_labels))

def preprocess_train_test_dataframes(
        X_train: pd.DataFrame, X_test: pd.DataFrame = None, 
        log_base : int = None, 
        scale_range : tuple = None, 
        use_standard = False, 
        use_gaussian = None,
    ):
    '''
    Preprocesses
    '''
    # apply a 1d gaussian filter to each sample
    if use_gaussian:
        def gaussian_filter_apply(data:pd.Series):
            return pd.Series(gaussian_filter1d(data, sigma=use_gaussian))
        train_index = X_train.index
        X_train = X_train.apply(gaussian_filter_apply, axis=1)
        X_train.index = train_index
        if X_test is not None:
            test_index = X_test.index
            X_test = X_test.apply(gaussian_filter_apply, axis=1)
            X_test.index = test_index

    # apply log function to all values
    if log_base:
        X_train = X_train.map(lambda x: log_skip_zeroes(x, log_base))
        X_test = X_test.map(lambda x: log_skip_zeroes(x, log_base)) if X_test is not None else None
    
    # scale data to be within a specific range
    if scale_range or use_standard:
        if scale_range:
            scaler = MinMaxScaler(scale_range)
            if use_standard:
                print("preprocess_train_test_dataframes(): use_standard overriden by scale_range minmax scaler")
        elif use_standard:
            scaler = StandardScaler()

        train_index = X_train.index
        X_train = pd.DataFrame(scaler.fit_transform(X_train))
        X_train.index = train_index
        if X_test is not None:
            test_index = X_test.index
            X_test = pd.DataFrame(scaler.transform(X_test))
            X_test.index = test_index
    
    

    return (X_train, X_test)

def create_dataloaders(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: list, y_test: list,
                         shuffle: bool, batch_size: int):
    '''
    Wraps train / test dataframes of any data format into a dataloader
    '''
    # convert data to tensors
    if (type(X_train) == pd.DataFrame):
        X_train = torch.from_numpy(X_train.to_numpy()).float().unsqueeze(dim=1)
        X_test = torch.from_numpy(X_test.to_numpy()).float().unsqueeze(dim=1)    
        y_train = torch.tensor(y_train).float()
        y_test = torch.tensor(y_test).float()

    # wrap in pytorch dataloader
    train_dataset = ActigraphDataset(X_train, y_train)
    test_dataset = ActigraphDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return (train_dataloader, test_dataloader)

def extract_stats_from_window(data: pd.Series, include_quarter_diff = False):
    '''
    Extracts stats from a window of actigraph data (total of ):
        - Sample Mean, Sample Std. Dev., Sample Skewness, Sample Kurtosis, Max, Min (6)
        - 1/2 win. change in means, max, min (3)
        - 1/4 win. means, Std. Dev.s, maxes, mins (16)
        - 1/4 win. means paired differences, maxes paired differences, mins paired differences

    Args
        - data: the actigraphy data series window to extract features from
    
    Returns
        - a dataframe containing a single row, where the columns are the extracted features 
    '''     
    data_np = data.to_numpy()
    
    stats = []
    labels = []

    # loading descriptive statistics
    labels.extend(['mean', 'std', 'max', 'min'])
    stats.extend([data_np.mean(), data_np.std(), data_np.max(), data_np.min()])
    
    # calculating half window statistics
    h_win = [data_np[0:len(data_np)//2], data_np[len(data_np)//2:]]
    labels.extend(['h_mean_change', 'h_max_change', 'h_min_change'])
    stats.extend([h_win[1].mean() - h_win[0].mean(), h_win[1].max() - h_win[0].max(), h_win[1].min() - h_win[0].min()])
    
    # calculating quarter window statistics
    if include_quarter_diff:
        q_win = []
        for i in range(0, len(data_np), len(data_np)//4):
            win = data_np[i:i+len(data_np)//4]
            q_win.append(win)
        
        q_means = [win.mean() for win in q_win]
        q_stds = [win.std() for win in q_win]
        q_maxs = [win.max() for win in q_win]
        q_mins = [win.min() for win in q_win]

        labels.extend([f'q{i}_mean' for i in range(1,len(q_win)+1)])
        stats.extend(q_means)
        
        labels.extend([f'q{i}_std' for i in range(1,len(q_win)+1)])
        stats.extend(q_stds)
        
        labels.extend([f'q{i}_max' for i in range(1,len(q_win)+1)])
        stats.extend(q_maxs)
        
        labels.extend([f'q{i}_min' for i in range(1,len(q_win)+1)])
        stats.extend(q_mins)

    # calculate difference in quarter statistics
    if include_quarter_diff:
        for i in range(len(q_means)):
            for j in range(i+1, len(q_means)):
                labels.append(f'q{i}_minus_q{j}_mean')
                stats.append(q_means[i] - q_means[j])

    # convert stats and labels into series
    features = pd.Series(stats, index = labels)
    return features

def create_feature_dataframe(data: pd.DataFrame, include_quarter_diff = False):
    extracted_stats = data.apply(extract_stats_from_window, axis=1, args=(include_quarter_diff,)).reset_index(drop=True)
    extracted_stats.index = data.index
    return extracted_stats

def create_long_feature_dataframe(data: pd.DataFrame, window_size = 30, include_quarter_diff = False):
    def extract_long_feature_series(x: pd.Series):
        return extract_feature_series(x, window_size, include_quarter_diff).stack()
    extracted_stats_long = data.apply(extract_long_feature_series, axis=1)
    extracted_stats_long.columns = ['_'.join(str(val) for val in col).strip() for col in extracted_stats_long.columns.values]
    print(extracted_stats_long)
    return extracted_stats_long

def extract_feature_series(data: pd.Series, window_size = 30, include_quarter_diff = False):
    '''
    Transforms a series of raw actigraphy data into a dataframe
    containing feature data across 24 hours using a sliding window
    approach for increasing increments of 30 minutes.
    '''
    features_by_window = []
    for i in range(0, len(data), window_size):
        window = data[i:i+window_size]
        features_by_window.append(extract_stats_from_window(window, include_quarter_diff))
    return pd.concat(features_by_window, axis=1).transpose()

def empty_dataframe_directory(dir_name: str):
    '''
    Clears a data directory, treating "processed_dataframes" as the root path.
    '''
    assert(len(dir_name) > 0)

    # remove entire directory and replace it with an empty one of the same name
    
    directory = os.path.join("data/processed_dataframes", dir_name)
    shutil.rmtree(directory)
    os.mkdir(directory)

def reset_feature_series(num_folds: int):
    '''
    Clears and resets the feature series data folder
    with the appropriate number of folds.
    '''
    empty_dataframe_directory("feature_series")
    directory = "data/processed_dataframes/feature_series"
    for i in range(num_folds):
        fold_dir = os.path.join(directory, str(i))
        os.mkdir(fold_dir)
        os.mkdir(os.path.join(fold_dir, "train"))
        os.mkdir(os.path.join(fold_dir, "test"))

def export_kfolds_split_indices(data: pd.DataFrame, labels: list, export_dir: str, n_splits: int, shuffle: bool, random_state: int = None):
    '''
    Splits data into N folds, writing each fold to a directory as a text file.
    '''

    # extracts all individual subject labels from data index
    sample_names = list(data.index)
    subject_names = list(dict.fromkeys(['_'.join(name.split(sep='_')[0:2]) for name in sample_names]))
    subject_labels = [int(name[0]) for name in subject_names]

    print(subject_names)
    print(len(subject_names))
    print(subject_labels)
    print(len(subject_labels))

    print(subject_names)

    kf = StratifiedKFold(n_splits=n_splits, 
               shuffle=shuffle, 
               random_state=random_state)

    for i, (train_index, test_index) in enumerate(kf.split(subject_names, subject_labels)):
        
        train_subjects = [name for i, name in enumerate(subject_names) if i in train_index]
        train_names = []
        for subject_name in train_subjects:
            train_names += [name for name in sample_names if (subject_name + '_') in name]
        
        test_subjects = [name for i, name in enumerate(subject_names) if i in test_index]
        test_names = []
        for subject_name in test_subjects:
            test_names += [name for name in sample_names if (subject_name + '_') in name]

        train_filename = f"fold{i}t.txt"
        test_filename = f"fold{i}e.txt"   

        # write names to files in the kfolds folder
        with open(os.path.join(export_dir, train_filename), "w") as file:
            for name in train_names:
                file.write(name + "\n")
        with open(os.path.join(export_dir, test_filename), "w") as file:
            for name in test_names:
                file.write(name + "\n")

def load_kf_actigraphy_dfs():
    pass

