import os
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader

from core.util import log_skip_zeroes, data_mean_med_std, subtract_corresponding_minute

CONDITION_SIZE = 23
CONTROL_SIZE = 32
DIR_PATH = "data/all"
scores = pd.read_csv("data/scores.csv", index_col='number')

class ActigraphDataset(Dataset):
    '''
    Pytorch dataset class to format data for batched dataloaders
    '''
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return X, y

def create_dataloaders(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: list, y_test: list,
                         shuffle: bool, batch_size: int):
    '''
    Wraps train / test dataframes of any data format into a dataloader

    Args
    - X_train - train dataframe 
    - X_test - test dataframe
    - y_train - train labels
    - y_test - test labels
    - shuffle - shuffle dataset when creating dataloaders
    - batch_size - batch size of dataloaders

    Returns
    - (DataLoader, DataLoader) tuple of train and test dataloaders
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


'''

Data loading and exporting to and from directories.

'''


def _load_data_from_folder(dir_name: str, class_type: str, class_label: int, scores_df: pd.DataFrame, start_time: str = None):
    '''
    Processes and groups all individual data files into a single dataframe, and then concatenates them to an existing dataframe.
    
    Args
    - dir_name - the directory to process the dataframe from
    - class_type - the class name of data to load ('control', 'condition')
    - class_label - the class number of data to load (0 / 1)
    - scores_df - dataframe containing the 'scores' which includes number of recorded days per subject
    - start_tile - arbitrary time selected to start counting data to ensure uniformity across subjects (ie. 12:00:00)

    Returns
        dataframe containing all data with labels
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
    
    Args
    - dir_names - a list of the directories to search for data
    - class_names - a list of the relevant class name corresponding to each directory
    - time - arbitrary time selected to start counting data to ensure uniformity across subjects (ie. 12:00:00)

    Returns 
    - a dataframe containing all the data
    '''
    # load scores dataframe (information about each datafile)
    scores = pd.read_csv("data/scores.csv", index_col='number')
    
    # fill control and condition dataframes
    data = pd.DataFrame()
    dfs = []
    for CLASS in range(len(dir_names)):
        dfs.append(_load_data_from_folder(dir_name=dir_names[CLASS], 
                                          class_type=class_names[CLASS], 
                                          class_label=CLASS,
                                          scores_df=scores, 
                                          start_time=time))
    
    # combine control and condition dfs
    data = pd.concat(dfs, axis=1)
    # transpose data so columns are time and rows are subjects
    data = data.transpose()
    # set labels
    labels = data.index.map(lambda x: int(x[0]))
    return data, labels

def export_kfolds_split_indices(data: pd.DataFrame, labels: list, export_dir: str, n_splits: int, shuffle: bool, random_state: int = None):
    '''
    Splits the data by subject into k folds while preserving the relative proportions of each class.
    
    Writes these splits to a specified directory to ensure different models are using the same split.

    Args
    - data - a dataframe of all the data from the dataset
    - export_dir - the directory to export the folds to
    - n_splits - number of splits to use
    - shuffle - whether to randomize the splits
    - random_state - random seed for the splits

    Returns
    - nothing
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
        
        print(train_index)
        print(test_index)

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


'''

Preprocessing functions.

'''


def _apply_smote(actigraph_data:pd.DataFrame, actigraph_labels:list, undersample_amount:float):
    '''
    Resamples the data using synthetic minority oversampling technique (smote)
    
    Args
    - actigraph_data - input dataframe
    - actigraph_labels - input labels
    - undersample amount - proportion of majority class to remove before generating samples for the minority class

    Returns
    - a tuple (pd.Dataframe, list) containing the resampled data and the respective labels
    '''
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
        settings : dict = None,
        log_base : int = None, 
        scale_range : tuple = None, 
        use_standard = False, 
        use_gaussian = None,
        subtract_mean = False,
        adjust_seasonality = False,
    ):
    '''
    Applies various preprocessing techniques to the data.

    Args
    - X_train - train split of input data
    - X_test - test split of input data
    - settings - dictionary of mixed values to determine which techniques are applied and how
    
    - log_base - int of log base to apply, None if no log transformation
    - scale_range - (int, int) tuple of the minmax range, None for no minmax scaling
    - use_standard - boolean for standard scaling(overriden by scale_range)
    - use_gaussian - sigma value for gaussian denoising/smoothing, None for no smoothing
    - adjust_seasonality - boolean for subtracting the best fit polynomial to reduce the effects of the circadian cycle

    Returns
    - (pd.Dataframe, pd.Dataframe) tuple of modified train and test data
    '''
    # use settings instead if available
    if settings is not None:
        key = settings.keys()
        log_base = settings['log_base'] if 'log_base' in key else log_base
        scale_range = settings['scale_range'] if 'scale_range' in key else scale_range
        use_standard = settings['use_standard'] if 'use_standard' in key else use_standard
        use_gaussian = settings['use_gaussian'] if 'use_gaussian' in key else use_gaussian
        adjust_seasonality = settings['adjust_seasonality'] if 'adjust_seasonality' in key else adjust_seasonality

    # apply log function to all values
    if log_base:
        X_train = X_train.map(lambda x: log_skip_zeroes(x, log_base))
        X_test = X_test.map(lambda x: log_skip_zeroes(x, log_base)) if X_test is not None else None

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
    
    # scale data to be within a specific range, either with minmax scaling or z score scaling
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

    # calculate seasonality on training data and subtract from all samples
    if adjust_seasonality:
        time_index = [ x for x in range(1440)]
        train_means = X_train.apply(data_mean_med_std, axis=0).loc['mean']
        degree = 5
        coef = np.polyfit(time_index, train_means, degree)

        curve = []
        for i in range(len(time_index)):
            val = coef[-1]
            for d in range(degree):
                val += time_index[i]**(degree-d)*coef[d]
            curve.append(val)

        X_train = X_train.apply(subtract_corresponding_minute, axis=1, args=(curve,))
        if X_test is not None:
            X_test = X_test.apply(subtract_corresponding_minute, axis=1, args=(curve,))

    return (X_train, X_test)


'''

Feature extraction functions.

'''


def _extract_stats_from_window(data: pd.Series, include_quarter_diff = False, simple_stats = False):
    '''
    Extracts stats from a window of actigraph data (total of ):
        - Sample Mean, Sample Std. Dev., Sample Skewness, Sample Kurtosis, Max, Min (6)
        - 1/2 win. change in means, max, min (3)
        - 1/4 win. means, Std. Dev.s, maxes, mins (16)
        - 1/4 win. means paired differences, maxes paired differences, mins paired differences

    Args
        - data: the actigraphy data series window to extract features from
        - include_quarter_diff - whether to include subtracted quarter differences as features
        - simple_stats - whether to exclude half window and quarter window statistics
    
    Returns
        - a dataframe containing a single row, where the columns are the extracted features 
    '''     
    data_np = data.to_numpy()
    
    stats = []
    labels = []

    # loading descriptive statistics
    labels.extend(['mean', 'median', 'std',  'max', 'min'])
    stats.extend([data_np.mean(), np.sort(data_np)[len(data_np)//2], data_np.std(), data_np.max(), data_np.min()])

    if not simple_stats:
        # calculating half window statistics
        h_win = [data_np[0:len(data_np)//2], data_np[len(data_np)//2:]]
        labels.extend(['h_mean_change', 'h_max_change', 'h_min_change'])
        stats.extend([h_win[1].mean() - h_win[0].mean(), h_win[1].max() - h_win[0].max(), h_win[1].min() - h_win[0].min()])
        
        # calculating quarter window statistics
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
    if not simple_stats and include_quarter_diff:
        for i in range(len(q_means)):
            for j in range(i+1, len(q_means)):
                labels.append(f'q{i}_minus_q{j}_mean')
                stats.append(q_means[i] - q_means[j])

    # convert stats and labels into series
    features = pd.Series(stats, index = labels)
    return features

def _create_feature_dataframe(data: pd.DataFrame, include_quarter_diff = False, simple_stats = False):
    '''
    Transforms actigraphy data into extracted feature data for an entire dataframe

    Args
    - data - the dataframe to extract features from
    - include_quarter_diff - whether to include subtracted quarter differences as features
    - simple_stats - whether to exclude half window and quarter window statistics

    Returns
    - dataframe like the input data, except with extracted features instead of the raw time series
    '''
    extracted_stats = data.apply(_extract_stats_from_window, axis=1, args=(include_quarter_diff, simple_stats)).reset_index(drop=True)
    extracted_stats.index = data.index

    return extracted_stats

def _create_long_feature_dataframe(data: pd.DataFrame, window_size = 30, include_quarter_diff = False, simple_stats=False):
    '''
    Transforms actigraphy data into extracted feature data for an entire dataframe with a sliding window approach.

    Args
    - data - the dataframe to extract features from
    - window_size - number of datapoints per window
    - include_quarter_diff - whether to include subtracted quarter differences as features
    - simple_stats - whether to exclude half window and quarter window statistics

    Returns
    - dataframe like the input data, except with extracted features instead of the raw time series for all windows
    '''
    def _extract_long_feature_series(x: pd.Series):
        '''
        Extracts long feature series from data, stacking each window side by side to make a single row.
        '''
        features_by_window = []
        for i in range(0, len(x), window_size):
            window = x[i:i+window_size]
            features_by_window.append(_extract_stats_from_window(window, include_quarter_diff, simple_stats))
        return pd.concat(features_by_window, axis=1).transpose().stack()
    
    extracted_stats_long = data.apply(_extract_long_feature_series, axis=1)
    extracted_stats_long.columns = ['_'.join(str(val) for val in col).strip() for col in extracted_stats_long.columns.values]
    return extracted_stats_long

'''

Preprocessing / Feature Extraction Pipeline

'''

def process_data_folds(data, kfolds, preprocessing_settings, feature_settings=None):
    '''
    Runs the entire data processing pipeline

    Args
    - data - the raw data dataframe
    - preprocessing_settings - settings dictionary for `preprocess_train_test_dataframes()`
    - feature_settings - settings dictionary for `_create_long_feature_dataframe()` and `_create_feature_dataframe()`

    Returns
    - a list of (DataFrame, DataFrame, list, list) tuples containing the final X_train, X_test, y_train, y_test data
    '''
    dataframes = []
    for i in tqdm(range(len(kfolds)),ncols=50):
        # extract train/test index names
        (train_index, test_index) = kfolds[i]
        
        # split data based on the extracted train/test split
        X = data.copy()
        X_train = X.drop(labels=test_index, axis=0)
        X_test = X.drop(labels=train_index, axis=0)
        y_train = list(X_train.pop('label'))
        y_test = list(X_test.pop('label'))
        
        if preprocessing_settings['resample']:
            # apply smote to training set
            X_train, y_train = _apply_smote(X_train, y_train, 0.1)
        
        # preprocess data accordingly for the model
        (X_train, X_test) = preprocess_train_test_dataframes(
                                X_train=X_train,
                                X_test=X_test,
                                settings=preprocessing_settings
                            )
       
        # extract features
        if feature_settings['use_feature'] and feature_settings['long_feature']:
            X_train = _create_long_feature_dataframe(X_train, window_size=feature_settings['window_size'], 
                                                    include_quarter_diff=feature_settings['quarter_diff'],
                                                    simple_stats=feature_settings['simple'])
            X_test = _create_long_feature_dataframe(X_test, window_size=feature_settings['window_size'], 
                                                   include_quarter_diff=feature_settings['quarter_diff'],
                                                   simple_stats=feature_settings['simple'])
        elif feature_settings['use_feature']:
            X_train = _create_feature_dataframe(X_train, feature_settings['quarter_diff'], simple_stats=feature_settings['simple'])
            X_test = _create_feature_dataframe(X_test, feature_settings['quarter_diff'], simple_stats=feature_settings['simple'])
        dataframes.append((X_train, X_test, y_train, y_test))

    return dataframes