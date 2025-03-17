import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import rfft
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
import torch
from torch.utils.data import Dataset, DataLoader

from util import log_skip_zeroes

CONDITION_SIZE = 23
CONTROL_SIZE = 32
DIR_PATH = "data/all"
scores = pd.read_csv("data/scores.csv", index_col='number')

class ActigraphDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float().unsqueeze(dim=1)
        self.y = torch.tensor(y).float()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        # code for reshaping 1d tensor to 2d with padding
        '''
        new_shape = math.ceil(X.shape[0] / INPUT_SIZE)
        padding = new_shape * INPUT_SIZE - X.shape[0]
        p = nn.ZeroPad1d((0,padding))
        X = p(X)
        X = X.reshape((new_shape, INPUT_SIZE))
        '''
        return X, y

def concat_data(dir_name: str, class_type: str, class_label: int, start_time: str, output_df: pd.DataFrame, scores_df: pd.DataFrame):

    dir_size = len(os.listdir(dir_name))
    for i in range(1, dir_size + 1):
        
        # read CSV into truncated dataframe
        
        filename = f"{class_type}_{i}"
        datapath = os.path.join(dir_name, filename + ".csv")
        file_df = pd.read_csv(datapath)

        # find occurences of 

        sr = file_df['timestamp'].map(lambda x: x.split()[1])
        indexes_of_time = list(sr[sr==start_time].index)

        # split file_df into intervals of days

        num_days = scores_df.loc[filename, "days"]
        for j in range(num_days):
            interval = [indexes_of_time[j], indexes_of_time[j+1]]
            day_df = file_df.iloc[interval[0]:interval[1]]
            day_df = day_df['activity'].rename(f"{class_label}_{i}_{j}").reset_index(drop=True)
            
            # concatenate data column to output dataframe
            if(len(day_df) == 1440):
                output_df = pd.concat([output_df, day_df], axis=1)

    return output_df

def load_dataframe_labels(dir_names: list, class_names: list, time: str):
    # load scores dataframe (information about each datafile)
    scores = pd.read_csv("data/scores.csv", index_col='number')
    # fill dataframe
    data = pd.DataFrame()
    for CLASS in range(len(dir_names)):
        data = concat_data(dir_names[CLASS], class_names[CLASS], 
                                    CLASS, time, data, scores)
    # transpose data so columns are time and rows are subjects
    data = data.transpose()
    # set labels
    labels = data.index.map(lambda x: int(x[0]))
    return data, labels

def kfolds_dataframes(data: pd.DataFrame, labels: list, numfolds: int, shuffle: bool, random_state: int, batch_size: int):
    # apply log function to all values
    data = data.map(lambda x: log_skip_zeroes(x))
    
    if shuffle:
        kf = KFold(n_splits=numfolds, shuffle=shuffle, random_state=42)
    else:
        kf = KFold(n_splits=numfolds, shuffle=shuffle)
    kf.get_n_splits(data)

    folds = []
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        
        X_train = data.iloc[train_index]
        X_test = data.iloc[test_index]
        y_train = [labels[i] for i in train_index]
        y_test = [labels[i] for i in test_index]

        folds.append((X_train, X_test, y_train, y_test))
    
    return folds

def preprocess_train_test_dataframes(X_train: pd.DataFrame, X_test: pd.DataFrame):
    # apply log function to all values
    X_train = X_train.map(lambda x: log_skip_zeroes(x))
    X_test = X_test.map(lambda x: log_skip_zeroes(x))
    
    # scale data to be within 0-1
    scaler = MinMaxScaler((0,1))
    X_train = pd.DataFrame(scaler.fit_transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))

    return (X_train, X_test)

def create_dataloaders(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: list, y_test: list,
                         shuffle: bool, batch_size: int):
    # wrap in pytorch dataloader
    train_dataset = ActigraphDataset(X_train.to_numpy(), y_train)
    test_dataset = ActigraphDataset(X_test.to_numpy(), y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return (train_dataloader, test_dataloader)

def extract_stats_from_window(data: pd.Series):
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
    # loading descriptive statistics
    
    desc_stats = pd.DataFrame(
        {
            'mean':[data.mean()],
            'std':[data.std()],
            'skew':[data.skew()],
            'kurt':[data.kurt()],
            'max':[data.max()],
            'min':[data.min()]
        }
    )
    
    # calculating half window statistics
    
    h_win = [data.iloc[0:len(data)//2], data.iloc[len(data)//2:]]
    half_stats = pd.DataFrame(
        {
            'h_mean_change':[h_win[1].mean() - h_win[0].mean()],
            'h_max_change':[h_win[1].max() - h_win[0].max()],
            'h_min_change':[h_win[1].min() - h_win[0].min()]
        }
    )
    
    # calculating quarter window statistics
    
    quarter_stats = pd.DataFrame()
    q_win = []
    for i in range(0, len(data), len(data)//4):
        win = data.iloc[i:i+len(data)//4]
        quarter_stats[f'q{i+1}_mean'] = win.mean()
        quarter_stats[f'q{i+1}_std'] = win.std()
        quarter_stats[f'q{i+1}_max'] = win.max()
        quarter_stats[f'q{i+1}_min'] = win.min()
        q_win.append(win)

    # concatenating all dfs into a single df
    features = pd.concat([desc_stats, half_stats, quarter_stats], axis=1)
    return features.iloc[0]

def extract_fft_from_window(data: pd.Series):
    '''
    Extract signal information from a window of actigraph data using a fast fourier transform:
        - Magnitude of Frequency Components of FFT (720)
        - Top 10 Frequency values
    '''
    # calculating fft values and listing the top 10 largest amplitudes
    
    scaler = MinMaxScaler((0,1))
    zero_dc_data = data.map(lambda x: x - data.mean())
    fourier_transform = rfft(zero_dc_data, len(zero_dc_data))
    ft_abs_scaled = scaler.fit_transform(np.abs(fourier_transform).reshape(-1, 1)).squeeze()
    top_fourier = list(pd.Series(ft_abs_scaled).sort_values(ascending=False).head(n=10).index)
    
    # dividing by number of frequencies to keep values at a reasonable value for ML

    top_fourier = [freq / len(ft_abs_scaled) for freq in top_fourier]
    
    # loading values into a dataframe with descriptive column names
    
    fft_data = pd.DataFrame(ft_abs_scaled).transpose()
    fft_col_names = [f'fft_{i+1}' for i in range(len(ft_abs_scaled))]
    fft_data.columns = fft_col_names
    for i in range(len(top_fourier)):
        fft_data[f'fft_top_{i+1}'] = top_fourier[i]
    return fft_data.iloc[0]

def create_feature_dataframe(data: pd.DataFrame, raw_data: pd.DataFrame):
    feature_rows = []

    # for i in range(len(data)):
    #     win = data.iloc[i]
    #     raw_win = raw_data.iloc[i]
    #     stats_row = extract_stats_from_window(win)
    #     fft_row = extract_fft_from_window(raw_win)
    #     feature_row = pd.concat([stats_row, fft_row], axis = 1)
    #     feature_rows.append(feature_row)

    extracted_stats = data.apply(extract_stats_from_window, axis=1).reset_index(drop=True)
    extracted_fft = raw_data.apply(extract_fft_from_window, axis=1).reset_index(drop=True)
    features = pd.concat([extracted_stats, extracted_fft], axis=1, )
    print(features)
    features.index = data.index
    return features