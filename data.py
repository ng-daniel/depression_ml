import os
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset, DataLoader
from util import log_skip_zeroes

CONDITION_SIZE = 23
CONTROL_SIZE = 32
DIR_PATH = "data/all"
scores = pd.read_csv("data/scores.csv", index_col='number')

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

def load_preprocess_dataframe_labels(dir_names: list, class_names: list, time: str):
    # load scores dataframe (information about each datafile)
    scores = pd.read_csv("data/scores.csv", index_col='number')
    # fill dataframe
    data = pd.DataFrame()
    for CLASS in range(len(dir_names)):
        data = concat_data(dir_names[CLASS], class_names[CLASS], 
                                    CLASS, time, data, scores)
    # transpose data so columns are time and rows are subjects
    data = data.transpose()
    # apply log function to all values
    data = data.map(lambda x: log_skip_zeroes(x))
    # set labels
    labels = data.index.map(lambda x: int(x[0]))
    return data, labels

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

def train_test_dataloaders(data: pd.DataFrame, labels: list, test_size: float, shuffle: bool, random_state: int, batch_size: int):
    # train test split
    if shuffle:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, 
                                                            test_size=test_size, 
                                                            shuffle=shuffle, 
                                                            random_state=random_state)
    else:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, 
                                                            test_size=test_size, 
                                                            shuffle=shuffle)
    # scale data to be within 0-1
    scaler = MinMaxScaler((0,1))
    X_train = pd.DataFrame(scaler.fit_transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))

    # wrap in pytorch dataloader
    train_dataset = ActigraphDataset(X_train.to_numpy(), y_train)
    test_dataset = ActigraphDataset(X_test.to_numpy(), y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, test_dataloader

def kfolds_dataloaders(data: pd.DataFrame, labels: list, numfolds: int, shuffle: bool, random_state: int, batch_size: int):
    if shuffle:
        kf = KFold(n_splits=numfolds, shuffle=shuffle, random_state=42)
    else:
        kf = KFold(n_splits=numfolds, shuffle=shuffle)
    kf.get_n_splits(data)
    scaler = MinMaxScaler((0,1))    

    folds = []
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        
        X_train = data.iloc[train_index]
        X_test = data.iloc[test_index]
        y_train = [labels[i] for i in train_index]
        y_test = [labels[i] for i in test_index]

        X_train = pd.DataFrame(scaler.fit_transform(X_train))
        X_test = pd.DataFrame(scaler.transform(X_test))

        train_dataset = ActigraphDataset(X_train.to_numpy(), y_train)
        test_dataset = ActigraphDataset(X_test.to_numpy(), y_test)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

        folds.append((train_dataloader, test_dataloader))
    
    return folds