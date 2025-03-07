print("--------------------------------------")
print("DEPRESSION CLASSIFICATION MODEL // V.0")
print("--------------------------------------")
print("Loading libraries...")

from data import concat_data, ActigraphDataset
from util import log_skip_zeroes
from model import ConvNN
from engine import train_step, test_step

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

print("Starting...")

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load scores dataframe (information about each datafile)
scores = pd.read_csv("data/scores.csv", index_col='number')
# fill dataframe
actigraph_data = pd.DataFrame()
actigraph_data = concat_data("data/control", 32, "control", 
                             0, "12:00:00", actigraph_data, scores)
actigraph_data = concat_data("data/condition", 23, "condition", 
                             1, "12:00:00", actigraph_data, scores)

# transpose data so columns are time and rows are subjects
actigraph_data = actigraph_data.transpose()
# apply log function to all values
actigraph_data = actigraph_data.map(lambda x: log_skip_zeroes(x))
# set labels
actigraph_labels = actigraph_data.index.map(lambda x: int(x[0]))
# train test split
X_train, X_test, y_train, y_test = train_test_split(actigraph_data, 
                                                    actigraph_labels, 
                                                    test_size=0.5, 
                                                    shuffle=True, 
                                                    random_state=42)

# scale data to be within 0-1
scaler = MinMaxScaler((0,1))
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))

# wrap in pytorch dataloader
train_dataset = ActigraphDataset(X_train.to_numpy(), y_train)
test_dataset = ActigraphDataset(X_test.to_numpy(), y_test)
train_dataloader = DataLoader(train_dataset, 32, shuffle=True)
test_dataloader = DataLoader(test_dataset, 32, shuffle=True)

print("Data finished loading...")

# model hyperparameters
IN_SHAPE = 1
OUT_SHAPE = 1
HIDDEN_SHAPE = 32
FLATTEN_FACTOR = 720

# construct model
model_0 = ConvNN(IN_SHAPE, OUT_SHAPE, HIDDEN_SHAPE, FLATTEN_FACTOR).to(device)

# define criterion
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(params = model_0.parameters(), lr=0.001)

print("Training...")
print("----------------")

epochs = 15
for epoch in range(epochs):
    train_loss, train_acc = train_step(model_0, train_dataloader, optimizer, criterion, device)
    test_loss, test_acc = test_step(model_0, test_dataloader, criterion, device)
    print(f"{epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

print("----------------")
print("Done.")