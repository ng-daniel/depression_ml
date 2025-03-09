print("--------------------------------------")
print("DEPRESSION CLASSIFICATION MODEL // V.0")
print("--------------------------------------")
print("Loading libraries...")

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from data import load_preprocess_dataframe_labels, ActigraphDataset
from engine import train_step, test_step
from eval import eval_model
from model import ConvNN, LSTM
from util import print_model_performance_table

print("Loading data...")

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load dataframe
actigraph_data, actigraph_labels = load_preprocess_dataframe_labels(dir_names = ["data/control", "data/condition"],
                                                                    class_names = ["control", "condition"],
                                                                    time = "12:00:00")
# train test split
X_train, X_test, y_train, y_test = train_test_split(actigraph_data, 
                                                    actigraph_labels, 
                                                    test_size=0.25, 
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

print("Constructing models...")

# model hyperparameters
IN_SHAPE = 1
OUT_SHAPE = 1
HIDDEN_SHAPE = 32
FLATTEN_FACTOR = 720

# construct CNN model
model_cnn = ConvNN(IN_SHAPE, OUT_SHAPE, HIDDEN_SHAPE, FLATTEN_FACTOR).to(device)
# construct LSTM model
model_lstm = LSTM

# pick model
model_0 = model_cnn

# define criterion
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(params = model_0.parameters(), lr=0.001)

print("Training...")
print("----------------")
epochs = 10
for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_step(model_0, train_dataloader, optimizer, criterion, device)
    test_loss, test_acc = test_step(model_0, test_dataloader, criterion, device)
    print(f"{epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
print("----------------")

print("Evaluating...")
model_performance = eval_model(model = model_0, 
                               dataloader = test_dataloader,
                               criterion = criterion,
                               device = device)
print_model_performance_table([model_performance, model_performance, model_performance])

print("Done.")