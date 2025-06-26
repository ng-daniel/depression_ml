import torch
from torch import nn

class LSTM(nn.Module):
    '''
    LSTM architecture
    "Batches" actigraphy data points in groups of `in_shape`, aka number of features per row.
    '''
    def __init__(self, in_shape, out_shape, hidden_shape, lstm_layers):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hidden_shape = hidden_shape
        self.lstm_layers = lstm_layers
        
        # N * 1 * L * in_shape
        # hidden shape = number of Long term memory values
        self.lstm = nn.LSTM(in_shape, hidden_shape, lstm_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_shape),
            nn.Dropout(0,1),
            nn.Linear(hidden_shape, out_shape)
        )
    def forward(self, x):
        # number of LSTM layers * L * hidden shape
        x = x.squeeze(dim=1).unsqueeze(dim=2)
        x = torch.reshape(x, (x.size(0), int(x.size(1) / self.in_shape), self.in_shape))
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

class ZeroR(nn.Module):
    '''
    Zero Rule Baseline Class that only predicts the majority class (zero)
    '''
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.zeros(len(x)).unsqueeze(1).to(x.device)

class ConvNN(nn.Module):
    '''
    1D Convolutional Neural Net
    2 Filters w. Batch Norm and ReLU Activation
    1 Max Pool to reduce dimensions in half
    Flatten + FC layer
    '''
    def __init__(self, in_shape, output_dim, hidden_shape, flatten_factor):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_shape, hidden_shape, kernel_size=3, padding = 1),
            nn.BatchNorm1d(hidden_shape),
            nn.ReLU(),
            nn.Conv1d(hidden_shape, hidden_shape, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_shape * flatten_factor, hidden_shape),
            nn.ReLU(),
            nn.Linear(hidden_shape, output_dim)
        )
    def forward(self, x):
        #print(x.shape)
        x = self.conv_block(x)
        #print(x.shape)
        x = self.linear(x)
        #print(x.shape)
        return x

class FeatureMLP(nn.Module):
    '''
    Multi-layer Perceptron
    '''
    def __init__(self, in_shape, out_shape, hidden_shape):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_shape, hidden_shape),
            #nn.BatchNorm1d(hidden_shape),
            nn.ReLU(),
            nn.Linear(hidden_shape, hidden_shape),
            #nn.BatchNorm1d(hidden_shape),
            nn.ReLU(),
            nn.Linear(hidden_shape, out_shape)
        )
    def forward(self, x):
        x = self.linear(x)
        return x

class LSTM_Feature(nn.Module):
    '''
    LSTM model on extracted feature data instead.
    '''
    def __init__(self, in_shape, out_shape, hidden_shape, lstm_layers, window_size):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hidden_shape = hidden_shape
        self.lstm_layers = lstm_layers
        self.LEN_DAY = 1440
        self.seq_len = self.LEN_DAY // window_size
        self.num_features = self.in_shape // self.seq_len
        
        # N * 1 * L * in_shape
        # hidden shape = number of Long term memory values
        self.lstm = nn.LSTM(self.num_features, hidden_shape, lstm_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(hidden_shape, out_shape)
        )
    def forward(self, x):
        x = torch.reshape(x, (x.size(0), self.seq_len, self.num_features))
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        # x = x[:, -1, :]
        x = self.fc(x)
        return x
    
class ConvLSTM(nn.Module):
    '''
    Hybrid model: ConvNN to extract features, LSTM to learn the sequence.
    3 Filters with Batch Normalization, ReLU Activation, and Max Pooling
    Lstm layers followed by a fully connected layer
    '''
    def __init__(self, in_shape, out_shape, hidden_shape, lstm_layers):
        super().__init__()
        
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hidden_shape = hidden_shape
        self.lstm_layers = lstm_layers
        self.conv_hidden = 32
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, self.conv_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.conv_hidden),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(self.conv_hidden, self.conv_hidden * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.conv_hidden * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(self.conv_hidden * 2, self.conv_hidden * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.conv_hidden * 4),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(self.conv_hidden * 4, 
                            hidden_shape, 
                            lstm_layers, 
                            batch_first=True)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_shape),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_shape, out_shape)
        )
    def forward(self, x):
        x = self.conv_block(x)
        x = x.permute((0,2,1))
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        # x = x[:,-1,:]
        x = self.fc(x)
        return x