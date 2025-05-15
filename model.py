import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, in_shape, out_shape, hidden_shape, lstm_layers):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hidden_shape = hidden_shape
        self.lstm_layers = lstm_layers
        
        # N * 1 * L * in_shape
        self.norm = nn.BatchNorm2d(num_features=1)
        # hidden shape = number of Long term memory values
        self.lstm = nn.LSTM(in_shape, hidden_shape, lstm_layers, batch_first=True)
        self.fc = nn.Sequential(
            # nn.Linear(hidden_shape, hidden_shape),
            # nn.ReLU(),
            nn.Linear(hidden_shape, out_shape)
        )
    def forward(self, x):
        # number of LSTM layers * L * hidden shape
        h0 = torch.zeros(self.lstm_layers, 
                         x.size(0), 
                         self.hidden_shape).to(x.device)
        c0 = torch.zeros(self.lstm_layers, 
                         x.size(0), 
                         self.hidden_shape).to(x.device)
        # print(x.shape)
        x = x.squeeze(dim=1).unsqueeze(dim=2)
        x = torch.reshape(x, (x.size(0), int(x.size(1) / self.in_shape), self.in_shape))
        # print(x.shape)
        x, (hn, cn) = self.lstm(x, (h0, c0))
        # print(f"{x.shape} | {hn.shape} | {cn.shape}")
        x = self.fc(x[:, -1, :])
        # print(x.shape)
        return x

class ZeroR(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.zeros(len(x)).unsqueeze(1).to(x.device)

class ConvNN(nn.Module):
    def __init__(self, in_shape, output_dim, hidden_shape, flatten_factor):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(in_shape, hidden_shape, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.Conv1d(hidden_shape, hidden_shape, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(hidden_shape, hidden_shape, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.Conv1d(hidden_shape, hidden_shape, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_shape * flatten_factor, hidden_shape),
            # nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_shape, output_dim)
        )
    def forward(self, x):
        #print(x.shape)
        x = self.conv_block_1(x)
        #print(x.shape)
        #x = self.conv_block_2(x)
        x = self.linear(x)
        #print(x.shape)
        return x

class FeatureMLP(nn.Module):
    def __init__(self, in_shape, out_shape, hidden_shape):
        super().__init__()
        self.linear_1 = nn.Sequential(
            nn.Linear(in_shape, hidden_shape),
            # nn.Dropout(),
            nn.ReLU(),
        )
        self.linear_2 = nn.Sequential(
            nn.Linear(hidden_shape, hidden_shape),
            nn.Dropout(),
            nn.ReLU(),
        )
        self.linear_3 = nn.Sequential(
            nn.Linear(hidden_shape, out_shape)
        )
    def forward(self, x):
        #print(x.shape)
        x = self.linear_1(x)
        #print(x.shape)
        # x = self.linear_2(x)
        #print(x.shape)
        x = self.linear_3(x)
        #print(x.shape)
        return x

class LSTM_Feature(nn.Module):
    def __init__(self, in_shape, out_shape, hidden_shape, lstm_layers):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hidden_shape = hidden_shape
        self.lstm_layers = lstm_layers
        
        # N * 1 * L * in_shape
        self.norm = nn.BatchNorm2d(num_features=1)
        # hidden shape = number of Long term memory values
        self.lstm = nn.LSTM(in_shape, hidden_shape, lstm_layers, batch_first=True)
        self.fc = nn.Sequential(
            # nn.Linear(hidden_shape, hidden_shape),
            # nn.ReLU(),
            nn.Linear(hidden_shape, out_shape)
        )
    def forward(self, x):
        # number of LSTM layers * L * hidden shape
        h0 = torch.zeros(self.lstm_layers, 
                         x.size(0), 
                         self.hidden_shape).to(x.device)
        c0 = torch.zeros(self.lstm_layers, 
                         x.size(0), 
                         self.hidden_shape).to(x.device)
        # print(x.shape)
        x, _ = self.lstm(x, (h0, c0))
        # print(f"{x.shape} | {hn.shape} | {cn.shape}")
        x = self.fc(x[:, -1, :])
        # print(x.shape)
        return x
    
class ConvLSTM(nn.Module):
    def __init__(self, in_shape, out_shape, hidden_shape, lstm_layers):
        super().__init__()
        
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hidden_shape = hidden_shape
        self.lstm_layers = lstm_layers
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(in_shape, hidden_shape, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.Conv1d(hidden_shape, hidden_shape, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(hidden_shape, hidden_shape, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.Conv1d(hidden_shape, hidden_shape, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.lstm = nn.LSTM(hidden_shape, hidden_shape, lstm_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_shape, hidden_shape),
            nn.ReLU(),
            nn.Linear(hidden_shape, out_shape)
        )
    def forward(self, x):
        h0 = torch.zeros(self.lstm_layers, 
                         x.size(0), 
                         self.hidden_shape).to(x.device)
        c0 = torch.zeros(self.lstm_layers, 
                         x.size(0), 
                         self.hidden_shape).to(x.device)
        
        x = self.conv_block_1(x)
        #x = torch.reshape(x, (x.size(0), int(x.size(1) / self.in_shape), self.in_shape))
        # print(x)
        # print(x.shape)
        x = x.permute((0,2,1))
        x, _ = self.lstm(x, (h0, c0))
        x = self.fc(x[:, -1, :])

        return x