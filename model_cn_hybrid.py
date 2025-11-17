import torch
import torch.nn as nn

class CNNLSTM_Hybrid(nn.Module):
    def __init__(self, feature_dim=5):
        super(CNNLSTM_Hybrid, self).__init__()
        
        # CNN for sequence (input shape: [batch, max_len=300, 4])
        self.cnn = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )
        
        # LSTM after CNN (input size = 128 channels)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2,
                            batch_first=True, bidirectional=True)
        
        # Fully connected layers after concatenating handcrafted features
        self.fc = nn.Sequential(
            nn.Linear(64 * 2 + feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    
    def forward(self, seq, features):
        # seq shape: [batch, max_len, 4]
        x = seq.permute(0, 2, 1)  # [batch, 4, max_len]
        x = self.cnn(x)           # [batch, 128, seq_len//(4*4)]
        x = x.permute(0, 2, 1)    # [batch, seq_len, channels]
        
        # LSTM returns (output, (hidden, cell))
        _, (h_n, _) = self.lstm(x)
        
        # Concatenate last forward and backward hidden states
        h_out = torch.cat((h_n[-2], h_n[-1]), dim=1)  # [batch, 128]
        
        # Concatenate handcrafted features
        combined = torch.cat((h_out, features), dim=1)  # [batch, 128 + feature_dim]
        
        # Fully connected layers
        out = self.fc(combined)
        return out

