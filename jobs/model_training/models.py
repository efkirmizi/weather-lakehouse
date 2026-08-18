import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, pred_len, dropout):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, pred_len * output_dim)

    def forward(self, x):
        x = self.pos_encoder(self.input_projection(x))
        encoded = self.transformer_encoder(x)
        return self.output_projection(encoded.mean(dim=1)).view(-1, self.pred_len, self.output_dim)


class ConvLSTMWeatherForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pred_len, dropout):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        
        self.feature_extractor = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=hidden_dim, 
            kernel_size=3, 
            padding=1 
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, pred_len * output_dim)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x) 
        last_state = lstm_out[:, -1, :] 
        out = self.fc_head(last_state)
        return out.view(-1, self.pred_len, self.output_dim)
