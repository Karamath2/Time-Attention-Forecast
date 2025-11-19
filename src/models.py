# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.V = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, hidden_states, last_hidden):
        # hidden_states: (batch, seq_len, hidden_dim)
        # last_hidden: (batch, hidden_dim)  (or (batch, 1, hidden_dim))
        seq_len = hidden_states.size(1)
        last_hidden_expanded = last_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        score = self.V(torch.tanh(self.W1(hidden_states) + self.W2(last_hidden_expanded))).squeeze(-1)
        # score shape: (batch, seq_len)
        attn_weights = torch.softmax(score, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, attn_weights

class LSTMWithAttention(nn.Module):
    def __init__(self, n_features, hidden_dim=64, n_layers=1, attn_dim=32, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers>1 else 0)
        self.attention = BahdanauAttention(hidden_dim, attn_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, (h_n, c_n) = self.lstm(x)  # out: (batch, seq_len, hidden_dim)
        # use last layer hidden:
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        context, attn_weights = self.attention(out, last_hidden)
        out = self.fc(context).squeeze(-1)
        return out, attn_weights

class LSTMPlain(nn.Module):
    def __init__(self, n_features, hidden_dim=64, n_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers>1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.fc(last_hidden).squeeze(-1)
        return out
