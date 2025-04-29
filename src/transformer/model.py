import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape becomes [1, max_len, d_model] to allow broadcasting
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x should be [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]  # broadcasts over batch_size
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids):
        # embedded is [batch_size, seq_len, hidden_dim]
        embedded = self.embedding(input_ids)
        embedded = self.pos_encoder(embedded)

        transformer_output = self.transformer(embedded)
        transformer_output = self.dropout(transformer_output[:, 0, :])
        return self.linear(transformer_output)