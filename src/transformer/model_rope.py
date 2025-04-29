import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings

# max_seq_len must be set - could do this dynamically from dataset but set it manually for now
# Longest token sequence in dataset is 98, + 3 special tokens = 101

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim, num_heads, num_layers, dropout=0.1, max_seq_len=110):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = RotaryPositionalEmbeddings(dim=hidden_dim // num_heads, max_seq_len=max_seq_len)
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
        
        # Reshape for RoPE: [batch_size, seq_len, num_heads, head_dim]
        batch_size, seq_len, hidden_dim = embedded.size()
        num_heads = self.transformer.layers[0].self_attn.num_heads
        head_dim = hidden_dim // num_heads
        embedded = embedded.view(batch_size, seq_len, num_heads, head_dim)
        
        # Apply RoPE
        embedded = self.pos_encoder(embedded)
        
        # Reshape back to [batch_size, seq_len, hidden_dim]
        embedded = embedded.view(batch_size, seq_len, hidden_dim)

        transformer_output = self.transformer(embedded)
        transformer_output = self.dropout(transformer_output[:, 0, :])
        return self.linear(transformer_output)