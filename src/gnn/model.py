import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, GATConv, LayerNorm

class GraphModel(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, conv_type=GATConv, norm_type=LayerNorm, pool_type=global_mean_pool, dropout=0.1, input_dim=10, cat_feature_dim=6):
        super(GraphModel, self).__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.pool_type = pool_type
        self.convs.append(conv_type(input_dim, hidden_dim))
        if norm_type is not None:
            self.norms.append(norm_type(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(conv_type(hidden_dim, hidden_dim))
            if norm_type is not None:
                self.norms.append(norm_type(hidden_dim))
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout)

        # Embedding layer for categorical features
        self.cat_embed = nn.Linear(cat_feature_dim, hidden_dim)

        # Final projection
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.norm_type is not None:
                norm = self.norms[i]
                x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
        x = self.pool_type(x, batch)

        # Convert data.cat_features from (6*batch_size) to (batch_size, 6)
        cat_features = data.cat_features.view(data.num_graphs, -1)
        cat_emb = torch.relu(self.cat_embed(cat_features))

        # Add embedding of cat_features to the pooled graph embedding
        x = x + cat_emb

        x = self.linear(x)
        return x