import optuna
import pandas as pd
import torch
import mlflow
import mlflow.pytorch
from gnn.dataset import create_loaders
from gnn.model import GraphModel
from train_gnn import train_one_epoch, validate
from torch_geometric.nn import (
    GATConv, GCNConv, SAGEConv, GraphConv, TAGConv,
    LayerNorm, BatchNorm,
    global_mean_pool, global_max_pool, global_add_pool
)
from rdkit import Chem

df = pd.read_csv("./data.csv")
elements = set()
for smi in df['SMILES']:
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        elements.add(atom.GetSymbol())
elements = list(elements)
batch_size = 768
train_loader, dev_loader, test_loader = create_loaders(df, batch_size, elements)

def objective(trial):
    # Hyperparameters to search
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768])
    num_layers = trial.suggest_int('num_layers', 3, 10)
    conv_type = trial.suggest_categorical("conv_type", [GATConv, GCNConv, SAGEConv, GraphConv, TAGConv])
    norm_type = trial.suggest_categorical("norm_type", [None, LayerNorm, BatchNorm])
    pool_type = trial.suggest_categorical("pool_type", [global_mean_pool, global_max_pool, global_add_pool])
    dropout_rate = 0.1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    model = GraphModel(
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        conv_type=conv_type,
        norm_type=norm_type,
        pool_type=pool_type,
        dropout=dropout_rate,
        output_dim=1, 
    ).to(device)

    # Set up training
    criterion = torch.nn.MSELoss()
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 50

    with mlflow.start_run():
        mlflow.log_params({
            "hidden_dim": hidden_dim,
            "dropout_rate": dropout_rate,
            "learning_rate": lr,
            "num_layers": num_layers,
            "batch_size": batch_size,
            "conv_type": conv_type,
            "norm_type": norm_type,
            "pool_type": pool_type
        })
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
            val_loss = validate(model, dev_loader, criterion, device, epoch)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.pytorch.log_model(model, "model")
    return val_loss

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")  # Set your MLflow tracking URI
    mlflow.set_experiment("GNN Hyperparameter Search")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    
    print("Best hyperparameters:", study.best_params)
    print("Best value (val_loss):", study.best_value)