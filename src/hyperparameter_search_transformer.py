import optuna
import torch
import pandas as pd
from torch import nn
from utils import build_vocabulary
from transformer.model import TransformerModel
from transformer.dataset import create_loaders
from train_transformer import train_one_epoch, validate
import mlflow

df = pd.read_csv("./data.csv")
specials = ["<PAD>", "pKi", "pIC50", "ENZ1", "ENZ2", "ENZ3", "ENZ4"]
vocab = build_vocabulary(df, specials)
batch_size = 256
train_loader, dev_loader, test_loader = create_loaders(df, vocab, batch_size)

def objective(trial):
    # Hyperparameters to search
    # hidden_dim = trial.suggest_categorical("hidden_dim", [512, 768])
    hidden_dim = 768
    # dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.25)
    dropout_rate = 0.1
    num_heads = trial.suggest_categorical('num_heads', [8, 16, 32])
    num_layers = trial.suggest_int('num_layers', 4, 12)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    model = TransformerModel(
        vocab_size=len(vocab),
        hidden_dim=hidden_dim,
        output_dim=1,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout_rate
    ).to(device)

    # Set up training
    criterion = nn.MSELoss()
    lr = 1e-4
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    anneal_strategy = "cos" # alternative is "linear"
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=100, anneal_strategy=anneal_strategy)
    inverted_vocab = {v: k for k, v in vocab.items()}

    with mlflow.start_run():
        mlflow.log_params({
            "hidden_dim": hidden_dim,
            "dropout_rate": dropout_rate,
            "learning_rate": lr,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "scheduler": "OneCycleLR",
            "anneal_strategy": anneal_strategy,
            "weight_decay": weight_decay
        })
        epochs = 100
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scheduler=scheduler)
            val_loss = validate(model, dev_loader, criterion, device, epoch, inverted_vocab)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=epoch)
        mlflow.pytorch.log_model(model, "model")
    return val_loss

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Transformer architecture search")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    
    print("Best hyperparameters:", study.best_params)
    print("Best value (val_loss):", study.best_value)