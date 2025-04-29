import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
import mlflow
import mlflow.pytorch
from sklearn.preprocessing import MinMaxScaler
from utils import build_vocabulary
from transformer.dataset import (
    create_loaders
)
# from model import TransformerModel
from transformer.model_rope import TransformerModel

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scaler=None, scheduler=None):
    model.train()
    running_loss = 0
    if scaler is not None:
        running_unscaled_loss = 0
    for batch in tqdm(loader, desc=f"Training Epoch {epoch+1}"):
        input_ids, targets = batch
        input_ids, targets = input_ids.to(device), targets.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs.squeeze(-1), targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item()

        # In case scaling is used, calculate unscaled loss for comparison
        if scaler is not None:
            unscaled_targets = scaler.inverse_transform(targets.cpu().numpy().reshape(-1, 1)).reshape(-1)
            unscaled_outputs = scaler.inverse_transform(outputs.squeeze(-1).detach().cpu().numpy().reshape(-1, 1)).reshape(-1)
            unscaled_loss = criterion(torch.tensor(unscaled_outputs), torch.tensor(unscaled_targets))
            running_unscaled_loss += unscaled_loss.item()

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch+1}, Training Loss: {avg_loss}")
    if scaler is not None:
        avg_unscaled_loss = running_unscaled_loss / len(loader)
        print(f"Epoch {epoch+1}, Unscaled Training Loss: {avg_unscaled_loss}")
        return avg_loss, avg_unscaled_loss
    return avg_loss

def validate(model, loader, criterion, device, epoch, invert_vocab, verbose=False, scaler=None):
    model.eval()
    val_loss = 0
    if scaler is not None:
        unscaled_val_loss = 0
    examples_printed = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Validation Epoch {epoch+1}"):
            input_ids, targets = batch
            input_ids, targets = input_ids.to(device), targets.to(device, dtype=torch.float32)
            outputs = model(input_ids)
            loss = criterion(outputs.squeeze(-1), targets)
            val_loss += loss.item()

            if scaler is not None:
                unscaled_targets = scaler.inverse_transform(targets.cpu().numpy().reshape(-1, 1)).reshape(-1)
                unscaled_outputs = scaler.inverse_transform(outputs.squeeze(-1).detach().cpu().numpy().reshape(-1, 1)).reshape(-1)
                unscaled_loss = criterion(torch.tensor(unscaled_outputs), torch.tensor(unscaled_targets))
                unscaled_val_loss += unscaled_loss.item()

            if verbose:
                if examples_printed < 5:
                    for i in range(min(input_ids.size(0), 5 - examples_printed)):
                        ids_cpu = input_ids[i].cpu()
                        measurement = invert_vocab.get(ids_cpu[0].item(), "?")
                        enzyme = invert_vocab.get(ids_cpu[1].item(), "?")
                        smi = [invert_vocab.get(tok.item(), "?") for tok in ids_cpu[2:] if tok.item() != 0]
                        predicted = outputs[i].item()
                        actual = targets[i].item()

                        if scaler is not None:
                            predicted = scaler.inverse_transform([[predicted]])[0][0]
                            actual = scaler.inverse_transform([[actual]])[0][0]

                        tqdm.write(
                            f"Measurement: {measurement} | enzyme: {enzyme} | SMILES: {''.join(smi)} | "
                            f"Predicted: {predicted:.4f} | Actual: {actual:.4f}"
                        )
                        examples_printed += 1
                        if examples_printed >= 5:
                            break

    avg_loss = val_loss / len(loader)
    print(f"Epoch {epoch+1}, Validation Loss: {avg_loss}")
    if scaler is not None:
        avg_unscaled_loss = unscaled_val_loss / len(loader)
        print(f"Epoch {epoch+1}, Unscaled Validation Loss: {avg_unscaled_loss}")
        return avg_loss, avg_unscaled_loss
    return avg_loss

if __name__ == "__main__":
    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Transformer RoPE retrain")

    # Prepare dataloaders
    df = pd.read_csv("./enzyme.csv")

    # For testing minmax scaling
    scaler = None
    use_minmax_scaler = False
    if use_minmax_scaler:
        scaler = MinMaxScaler()
        column_to_scale = "measurement_value"
        df.loc[:, column_to_scale] = scaler.fit_transform(df.loc[:, column_to_scale].values.reshape(-1, 1))

    # Set up dataloaders
    specials = ["<PAD>", "pKi", "pIC50", "ENZ1", "ENZ2", "ENZ3", "ENZ4"]
    vocab = build_vocabulary(df, specials)
    pad_token_idx = vocab["<PAD>"]
    batch_size = 256
    train_loader, dev_loader, test_loader = create_loaders(df, vocab, batch_size, randomize_smiles=True)

    # Build and train model
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    dropout_rate = 0.1
    hidden_dim = 768
    num_heads = 32
    num_layers = 8

    model = TransformerModel(
        vocab_size=len(vocab),
        hidden_dim=hidden_dim,
        output_dim=1,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout_rate
    ).to(device)

    criterion = nn.MSELoss()
    lr = 1e-4
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    epochs = 100
    anneal_strategy = "cos" # alternative is "linear"
    pct_start = 0.3 # fraction of run used to ramp learning rate
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs, anneal_strategy=anneal_strategy, pct_start = pct_start)
    # scheduler = None
    inverted_vocab = {v: k for k, v in vocab.items()}

    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("num_heads", num_heads)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("dropout_rate", dropout_rate)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("scheduler", "OneCycleLR")
        mlflow.log_param("anneal_strategy", anneal_strategy)

        for epoch in range(epochs):
            if scaler is not None:
                train_loss, unscaled_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler=scaler, scheduler=scheduler)
                val_loss, unscaled_val_loss = validate(model, dev_loader, criterion, device, epoch, inverted_vocab, verbose=True, scaler=scaler)
                mlflow.log_metric("train_unscaled_loss", unscaled_train_loss, step=epoch)
                mlflow.log_metric("val_unscaled_loss", unscaled_val_loss, step=epoch)
            else:
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler=scaler, scheduler=scheduler)
                val_loss = validate(model, dev_loader, criterion, device, epoch, inverted_vocab, verbose=True, scaler=scaler)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=epoch)
            if (epoch + 1) % 100 == 0:
                mlflow.pytorch.log_model(model, f"model_epoch_{epoch+1}")