import pandas as pd
import torch
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from gnn.dataset import create_loaders
from gnn.model import GraphModel

from torch_geometric.nn import TAGConv, BatchNorm, global_mean_pool
from rdkit import Chem

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs.squeeze(-1), batch.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss}')
    return avg_train_loss

def validate(model, test_loader, criterion, device, epoch):
    model.eval()
    val_loss = 0
    examples_printed = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Validation Epoch {epoch+1}"):
            batch = batch.to(device)
            outputs = model(batch)
            loss = criterion(outputs.squeeze(-1), batch.y)
            val_loss += loss.item()

            # Print examples
            if examples_printed < 5:
                for i in range(min(batch.num_graphs, 5 - examples_printed)):
                    predicted = outputs[i].item()
                    actual = batch.y[i].item()
                    tqdm.write(
                        f"SMILES: {batch.smiles[i]}, "
                        f"Type: {batch.measurement_type[i]}, "
                        f"enzyme: {batch.enzyme_name[i]}, "
                        f"Predicted: {predicted:.4f} | Actual: {actual:.4f}"
                    )
                    examples_printed += 1
                    if examples_printed >= 5:
                        break

    avg_val_loss = val_loss / len(test_loader)
    print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss}')
    return avg_val_loss

if __name__ == "__main__":
    # Initialize MLflow
    mlflow.set_tracking_uri("http://localhost:5000")  # Set your MLflow tracking URI
    mlflow.set_experiment("GNN retrain best")
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    # Extract all elements from dataset for one-hot feature encoding
    df = pd.read_csv('./enzyme.csv')
    elements = set()
    for smi in df['SMILES']:
        mol = Chem.MolFromSmiles(smi)
        for atom in mol.GetAtoms():
            elements.add(atom.GetSymbol())
    elements = list(elements)
    
    # Create DataLoaders
    batch_size = 768
    train_loader, dev_loader, test_loader = create_loaders(df, batch_size, elements)

    # Model parameters
    dropout_rate = 0.1
    hidden_dim = 768
    num_layers = 5
    conv_type = TAGConv
    norm_type = BatchNorm
    pool_type = global_mean_pool

    model = GraphModel(
        hidden_dim=hidden_dim, 
        output_dim=1, 
        num_layers=num_layers, 
        conv_type=conv_type,
        norm_type=norm_type,
        pool_type=pool_type,
        dropout=dropout_rate,
    ).to(device)

    criterion = torch.nn.MSELoss()
    lr=1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    epochs = 50
    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("dropout_rate", dropout_rate)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("conv_type", conv_type)
        mlflow.log_param("norm_type", device)

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
            val_loss = validate(model, test_loader, criterion, device, epoch)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        mlflow.pytorch.log_model(model, "model")