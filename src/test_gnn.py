import torch
import pandas as pd
import mlflow.pytorch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from gnn.dataset import create_loaders
from gnn.model import GraphModel
from torch_geometric.nn import TAGConv, BatchNorm, global_mean_pool

# Ensure all random seeds are set for reproducibility # Does not ensure reproducibility
# torch.manual_seed(0)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def run_model_with_test_loader(model, test_loader, device):
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0
    
    # Collect predictions for per-class evaluation
    all_measurements = []
    all_enzymes = []
    all_y_true = []
    all_y_pred = []

    examples_printed = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing GNN"):
            batch = batch.to(device)
            outputs = model(batch)
            loss = criterion(outputs.squeeze(-1), batch.y)
            total_loss += loss.item()

            # Track all measurements and predictions
            for i in range(batch.num_graphs):
                measurement = batch.measurement_type[i]
                enzyme = batch.enzyme_name[i]
                all_measurements.append(measurement)
                all_enzymes.append(enzyme)
                all_y_true.append(batch.y[i].item())
                all_y_pred.append(outputs[i].item())

            # Print up to 5 examples
            if examples_printed < 5:
                for i in range(min(batch.num_graphs, 5 - examples_printed)):
                    tqdm.write(
                        f"SMILES: {batch.smiles[i]}, "
                        f"Type: {batch.measurement_type[i]}, "
                        f"enzyme: {batch.enzyme_name[i]}, "
                        f"Predicted: {outputs[i].item():.4f} | Actual: {batch.y[i].item():.4f}"
                    )
                    examples_printed += 1
                    if examples_printed >= 5:
                        break

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss}")

    # Perform per-class evaluation
    test_results = pd.DataFrame({
        'measurement_type': all_measurements,
        'enzyme_name': all_enzymes,
        'y_true': all_y_true,
        'y_pred': all_y_pred
    })
    grouped = test_results.groupby(['measurement_type', 'enzyme_name'])

    rows = []
    for (mtype, enzyme), group_df in grouped:
        group_loss = mean_squared_error(group_df['y_true'], group_df['y_pred'])
        mlflow.log_metric(f"test_loss_{mtype}_{enzyme}", group_loss)
        rows.append({'measurement_type': mtype, 'enzyme_name': enzyme, 'group_loss': group_loss})
    results_df = pd.DataFrame(rows)

    print("Per-group losses:")
    print(results_df)

    return avg_loss

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("GNN test loss")


    # Prepare DataFrame and extract elements for GNN feature encoding
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

    # mlflow model is not running correctly, try loading architecture explicitly
        # Define your model architecture
    dropout_rate = 0.0
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
    )

    # Load your trained GNN model from MLflow or elsewhere
    model_uri = 'runs:/076f44c8918a4fa09a0e5891dd6740f6/model'
    model = mlflow.pytorch.load_model(model_uri)
    model.load_state_dict(mlflow.pytorch.load_model(model_uri).state_dict())

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model.to(device)

    with mlflow.start_run():
        test_loss = run_model_with_test_loader(model, test_loader, device)
        mlflow.log_metric("test_loss", test_loss)
        print(f"Final Test Loss: {test_loss}")