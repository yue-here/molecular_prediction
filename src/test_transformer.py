import torch
import pandas as pd
import mlflow.pytorch
from tqdm import tqdm
from transformer.dataset import create_loaders
from utils import build_vocabulary
import torch.nn as nn
from sklearn.metrics import mean_squared_error

def run_model_with_test_loader(model, test_loader, device, invert_vocab):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0

    # Collect predictions for per-class evaluation
    all_measurements = []
    all_enzymes = []
    all_y_true = []
    all_y_pred = []

    examples_printed = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running Model on Test Data"):
            input_ids, targets = batch
            input_ids, targets = input_ids.to(device), targets.to(device, dtype=torch.float32)
            outputs = model(input_ids)
            loss = criterion(outputs.squeeze(-1), targets)
            total_loss += loss.item()

            # Track all measurements and predictions
            for i in range(input_ids.size(0)):
                measurement = invert_vocab.get(input_ids[i, 0].item(), "?")
                enzyme = invert_vocab.get(input_ids[i, 1].item(), "?")
                all_measurements.append(measurement)
                all_enzymes.append(enzyme)
                all_y_true.append(targets[i].item())
                all_y_pred.append(outputs[i].item())

            # Print up to 5 examples
            if examples_printed < 5:
                for i in range(min(input_ids.size(0), 5 - examples_printed)):
                    ids_cpu = input_ids[i].cpu()
                    measurement = invert_vocab.get(ids_cpu[0].item(), "?")
                    enzyme = invert_vocab.get(ids_cpu[1].item(), "?")
                    smi = [invert_vocab.get(tok.item(), "?") for tok in ids_cpu[2:] if tok.item() != 0]
                    predicted = outputs[i].item()
                    actual = targets[i].item()
                    tqdm.write(
                        f"Measurement: {measurement} | enzyme: {enzyme} | SMILES: {''.join(smi)} | "
                        f"Predicted: {predicted:.4f} | Actual: {actual:.4f}"
                    )
                    examples_printed += 1
                    if examples_printed >= 5:
                        break

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss}")

    # Perform per-class evaluation (similar to test_baseline)
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
    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Transformer test loss")

    # Prepare dataloaders
    df = pd.read_csv('./enzyme_JAK.csv')
    specials = ['<PAD>', 'pKi', 'pIC50', 'JAK1', 'JAK2', 'JAK3', 'TYK2']
    vocab = build_vocabulary(df, specials)
    pad_token_idx = vocab['<PAD>']
    batch_size = 256
    train_loader, dev_loader, test_loader = create_loaders(df, vocab, batch_size)

    # Load model from MLflow
    model_uri = 'runs:/f6ef772fad0346daacd081280392cc12/model_epoch_100'
    model = mlflow.pytorch.load_model(model_uri)

    # Set device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model.to(device)

    # Inverted vocabulary
    inverted_vocab = {v: k for k, v in vocab.items()}

    # Run model with test loader and calculate loss
    test_loss = run_model_with_test_loader(model, test_loader, device, inverted_vocab)
    mlflow.log_metric("test_loss", test_loss)
    print(f"Final Test Loss: {test_loss}")