import torch
from torch.utils.data import Dataset, DataLoader
from utils import smiles_tokenizer, build_vocabulary,  create_stratified_splits
import pandas as pd
from rdkit import Chem
from functools import partial


class SMILESDataset(Dataset):
    """PyTorch Dataset for SMILES strings, prepended by measurement type and enzyme name."""
    def __init__(self, dataframe, vocab):
        self.dataframe = dataframe
        self.vocab = vocab

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        smiles = row['SMILES']
        measurement_type = row['measurement_type']
        enzyme_name = row['enzyme_name']
        measurement_value = row['measurement_value']

        tokens = smiles_tokenizer(smiles)
        token_ids = [self.vocab[token] for token in tokens]
        token_ids = [self.vocab[measurement_type], self.vocab[enzyme_name]] + token_ids

        return torch.tensor(token_ids), torch.tensor(measurement_value)

def collate_fn(batch):
    input_ids, targets = zip(*batch)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    return input_ids, torch.stack(targets)

# Inelegant way to randomize SMILES strings to work around existing codebase
df = pd.read_csv("./data.csv")
specials = ["<PAD>", "pKi", "pIC50", "ENZ1", "ENZ2", "ENZ3", "ENZ4"]
vocab = build_vocabulary(df, specials)
inverted_vocab = {v: k for k, v in vocab.items()}

def collate_fn_randomize(batch, vocab, inverted_vocab):
    new_input_ids = []
    new_targets = []

    for input_ids_tensor, target_tensor in batch:
        # Convert input IDs to list for processing
        input_ids_list = input_ids_tensor.tolist()

        # The first two tokens are measurement_type and enzyme_name
        measurement_type_id = input_ids_list[0]
        enzyme_name_id = input_ids_list[1]

        # Extract SMILES token IDs (skipping the first two)
        smiles_token_ids = input_ids_list[2:]

        # Detokenize
        smiles_tokens = [inverted_vocab[t_id] for t_id in smiles_token_ids if t_id != 0]
        original_smiles = "".join(smiles_tokens)

        # Randomize with RDKit
        mol = Chem.MolFromSmiles(original_smiles)
        if mol is not None:
            randomized_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        else:
            randomized_smiles = original_smiles

        # Retokenize
        randomized_tokens = smiles_tokenizer(randomized_smiles)

        # Reconstruct input IDs (prepend measurement_type and enzyme_name IDs)
        new_token_ids = [measurement_type_id, enzyme_name_id] + [vocab.get(tok, 0) for tok in randomized_tokens]
        new_input_ids.append(torch.tensor(new_token_ids))
        new_targets.append(target_tensor)

    # Pad for batch
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(new_input_ids, batch_first=True, padding_value=0)
    targets_tensor = torch.stack(new_targets)

    return padded_input_ids, targets_tensor

def create_loaders(df, vocab, batch_size, randomize_smiles=False):
    train_df, dev_df, test_df = create_stratified_splits(
        df,
        measurement_col='measurement_type',
        enzyme_col='enzyme_name',
        test_per_class=60,
        dev_per_class=60,
        pki_only=False,
        pki_only_dev_test=True
    )

    # Create PyTorch datasets and loaders
    train_dataset = SMILESDataset(train_df, vocab)
    dev_dataset = SMILESDataset(dev_df, vocab)
    test_dataset = SMILESDataset(test_df, vocab)

    if randomize_smiles:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_fn_randomize, vocab=vocab, inverted_vocab=inverted_vocab))
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader