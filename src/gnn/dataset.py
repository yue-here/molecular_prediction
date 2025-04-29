import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
from rdkit import Chem
from sklearn.preprocessing import OneHotEncoder
from utils import create_stratified_splits

class GraphDataset(Dataset):
    def __init__(self, dataframe, parent_dataframe, all_elements):
        self.dataframe = dataframe
        self.all_elements = all_elements
        self.element_to_idx = {elem: i for i, elem in enumerate(self.all_elements)}

        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(parent_dataframe[['measurement_type', 'enzyme_name']])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        measurement_type = row['measurement_type']
        enzyme_name = row['enzyme_name']
        measurement_value = row['measurement_value']

        cat_features = self.encoder.transform(pd.DataFrame([[measurement_type, enzyme_name]], columns=['measurement_type', 'enzyme_name']))[0] # to suppress warning
        cat_features = torch.tensor(cat_features, dtype=torch.float)

        smiles = row['SMILES']
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        # Combine one-hot features for each atom
        features_list = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            elem_onehot = [0.0] * len(self.all_elements)
            if symbol in self.element_to_idx:
                elem_onehot[self.element_to_idx[symbol]] = 1.0
            features_list.append(elem_onehot)

        x = torch.tensor(features_list, dtype=torch.float)
        edge_index = torch.tensor(
            [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()],
            dtype=torch.long
        ).t().contiguous()
        y = torch.tensor([measurement_value], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.measurement_type = measurement_type
        data.enzyme_name = enzyme_name
        data.smiles = smiles
        data.cat_features = cat_features
        return data

def create_loaders(df, batch_size, elements):
    train_df, dev_df, test_df = create_stratified_splits(
        df,
        measurement_col='measurement_type',
        enzyme_col='enzyme_name',
        test_per_class=60,
        dev_per_class=60,
        pki_only=False,
        pki_only_dev_test=True
    )

    # Convert each DataFrame to a dataset
    train_dataset = GraphDataset(train_df, df, elements)
    dev_dataset = GraphDataset(dev_df, df, elements)
    test_dataset = GraphDataset(test_df, df, elements)

    # Create DataLoaders
    train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = GeometricDataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = GeometricDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, dev_loader, test_loader