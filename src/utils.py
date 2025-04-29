import pandas as pd
import re

def smiles_tokenizer(smi):
    """Tokenize a SMILES string into basic chemical tokens."""
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return tokens

def build_vocabulary(df, specials):
    """Build a vocab dict from the main dataframe and add special tokens."""
    vocab = {token: idx for idx, token in enumerate(specials)}
    for smi in df['SMILES']:
        tokens = smiles_tokenizer(smi)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

def create_stratified_splits(
    df,
    measurement_col='measurement_type',
    enzyme_col='enzyme_name',
    test_per_class=60,
    dev_per_class=60,
    random_state=42,
    pki_only=False,
    pki_only_dev_test=False
):
    """
    Create stratified train/dev/test splits that use an equal number of samples for each class
    Set pki_only=True to only include pKi measurements
    Set pki_only_dev_test=True to only include pKi measurements in dev/test sets
    """
    if pki_only:
        df = df[df[measurement_col] == 'pKi'].copy()

    if pki_only_dev_test:
        pki_df = df[df[measurement_col] == 'pKi'].copy()
        non_pki_df = df[df[measurement_col] != 'pKi'].copy()
    else:
        pki_df = df
        non_pki_df = pd.DataFrame()

    groups = pki_df.groupby([measurement_col, enzyme_col])
    test_list, dev_list, train_list = [], [], []

    for _, group in groups:
        shuffled = group.sample(frac=1, random_state=random_state)
        test_part = shuffled[:test_per_class]
        dev_part = shuffled[test_per_class:test_per_class + dev_per_class]
        train_part = shuffled[test_per_class + dev_per_class:]
        test_list.append(test_part)
        dev_list.append(dev_part)
        train_list.append(train_part)

    test_df = pd.concat(test_list)
    dev_df = pd.concat(dev_list)
    train_df = pd.concat(train_list + [non_pki_df])
    
    print(f"Train size: {len(train_df)}, Dev size: {len(dev_df)}, Test size: {len(test_df)}")
    return train_df, dev_df, test_df
