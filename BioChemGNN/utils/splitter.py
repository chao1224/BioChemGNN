import random
from rdkit.Chem.Scaffolds import MurckoScaffold


def random_split(dataset, split_seed, split_size=(0.8, 0.1, 0.1)):
    indices = list(range(len(dataset)))
    random.seed(split_seed)
    random.shuffle(indices)

    train_size = int(split_size[0] * len(dataset))
    train_val_size = int((split_size[0] + split_size[1]) * len(dataset))

    train_idx = [i for i in indices[:train_size]]
    valid_idx = [i for i in indices[train_size:train_val_size]]
    test_idx = [i for i in indices[train_val_size:]]

    return train_idx, valid_idx, test_idx


def generate_scaffold(smiles, include_chirality=False):
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


def scaffold_split(dataset, split_size=(0.8, 0.1, 0.1), include_chirality=True):
    frac_train, frac_valid, frac_test = split_size
    assert frac_train + frac_valid + frac_test == 1.

    smiles_list = dataset.smiles_list
    all_scaffolds = {}
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    return train_idx, valid_idx, test_idx