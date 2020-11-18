from BioChemGNN import data
from BioChemGNN import datasets
from BioChemGNN.data import scaffold_split

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def test_delaney():
    kwargs = {'node_feature': 'chemprop', 'edge_feature': 'chemprop'}
    dataset = datasets.Delaney('./', **kwargs)

    print(len(dataset.smiles_list))
    print(dataset.target_fields)

    train_idx, val_idx, test_idx = scaffold_split(dataset)
    print('len of train: {}\tval: {}\ttest: {}'.format(len(train_idx), len(val_idx), len(test_idx)))
    return


def test_bbbp():
    kwargs = {'node_feature': 'chemprop', 'edge_feature': 'chemprop'}
    dataset = datasets.BBBP('./', **kwargs)

    print(len(dataset.smiles_list))
    print(dataset.target_fields)

    train_idx, val_idx, test_idx = scaffold_split(dataset)
    print('len of train: {}\tval: {}\ttest: {}'.format(len(train_idx), len(val_idx), len(test_idx)))
    return


if __name__ == '__main__':
    test_delaney()
    test_bbbp()
