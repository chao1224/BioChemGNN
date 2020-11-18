import torch

from BioChemGNN import data, datasets, utils
from BioChemGNN.data import scaffold_split
from BioChemGNN.utils import Registry as R

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def test_00():
    class TestDataset(datasets.MoleculeDataset):
        def __init__(self):
            smiles_list = ['C[NH1]']
            # https://molview.org/?q=C[NH1]
            label_list = [[0 for _ in smiles_list]]
            self.load_smiles(smiles_list, label_list)

    dataset = TestDataset()
    dataloader = data.DataLoader(dataset, batch_size=128)

    for batch in dataloader:
        graph = batch['graph']
        label = batch['label']

        a_to_b, a_from_b = graph.a_to_b, graph.a_from_b
        b_to_a, b_from_a = graph.b_to_a, graph.b_from_a
        b2revb = graph.b2revb
        assert torch.equal(b2revb[a_from_b], a_to_b)
        assert torch.equal(b_from_a[b2revb], b_to_a)
        print('a_from_b\n', a_from_b.squeeze()) # in-bond for a
        print('a_to_b\n', a_to_b.squeeze()) # out-bond for a
        print('b_from_a\n', b_from_a.squeeze()) # starting atom of each bond
        print('b_to_a\n', b_to_a.squeeze()) # ending atom of each bond
        print()

        assert torch.equal(a_from_b.squeeze(), torch.LongTensor([0, 2, 1]))
        assert torch.equal(b_from_a.squeeze(), torch.LongTensor([0, 1, 2]))
        a2a = b_from_a[a_from_b]
        assert torch.equal(a2a.squeeze(), torch.LongTensor([0, 2, 1]))

        ### test node-level message passing
        atom_feature = torch.arange(3)
        assert torch.equal(atom_feature[a2a].squeeze(), torch.LongTensor([0, 2, 1]))
        atom_feature = atom_feature + torch.sum(atom_feature[a2a], dim=1)
        assert torch.equal(atom_feature.squeeze(), torch.LongTensor([0, 3, 3]))

        ### test node-level message passing
        atom_feature = torch.arange(3)
        bond_feature = torch.arange(3)
        assert torch.equal(atom_feature[b_from_a].squeeze(), torch.LongTensor([0, 1, 2]))
        bond_feature = bond_feature + atom_feature[b_from_a]
        assert torch.equal(bond_feature.squeeze(), torch.LongTensor([0, 2, 4]))

        assert torch.equal(data.index_select_ND(bond_feature, a_from_b).squeeze(), torch.LongTensor([0, 4, 2]))
        atom_feature = atom_feature + torch.sum(data.index_select_ND(bond_feature, a_from_b), dim=1)
        assert torch.equal(atom_feature.squeeze(), torch.LongTensor([0, 5, 4]))

    return


def test_01():
    class TestDataset(datasets.MoleculeDataset):
        def __init__(self):
            smiles_list = [ 'C1=C[N]C=C1'] # 'C[NH1]',
            # https://molview.org/?q=C1=C[N]C=C1
            label_list = [[0 for _ in smiles_list]]
            self.load_smiles(smiles_list, label_list)

    dataset = TestDataset()
    dataloader = data.DataLoader(dataset, batch_size=128)

    for batch in dataloader:
        graph = batch['graph']
        label = batch['label']

        a_to_b, a_from_b = graph.a_to_b, graph.a_from_b
        b_to_a, b_from_a = graph.b_to_a, graph.b_from_a
        b2revb = graph.b2revb
        assert torch.equal(b2revb[a_from_b], a_to_b)
        assert torch.equal(b_from_a[b2revb], b_to_a)
        print('a_from_b\n', a_from_b.squeeze()) # in-bond for a
        print('a_to_b\n', a_to_b.squeeze()) # out-bond for a
        print('b_from_a\n', b_from_a.squeeze()) # starting atom of each bond
        print('b_to_a\n', b_to_a.squeeze()) # ending atom of each bond
        print()

        assert torch.equal(a_from_b.squeeze(), torch.LongTensor([[0, 0], [2, 9], [1, 4], [3, 6], [5, 8], [7, 10]]))
        assert torch.equal(b_from_a.squeeze(), torch.LongTensor([0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1]))
        a2a = b_from_a[a_from_b]
        assert torch.equal(a2a.squeeze(), torch.LongTensor([[0, 0], [2, 5], [1, 3], [2, 4], [3, 5], [4, 1]]))

        ### test node-level message passing
        atom_feature = torch.arange(6)
        assert torch.equal(atom_feature[a2a].squeeze(), torch.LongTensor([[0, 0], [2, 5], [1, 3], [2, 4], [3, 5], [4, 1]]))
        atom_feature = atom_feature + torch.sum(atom_feature[a2a], dim=1)
        assert torch.equal(atom_feature.squeeze(), torch.LongTensor([0, 8, 6, 9, 12, 10]))

        ### test node-level message passing
        atom_feature = torch.arange(6)
        bond_feature = torch.arange(11)
        assert torch.equal(atom_feature[b_from_a].squeeze(), torch.LongTensor([0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1]))
        bond_feature = bond_feature + atom_feature[b_from_a]
        assert torch.equal(bond_feature.squeeze(), torch.LongTensor([0, 2, 4, 5, 7, 8, 10, 11, 13, 14, 11]))

        assert torch.equal(data.index_select_ND(bond_feature, a_from_b).squeeze(), torch.LongTensor([[0, 0], [4, 14], [2, 7], [5, 10], [8, 13], [11, 11]]))
        atom_feature = atom_feature + torch.sum(data.index_select_ND(bond_feature, a_from_b), dim=1)
        assert torch.equal(atom_feature.squeeze(), torch.LongTensor([0, 19, 11, 18, 25, 27]))
    return


def test_02():
    class TestDataset(datasets.MoleculeDataset):
        def __init__(self):
            smiles_list = ['[Br-].[Na+]']
            # https://molview.org/?q=C1=C[N]C=C1
            label_list = [[0 for _ in smiles_list]]
            self.load_smiles(smiles_list, label_list)

    dataset = TestDataset()
    dataloader = data.DataLoader(dataset, batch_size=128)


if __name__ == '__main__':
    test_00()
    test_01()
    test_02()
