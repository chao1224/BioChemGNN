from BioChemGNN import data, datasets, utils
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

    dataloader = data.DataLoader(dataset, batch_size=128)

    for batch in dataloader:
        print(batch.keys())
        batch = utils.cuda(batch)
        graph = batch['graph']
        label = batch['label']
        print(label.size())

        print(dir(graph))

        a2b, b2a, b2revb = graph.a_to_b, graph.b_to_a, graph.b2revb
        atom_feature, bond_feature = graph.atom_feat_list, graph.bond_feat_list

        print('a2b: {}\tb2a: {}\tb2revb {}'.format(a2b.size(), b2a.size(), b2revb.size()))
        print('atom feature\t', atom_feature.size())
        print('bond feature\t', bond_feature.size())

        break
    return


if __name__ == '__main__':
    test_delaney()
