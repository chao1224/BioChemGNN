import os
import math
import csv
from tqdm import tqdm
from collections import defaultdict

import torch
from BioChemGNN import data, utils
from BioChemGNN.utils import Registry as R


class MoleculeDataset(torch.utils.data.Dataset):
    def load_smiles(self, smiles_list, label_list, transform=None, verbose=0, **kwargs):
        self.smiles_list, self.label_list, self.data_list = [], [], []
        self.graph_list = []
        if verbose:
            smiles_list = tqdm(smiles_list, 'Constructing molecules from SMILES')
        for smiles, label in zip(smiles_list, label_list):
            graph = data.MoleculeGraph.from_smiles(smiles, **kwargs)
            if not graph:
                continue
            self.graph_list.append(graph)
            self.smiles_list.append(smiles)
            self.label_list.append(label)
            self.data_list.append(graph)
        self.transform = transform

    def load_csv(self, csv_file, smiles_field='smiles', target_fields=None, transform=None, verbose=0, **kwargs):
        if target_fields is not None:
            target_fields = set(target_fields)

        smiles, labels = [], []
        with open(csv_file, 'r') as fin:
            reader = csv.reader(fin)
            if verbose:
                reader = iter(tqdm(reader, 'Loading %s' % csv_file, utils.get_line_count(csv_file)))
            fields = next(reader)
            for values in reader:
                if not any(values):
                    continue
                label_row = []
                for field, value in zip(fields, values):
                    if field == smiles_field:
                        smiles.append(value)
                    elif field in target_fields:
                        value = utils.literal_eval(value)
                        if value == "":
                            value = math.nan
                        label_row.append(value)
                labels.append(label_row)
        self.load_smiles(smiles, labels, verbose=verbose, transform=transform, **kwargs)

    def __getitem__(self, index):
        data = self.data_list[index]
        label = self.label_list[index]
        item = {'graph': data, 'label': label}
        if self.transform:
            item = self.transform(item)
        return item

    def tasks(self):
        return list(self.targets.keys())

    @property
    def node_feature_dim(self):
        return self.data_list[0].atom_feature_dim

    @property
    def edge_feature_dim(self):
        return self.data_list[0].bond_feature_dim

    def num_atom_type(self):
        return len(self.atom_types)

    def num_bond_type(self):
        return len(self.bond_types)

    def atom_types(self):
        if not hasattr(self, '_atom_types'):
            atom_types = set()
            for graph in self.data_list:
                atom_types.update(graph.atom_type.tolist())
            self._atom_types = sorted(atom_types)
        return self._atom_types

    def bond_types(self):
        if not hasattr(self, '_bond_types'):
            bond_types = set()
            for graph in self.data_list:
                bond_types.update(graph.bond_type.tolist())
            self._bond_types = sorted(bond_types)
        return self._bond_types

    def __len__(self):
        return len(self.label_list)


class InMemoryMoleculeDataset(MoleculeDataset):
    def get(self, idx):
        graph = data.MoleculeGraph()
        atom_idx, atom_size = self.collated_data.atom_scope[idx]
        bond_idx, bond_size = self.collated_data.bond_scope[idx]

        graph.atom_num = atom_size.tolist()
        graph.bond_num = bond_size.tolist()

        graph.atom_feat_list = self.collated_data.atom_feat_list[atom_idx: atom_idx+atom_size].tolist()
        a_to_b = self.collated_data.a_to_b[atom_idx: atom_idx+atom_size].tolist()
        a_from_b = self.collated_data.a_from_b[atom_idx: atom_idx+atom_size].tolist()
        a_to_b = [list(filter(lambda bid: bid >= 0, bond_idx_list)) for bond_idx_list in a_to_b]
        a_from_b = [list(filter(lambda bid: bid >= 0, bond_idx_list)) for bond_idx_list in a_from_b]

        graph.a_to_b = a_to_b
        graph.a_from_b = a_from_b

        graph.bond_feat_list = self.collated_data.bond_feat_list[bond_idx: bond_idx+bond_size].tolist()
        graph.b_from_a = self.collated_data.b_from_a[bond_idx: bond_idx+bond_size].tolist()
        graph.b_to_a = self.collated_data.b_to_a[bond_idx: bond_idx+bond_size].tolist()
        graph.b2revb = self.collated_data.b2revb[bond_idx: bond_idx+bond_size].tolist()

        return graph

    @staticmethod
    def collate(graph_list):
        packed_graph = data.MoleculeGraph()

        packed_graph.atom_num = 0
        packed_graph.bond_num = 0
        packed_graph.atom_scope = []
        packed_graph.bond_scope = []
        packed_graph.small_molecule_graph_num = 0
        atom_num_list = []
        atom_feat_list = []
        bond_feat_list = []
        a_to_b, a_from_b = [], []
        b_to_a, b_from_a = [], []
        b2revb = []

        for graph in graph_list:
            atom_num_list.append(graph.atom_num)
            atom_feat_list.extend(graph.atom_feat_list)
            bond_feat_list.extend(graph.bond_feat_list)

            for a in range(graph.atom_num):
                a_to_b.append([b for b in graph.a_to_b[a]])
                a_from_b.append([b for b in graph.a_from_b[a]])

            b_to_a.extend([b for b in graph.b_to_a])
            b_from_a.extend([b for b in graph.b_from_a])
            b2revb.extend([b for b in graph.b2revb])

            packed_graph.atom_scope.append([packed_graph.atom_num, graph.atom_num])
            packed_graph.bond_scope.append([packed_graph.bond_num, graph.bond_num])
            packed_graph.atom_num += graph.atom_num
            packed_graph.bond_num += graph.bond_num
            packed_graph.small_molecule_graph_num += 1

        packed_graph.atom_num_list = torch.LongTensor(atom_num_list)
        packed_graph.atom_feat_list = torch.FloatTensor(atom_feat_list)
        packed_graph.bond_feat_list = torch.FloatTensor(bond_feat_list)

        packed_graph.max_bond_per_node = max(1, max(len(in_bonds) for in_bonds in a_from_b))
        packed_graph.a_to_b = torch.LongTensor([a_to_b[a] + [-1] * (packed_graph.max_bond_per_node - len(a_to_b[a])) for a in range(packed_graph.atom_num)])
        packed_graph.a_from_b = torch.LongTensor([a_from_b[a] + [-1] * (packed_graph.max_bond_per_node - len(a_from_b[a])) for a in range(packed_graph.atom_num)])
        packed_graph.b_to_a = torch.LongTensor(b_to_a)
        packed_graph.b_from_a = torch.LongTensor(b_from_a)
        packed_graph.b2revb = torch.LongTensor(b2revb)
        packed_graph.atom_scope = torch.LongTensor(packed_graph.atom_scope)
        packed_graph.bond_scope = torch.LongTensor(packed_graph.bond_scope)

        return packed_graph

    def __getitem__(self, index):
        data = self.get(index)
        label = self.label_list[index]
        item = {'graph': data, 'label': label}
        if self.transform:
            item = self.transform(item)
        return item


class MoleculeECFPDataset(MoleculeDataset):
    def __init__(self, dataset):
        self.data_list = [torch.LongTensor(g.graph_feature) for g in dataset.graph_list]
        self.label_list = [label for label in dataset.label_list]
        self.transform = dataset.transform