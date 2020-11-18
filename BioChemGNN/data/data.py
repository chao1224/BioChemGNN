from rdkit import Chem

import torch
from BioChemGNN import data
from BioChemGNN.utils import Registry as R


class MoleculeGraph:
    def __init__(self, mol=None, smiles=None,
                 node_feature='default', edge_feature='default', graph_feature='default'):

        self.smiles = smiles
        atom_feature_func = R.get('features.atom.{}'.format(node_feature))
        bond_feature_func = R.get('features.bond.{}'.format(edge_feature))
        graph_feature_func = R.get('features.molecule.{}'.format(graph_feature))

        if mol is None:
            self.atom_feature_dim = data.get_atom_feature_dim(atom_feature_func)
            self.bond_feature_dim = data.get_bond_feature_dim(bond_feature_func)
            self.graph_feature_dim = data.get_graph_feature_dim(graph_feature_func)

            self.atom_num = 1
            self.bond_num = 1
            self.atom_feat_list = [[0 for _ in range(self.atom_feature_dim)]]
            self.bond_feat_list = [[0 for _ in range(self.bond_feature_dim)]]
            self.graph_feature = [[0 for _ in range(self.graph_feature_dim)]]

            self.a_to_b, self.a_from_b = [[]], [[]]
            self.b_to_a, self.b_from_a = [0], [0]
            self.b2revb = [0]
        else:
            self.atom_feat_list = [atom_feature_func(atom) for atom in mol.GetAtoms()]
            self.atom_num = len(self.atom_feat_list)

            self.bond_num = 0
            self.bond_feat_list = []
            self.a_from_b = [[] for _ in range(self.atom_num)]
            self.a_to_b = [[] for _ in range(self.atom_num)]
            self.b_from_a = []
            self.b_to_a = []
            self.b2revb = []

            for bond in mol.GetBonds():
                atom_index_1 = bond.GetBeginAtomIdx()
                atom_index_2 = bond.GetEndAtomIdx()
                bond_feature = bond_feature_func(bond)

                self.bond_feat_list.append(bond_feature)
                self.bond_feat_list.append(bond_feature)

                # bond 1: atom 1 -> atom 2
                # bond 2: atom 2 -> atom 1

                bond_index_1 = self.bond_num
                bond_index_2 = bond_index_1 + 1

                self.a_to_b[atom_index_1].append(bond_index_1)
                self.a_from_b[atom_index_2].append(bond_index_1)
                self.a_to_b[atom_index_2].append(bond_index_2)
                self.a_from_b[atom_index_1].append(bond_index_2)

                self.b_to_a.append(atom_index_2)
                self.b_from_a.append(atom_index_1)
                self.b_to_a.append(atom_index_1)
                self.b_from_a.append(atom_index_2)

                self.b2revb.append(bond_index_2)
                self.b2revb.append(bond_index_1)

                self.bond_num += 2

            self.graph_feature = graph_feature_func(mol)

            self.atom_feature_dim = len(self.atom_feat_list[0])
            if len(self.bond_feat_list) == 0:
                self.bond_feature_dim = data.get_bond_feature_dim(bond_feature_func)
            else:
                self.bond_feature_dim = len(self.bond_feat_list[0])

        return

    @classmethod
    def from_smiles(cls, smiles, node_feature='default', edge_feature='default', graph_feature='default',
                    with_hydrogen=False, kekulize=False):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print('Invalid SMILES `%s`' % smiles)
            return None
        return cls.from_molecule(mol, smiles, node_feature, edge_feature, graph_feature, with_hydrogen, kekulize)

    @classmethod
    def from_molecule(cls, mol, smiles, node_feature='default', edge_feature='default', graph_feature='default',
                      with_hydrogen=False, kekulize=False):
        if with_hydrogen:
            mol = Chem.AddHs(mol)
        if kekulize:
            Chem.Kekulize(mol)
        self = cls(mol, smiles, node_feature=node_feature, edge_feature=edge_feature, graph_feature=graph_feature)
        return self

    @classmethod
    def pack(cls, graph_list):
        return PackedMoleculeGraph(graph_list)

    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    @property
    def keys(self):
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        return keys

    @property
    def device(self):
        return self.atom_feat_list.device

    def cuda(self, *args, **kwargs):
        atom_feat_list = self.atom_feat_list.cuda(*args, **kwargs)
        bond_feat_list = self.bond_feat_list.cuda(*args, **kwargs)
        a_from_b = self.a_from_b.cuda(*args, **kwargs)
        a_to_b = self.a_to_b.cuda(*args, **kwargs)
        b_from_a = self.b_from_a.cuda(*args, **kwargs)
        b_to_a = self.b_to_a.cuda(*args, **kwargs)
        return type(self)(atom_feat_list, bond_feat_list, a_to_b, a_from_b, b_to_a, b_from_a)

    def __repr__(self):
        out_str = ''
        if hasattr(self, 'smiles'):
            out_str = 'SMILES: {}'.format(self.smiles)
        if not torch.is_tensor(self.atom_feat_list):
            out_str = '{}\nnum atom: {}\tnum bond: {}'.format(out_str, self.atom_num, self.bond_num)
        else:
            out_str = '{}\nnum atom: {}\tatom feat: {}\ta_to_b: {}\tnum bond: {}\tbond feat: {}\tb_to_a: {}'.format(
                out_str, self.atom_num, self.atom_feat_list.size(), self.a_to_b.size(), self.bond_num, self.bond_feat_list.size(), self.b_to_a.size()
            )
        return out_str


class PackedMoleculeGraph(MoleculeGraph):
    def __init__(self, graph_list):
        self.atom_feat_dim = graph_list[0].atom_feature_dim
        self.bond_feat_dim = graph_list[0].bond_feature_dim

        self.atom_num = 1
        self.bond_num = 1
        self.atom_scope = []
        self.bond_scope = []
        self.small_molecule_graph_num = 0
        atom_num_list = [1]
        atom_feat_list = [[0 for _ in range(self.atom_feat_dim)]]
        bond_feat_list = [[0 for _ in range(self.bond_feat_dim)]]
        a_to_b, a_from_b = [[]], [[]]
        b_to_a, b_from_a = [0], [0]
        b2revb = [0]

        for graph in graph_list:
            atom_num_list.append(graph.atom_num)
            atom_feat_list.extend(graph.atom_feat_list)
            bond_feat_list.extend(graph.bond_feat_list)

            for a in range(graph.atom_num):
                a_to_b.append([b + self.bond_num for b in graph.a_to_b[a]])
                a_from_b.append([b + self.bond_num for b in graph.a_from_b[a]])

            b_to_a.extend([b + self.atom_num for b in graph.b_to_a])
            b_from_a.extend([b + self.atom_num for b in graph.b_from_a])
            b2revb.extend([b + self.bond_num for b in graph.b2revb])

            self.atom_scope.append([self.atom_num, graph.atom_num])
            self.bond_scope.append([self.bond_num, graph.bond_num])
            self.atom_num += graph.atom_num
            self.bond_num += graph.bond_num
            self.small_molecule_graph_num += 1

        self.atom_num_list = torch.LongTensor(atom_num_list)
        self.atom_feat_list = torch.FloatTensor(atom_feat_list)
        self.bond_feat_list = torch.FloatTensor(bond_feat_list)

        self.max_bond_per_node = max(1, max(len(in_bonds) for in_bonds in a_from_b))
        self.a_to_b = torch.LongTensor([a_to_b[a] + [0] * (self.max_bond_per_node - len(a_to_b[a])) for a in range(self.atom_num)])
        self.a_from_b = torch.LongTensor([a_from_b[a] + [0] * (self.max_bond_per_node - len(a_from_b[a])) for a in range(self.atom_num)])
        self.b_to_a = torch.LongTensor(b_to_a)
        self.b_from_a = torch.LongTensor(b_from_a)
        self.b2revb = torch.LongTensor(b2revb)
        self.atom_scope = torch.LongTensor(self.atom_scope)
        self.bond_scope = torch.LongTensor(self.bond_scope)

        return

    def _init_determined(self, atom_feat_list, bond_feat_list, a_to_b, a_from_b, b_to_a, b_from_a, b2revb,
                         max_bond_per_node, atom_scope, bond_scope, atom_num_list):
        self.atom_feat_list = atom_feat_list
        self.bond_feat_list = bond_feat_list
        self.a_to_b = a_to_b
        self.a_from_b = a_from_b
        self.b_to_a = b_to_a
        self.b_from_a = b_from_a
        self.b2revb = b2revb
        self.max_bond_per_node = max_bond_per_node
        self.atom_scope = atom_scope
        self.bond_scope = bond_scope
        self.atom_num_list = atom_num_list

    def node_mask(self):
        raise NotImplementedError

    def graph_mask(self):
        raise NotImplementedError

    def cuda(self, *args, **kwargs):
        # atom_feat_list = self.atom_feat_list.cuda(*args, **kwargs)
        # bond_feat_list = self.bond_feat_list.cuda(*args, **kwargs)
        # a_to_b = self.a_to_b.cuda(*args, **kwargs)
        # a_from_b = self.a_from_b.cuda(*args, **kwargs)
        # b_to_a = self.b_to_a.cuda(*args, **kwargs)
        # b_from_a = self.b_from_a.cuda(*args, **kwargs)
        # b2revb = self.b2revb.cuda(*args, **kwargs)
        # max_bond_per_node = self.max_bond_per_node
        # atom_scope = self.atom_scope.cuda(*args, **kwargs)
        # bond_scope = self.bond_scope.cuda(*args, **kwargs)
        # atom_num_list = self.atom_num_list.cuda(*args, **kwargs)

        self.atom_feat_list = self.atom_feat_list.cuda(*args, **kwargs)
        self.bond_feat_list = self.bond_feat_list.cuda(*args, **kwargs)
        self.a_to_b = self.a_to_b.cuda(*args, **kwargs)
        self.a_from_b = self.a_from_b.cuda(*args, **kwargs)
        self.b_to_a = self.b_to_a.cuda(*args, **kwargs)
        self.b_from_a = self.b_from_a.cuda(*args, **kwargs)
        self.b2revb = self.b2revb.cuda(*args, **kwargs)
        self.max_bond_per_node = self.max_bond_per_node
        self.atom_scope = self.atom_scope.cuda(*args, **kwargs)
        self.bond_scope = self.bond_scope.cuda(*args, **kwargs)
        self.atom_num_list = self.atom_num_list.cuda(*args, **kwargs)
        return self

    def to(self, device, *keys, **kwargs):
        return self.apply(lambda x: x.to(device, **kwargs), *keys)


def index_select_ND(source, index):
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    return target
