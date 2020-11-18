from rdkit import Chem
from rdkit.Chem import AllChem, Fragments
import numpy as np

from BioChemGNN.utils import Registry as R


atom_candidates = ['C', 'Cl', 'I', 'F', 'O', 'N', 'P', 'S', 'Br', 'Unknown']
atom_vocab = ['H', 'B', 'C', 'N', 'O', 'F', 'Mg', 'Si', 'P', 'S', 'Cl', 'Cu', 'Zn', 'Se', 'Br', 'Sn', 'I']
atom_vocab = {a: i for i, a in enumerate(atom_vocab)}
degree_vocab = range(7)
num_hs_vocab = range(7)
formal_charge_vocab = range(-5, 6)
total_valence_vocab = range(8)
num_radical_vocab = range(8)
hybridization_vocab = range(len(Chem.rdchem.HybridizationType.values))

bond_type_vocab = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                   Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
bond_dir_vocab = range(len(Chem.rdchem.BondDir.values))
stereo_vocab = range(len(Chem.rdchem.BondStereo.values))


def onehot(x, vocab, allow_unknown=False):
    if x in vocab:
        if isinstance(vocab, dict):
            index = vocab[x]
        else:
            index = vocab.index(x)
    else:
        index = -1
    if allow_unknown:
        feature = [0] * (len(vocab) + 1)
        if index == -1:
            print("Unknown value `%s`" % x)
        feature[index] = 1
    else:
        feature = [0] * len(vocab)
        if index == -1:
            raise ValueError("Unknown value `%s`. Available vocabulary is `%s`" % (x, vocab))
        feature[index] = 1

    return feature


def atom_position(atom):
    mol = atom.GetOwningMol()
    if mol.GetNumConformers() == 0:
        mol.Compute2DCoords()
    conformer = mol.GetConformer()
    pos = conformer.GetAtomPosition(atom.GetIdx())
    return list(pos)


@R.register('features.atom.default')
def atom_default(atom):
    atom_feature = \
        onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
        onehot(atom.GetChiralTag(), [0, 1, 2, 3]) + \
        onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True) + \
        onehot(atom.GetFormalCharge(), formal_charge_vocab) + \
        onehot(atom.GetTotalNumHs(), num_hs_vocab) + \
        onehot(atom.GetNumRadicalElectrons(), num_radical_vocab) + \
        onehot(atom.GetHybridization(), hybridization_vocab) + \
        [atom.GetIsAromatic(), atom.IsInRing()] + \
        atom_position(atom)

    atom_feature = np.array(atom_feature)
    return atom_feature


@R.register('features.atom.chemprop')
def atom_default(atom):
    MAX_ATOMIC_NUM = 100
    ATOM_FEATURES = {
        'atomic_num': list(range(MAX_ATOMIC_NUM)),
        'degree': [0, 1, 2, 3, 4, 5],
        'formal_charge': [-1, -2, 1, 2, 0],
        'chiral_tag': [0, 1, 2, 3],
        'num_Hs': [0, 1, 2, 3, 4],
        'hybridization': [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ],
    }
    atom_feature = \
        onehot(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num'], allow_unknown=True) + \
        onehot(atom.GetTotalDegree(), ATOM_FEATURES['degree'], allow_unknown=True) + \
        onehot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'], allow_unknown=True) + \
        onehot(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag'], allow_unknown=True) + \
        onehot(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs'], allow_unknown=True) + \
        onehot(int(atom.GetHybridization()), ATOM_FEATURES['hybridization'], allow_unknown=True) + \
        [1 if atom.GetIsAromatic() else 0] + \
        [atom.GetMass() * 0.01]

    atom_feature = np.array(atom_feature)
    return atom_feature


@R.register("features.bond.length")
def bond_length(bond):
    mol = bond.GetOwningMol()
    if mol.GetNumConformers() == 0:
        mol.Compute2DCoords()
    conformer = mol.GetConformer()
    h = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
    t = conformer.GetAtomPosition(bond.GetEndAtomIdx())
    return [h.Distance(t)]


@R.register('features.bond.default')
def bond_default(bond):
    bt = bond.GetBondType()
    bond_feature = \
        onehot(bond.GetBondType(), bond_type_vocab) + \
        [bond.GetIsConjugated() if bt is not None else 0] + \
        [bond.IsInRing() if bt is not None else 0] + \
        onehot(bond.GetBondDir(), bond_dir_vocab) + \
        onehot(bond.GetStereo(), stereo_vocab) + \
        bond_length(bond)
    return bond_feature


@R.register('features.bond.chemprop')
def bond_default(bond):
    bt = bond.GetBondType()
    bond_feature = [
        0,
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        (bond.GetIsConjugated() if bt is not None else 0),
        (bond.IsInRing() if bt is not None else 0)
    ] + onehot(bond.GetStereo(), stereo_vocab)
    return bond_feature


def get_atom_feature_dim(atom_feature_func):
    simple_mol = Chem.MolFromSmiles('CC')
    atom_feature_dim = len(atom_feature_func(simple_mol.GetAtoms()[0]))
    return atom_feature_dim


def get_bond_feature_dim(bond_feature_func):
    simple_mol = Chem.MolFromSmiles('CC')
    bond_feature_dim = len(bond_feature_func(simple_mol.GetBonds()[0]))
    return bond_feature_dim


def get_graph_feature_dim(graph_feature_func):
    simple_mol = Chem.MolFromSmiles('CC')
    graph_feature_dim = len(graph_feature_func(simple_mol))
    return graph_feature_dim


@R.register('features.molecule.ecfp')
@R.register('features.molecule.default')
def ECFP(mol, radius=2, length=1024):
    '''Extended Connectivity Fingerprint graph feature.'''
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, length)
    return list(ecfp)
