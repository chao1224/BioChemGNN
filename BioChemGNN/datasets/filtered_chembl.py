from BioChemGNN import data, datasets, utils
from BioChemGNN.utils import Registry as R
import os
import pickle
import time
from itertools import chain, product
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import torch


@R.register('datasets/FilteredChEMBL')
class FilteredChEMBL(datasets.InMemoryMoleculeDataset):
    '''
    mkdir -p ./datasets
    cd datasets
    wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced.zip
    cd dataPythonReduced
    wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20Smiles.pckl
    wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20LSTM.pckl
    cd ..
    rm dataPythonReduced.zip
    cd ..
    '''

    def __init__(self, path, with_hydrogen=False, kekulize=False, verbose=True, transform=None, reload=True, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = '{}/dataPythonReduced'.format(path)
        self.transform = transform

        data_smiles_list, data_list, label_list = [], [], []


        filename = '{}/filtered_chembl_processed_graph.pt'.format(path)
        start_time = time.time()
        if (not reload) or (not os.path.exists(filename)):
            smiles_list, rdkit_mol_objs, folds, labels = load_chembl_with_labels_dataset(self.path)

            print('processing')
            for i in tqdm(range(len(rdkit_mol_objs))):
                rdkit_mol = rdkit_mol_objs[i]
                rdkit_smiles = smiles_list[i]
                if rdkit_mol != None:
                    mw = Descriptors.MolWt(rdkit_mol)
                    if 50 <= mw <= 900:
                        inchi = create_standardized_mol_id(smiles_list[i])
                        if inchi != None:
                            graph = data.MoleculeGraph.from_molecule(mol=rdkit_mol, smiles=rdkit_smiles, **kwargs)
                            data_list.append(graph)
                            data_smiles_list.append(smiles_list[i])
                            label_list.append(labels[i, :])

            self.label_list = np.array(label_list)
            self.label_list[self.label_list == 0] = -1
            self.label_list = np.nan_to_num(self.label_list)
            self.label_list = torch.FloatTensor(self.label_list)

            self.collated_data = self.collate(data_list)
            torch.save((self.collated_data, self.label_list), filename)

            # self.data_list = [self.get(idx) for idx in range(len(self))]
        else:
            self.collated_data, self.label_list = torch.load(filename)
            # self.data_list = [self.get(idx) for idx in range(len(self))]
        print('processing takes {}'.format(time.time() - start_time))
        return


def load_chembl_with_labels_dataset(path):
    '''
    Credit to https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/loader.py
    '''
    f = open(os.path.join(path, 'folds0.pckl'), 'rb')
    folds = pickle.load(f)
    f.close()

    f = open(os.path.join(path, 'labelsHard.pckl'), 'rb')
    targetMat = pickle.load(f)
    sampleAnnInd = pickle.load(f)
    targetAnnInd = pickle.load(f)
    f.close()

    targetMat = targetMat
    targetMat = targetMat.copy().tocsr()
    targetMat.sort_indices()
    targetAnnInd = targetAnnInd
    targetAnnInd = targetAnnInd - targetAnnInd.min()

    folds = [np.intersect1d(fold, sampleAnnInd.index.values).tolist() for fold in folds]
    targetMatTransposed = targetMat[sampleAnnInd[list(chain(*folds))]].T.tocsr()
    targetMatTransposed.sort_indices()
    # # num positive examples in each of the 1310 targets
    trainPosOverall = np.array(
        [np.sum(targetMatTransposed[x].data > 0.5) for x in range(targetMatTransposed.shape[0])])
    # # num negative examples in each of the 1310 targets
    trainNegOverall = np.array(
        [np.sum(targetMatTransposed[x].data < -0.5) for x in range(targetMatTransposed.shape[0])])
    # dense array containing the labels for the 456331 molecules and 1310 targets
    denseOutputData = targetMat.A  # possible values are {-1, 0, 1}

    # 2. load structures
    f = open(os.path.join(path, 'chembl20LSTM.pckl'), 'rb')
    rdkitArr = pickle.load(f)
    f.close()

    assert len(rdkitArr) == denseOutputData.shape[0]
    assert len(rdkitArr) == len(folds[0]) + len(folds[1]) + len(folds[2])

    preprocessed_rdkitArr = []
    print('preprocessing')
    for i in tqdm(range(len(rdkitArr))):
        m = rdkitArr[i]
        if m == None:
            preprocessed_rdkitArr.append(None)
        else:
            mol_species_list = split_rdkit_mol_obj(m)
            if len(mol_species_list) == 0:
                preprocessed_rdkitArr.append(None)
            else:
                largest_mol = get_largest_mol(mol_species_list)
                if len(largest_mol.GetAtoms()) <= 2:
                    preprocessed_rdkitArr.append(None)
                else:
                    preprocessed_rdkitArr.append(largest_mol)

    assert len(preprocessed_rdkitArr) == denseOutputData.shape[0]

    smiles_list = [AllChem.MolToSmiles(m) if m != None else None for m in preprocessed_rdkitArr]  # bc some empty mol in the

    assert len(preprocessed_rdkitArr) == len(smiles_list)

    return smiles_list, preprocessed_rdkitArr, folds, denseOutputData


def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively
    :param mol:
    :return:
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list


def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one
    :param mol_list:
    :return:
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]


def create_standardized_mol_id(smiles):
    if check_smiles_validity(smiles):
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles), isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        if mol != None: # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            if '.' in smiles: # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
        else:
            return
    else:
        return
