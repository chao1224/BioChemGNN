from BioChemGNN import datasets, utils
from BioChemGNN.utils import Registry as R
import os
import numpy as np


@R.register('datasets/HIV')
class HIV(datasets.MoleculeDataset):
    url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/HIV.csv'
    md5 = '9ad10c88f82f1dac7eb5c52b668c30a7'
    target_fields = ['HIV_active']

    def __init__(self, path, with_hydrogen=False, kekulize=False, verbose=True, transform=None, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        file_name = utils.download(self.url, path, md5=self.md5)

        self.load_csv(file_name, smiles_field='smiles', target_fields=self.target_fields, transform=transform,
                      with_hydrogen=with_hydrogen, kekulize=kekulize, verbose=verbose, **kwargs)
        self.label_list = np.array(self.label_list)
        self.label_list[self.label_list == 0] = -1
        self.label_list = np.nan_to_num(self.label_list)
        self.label_list = self.label_list.tolist()