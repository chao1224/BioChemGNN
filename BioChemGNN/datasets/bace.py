from BioChemGNN import datasets, utils
from BioChemGNN.utils import Registry as R
import os
import numpy as np


@R.register('datasets/BACE')
class BACE(datasets.MoleculeDataset):
    url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/bace.csv'
    md5 = 'ba7f8fa3fdf463a811fa7edea8c982c2'
    target_fields = ['Class']

    def __init__(self, path, with_hydrogen=False, kekulize=False, verbose=True, transform=None, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        file_name = utils.download(self.url, self.path, md5=self.md5)

        self.load_csv(file_name, smiles_field='mol', target_fields=self.target_fields, transform=transform,
                      with_hydrogen=with_hydrogen, kekulize=kekulize, verbose=verbose, **kwargs)
        self.label_list = np.array(self.label_list)
        self.label_list[self.label_list == 0] = -1
        self.label_list = np.nan_to_num(self.label_list)
        self.label_list = self.label_list.tolist()
