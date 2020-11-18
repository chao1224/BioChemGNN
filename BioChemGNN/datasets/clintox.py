from BioChemGNN import datasets, utils
from BioChemGNN.utils import Registry as R
import os
import numpy as np


@R.register('datasets/ClinTox')
class ClinTox(datasets.MoleculeDataset):
    url = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
    md5 = "db4f2df08be8ae92814e9d6a2d015284"
    target_fields = ["FDA_APPROVED", "CT_TOX"]

    def __init__(self, path, with_hydrogen=False, kekulize=False, verbose=True, transform=None, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        csv_file = utils.extract(zip_file)

        self.load_csv(csv_file, smiles_field='smiles', target_fields=self.target_fields, transform=transform,
                      with_hydrogen=with_hydrogen, kekulize=kekulize, verbose=verbose, **kwargs)
        self.label_list = np.array(self.label_list)
        self.label_list[self.label_list == 0] = -1
        self.label_list = np.nan_to_num(self.label_list)
        self.label_list = self.label_list.tolist()