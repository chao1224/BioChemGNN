from BioChemGNN import datasets, utils
from BioChemGNN.utils import Registry as R
import os
import numpy as np


@R.register('datasets/Tox21')
class Tox21(datasets.MoleculeDataset):
    url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/tox21.csv.gz'
    md5 = '2882d69e70bba0fec14995f26787cc25'
    target_fields = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                     'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

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