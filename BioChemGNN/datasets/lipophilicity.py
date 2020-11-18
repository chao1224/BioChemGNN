from BioChemGNN import datasets, utils
from BioChemGNN.utils import Registry as R
import os


@R.register('datasets/Lipophilicity')
class Lipophilicity(datasets.MoleculeDataset):
    url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/Lipophilicity.csv'
    md5 = '85a0e1cb8b38b0dfc3f96ff47a57f0ab'
    target_fields = ['exp']

    def __init__(self, path, with_hydrogen=False, kekulize=False, verbose=True, transform=None, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        file_name = utils.download(self.url, path, md5=self.md5)

        self.load_csv(file_name, smiles_field='smiles', target_fields=self.target_fields, transform=transform,
                      with_hydrogen=with_hydrogen, kekulize=kekulize, verbose=verbose, **kwargs)