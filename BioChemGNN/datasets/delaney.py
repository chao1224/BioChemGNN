from BioChemGNN import datasets, utils
from BioChemGNN.utils import Registry as R
import os


@R.register('datasets/Delaney')
class Delaney(datasets.MoleculeDataset):
    url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/delaney-processed.csv'
    md5 = '0c90a51668d446b9e3ab77e67662bd1c'
    target_fields = ['measured log solubility in mols per litre']

    def __init__(self, path, with_hydrogen=False, kekulize=False, verbose=True, transform=None, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        file_name = utils.download(self.url, self.path, md5=self.md5)

        self.load_csv(file_name, smiles_field='smiles', target_fields=self.target_fields, transform=transform,
                      with_hydrogen=with_hydrogen, kekulize=kekulize, verbose=verbose, **kwargs)