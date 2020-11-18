from BioChemGNN import datasets, utils
from BioChemGNN.utils import Registry as R
import os


@R.register('datasets/FreeSolv')
class FreeSolv(datasets.MoleculeDataset):
    url = 'https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/molnet_publish/FreeSolv.zip'
    md5 = '8d681babd239b15e2f8b2d29f025577a'
    target_fields = ['expt']

    def __init__(self, path, with_hydrogen=False, kekulize=False, verbose=True, transform=None, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, self.path, md5=self.md5)
        csv_file = utils.extract(zip_file, 'SAMPL.csv')

        self.load_csv(csv_file, smiles_field='smiles', target_fields=self.target_fields, transform=transform,
                      with_hydrogen=with_hydrogen, kekulize=kekulize, verbose=verbose, **kwargs)