from BioChemGNN import datasets, utils
from BioChemGNN.utils import Registry as R
import os


@R.register('datasets/CEP')
class CEP(datasets.MoleculeDataset):
    url = 'https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-06-02-cep-pce/cep-processed.csv'
    md5 = 'b6d257ff416917e4e6baa5e1103f3929'
    target_fields = ['PCE']

    def __init__(self, path, with_hydrogen=False, kekulize=False, verbose=True, transform=None, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        file_name = utils.download(self.url, self.path, md5=self.md5)

        self.load_csv(file_name, smiles_field='smiles', target_fields=self.target_fields, transform=transform,
                      with_hydrogen=with_hydrogen, kekulize=kekulize, verbose=verbose, **kwargs)