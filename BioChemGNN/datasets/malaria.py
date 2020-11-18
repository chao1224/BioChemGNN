from BioChemGNN import datasets, utils
from BioChemGNN.utils import Registry as R
import os


@R.register('datasets/Malaria')
class Malaria(datasets.MoleculeDataset):
    url = "https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-06-03-malaria/malaria-processed.csv"
    md5 = "ef40ddfd164be0e5ed1bd3dd0cce9b88"
    target_fields = ["activity"]

    def __init__(self, path, with_hydrogen=False, kekulize=False, verbose=True, transform=None, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        file_name = utils.download(self.url, path, md5=self.md5)

        self.load_csv(file_name, smiles_field='smiles', target_fields=self.target_fields, transform=transform,
                      with_hydrogen=with_hydrogen, kekulize=kekulize, verbose=verbose, **kwargs)