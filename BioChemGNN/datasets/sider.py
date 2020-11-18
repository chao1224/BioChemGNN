from BioChemGNN import datasets, utils
from BioChemGNN.utils import Registry as R
import os
import numpy as np


@R.register('datasets/SIDER')
class SIDER(datasets.MoleculeDataset):
    url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/sider.csv.gz'
    md5 = '77c0ef421f7cc8ce963c5836c8761fd2'
    target_fields = [
        'Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
        'Investigations', 'Musculoskeletal and connective tissue disorders', 'Gastrointestinal disorders',
        'Social circumstances', 'Immune system disorders', 'Reproductive system and breast disorders',
        'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
        'General disorders and administration site conditions', 'Endocrine disorders',
        'Surgical and medical procedures', 'Vascular disorders', 'Blood and lymphatic system disorders',
        'Skin and subcutaneous tissue disorders', 'Congenital, familial and genetic disorders',
        'Infections and infestations', 'Respiratory, thoracic and mediastinal disorders',
        'Psychiatric disorders', 'Renal and urinary disorders', 'Pregnancy, puerperium and perinatal conditions',
        'Ear and labyrinth disorders', 'Cardiac disorders', 'Nervous system disorders',
        'Injury, poisoning and procedural complications'
    ]

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