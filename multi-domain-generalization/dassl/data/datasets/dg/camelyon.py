import os.path as osp
import numpy as np
import pandas as pd
from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase

TEST_CENTER = 2
VAL_CENTER = 1

@DATASET_REGISTRY.register()
class Camelyon(DatasetBase):
    """Camelyon.
    Statistics:

    """
    dataset_dir = 'camelyon17_v1.0'
    domains = [
        '0', '1', '2', '3', '4'
    ]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )
        
        self._metadata_df = pd.read_csv(
            osp.join(self.dataset_dir, 'metadata.csv'),
            index_col=0,
            dtype={'patient': 'str'})
        
        self._input_array = [
            f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png'
            for patient, node, x, y in
            self._metadata_df.loc[:, ['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False, name=None)]

        train = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split='train')
        val = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split='test')
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, split='test')

        super().__init__(train_x=train, val=val, test=test)

    def _read_data(self, input_domains, split='train'):
        items = []

        for domain, dname in enumerate(input_domains):
            domain_filenames = self._metadata_df[self._metadata_df['center'] == int(domain)]
            domain_filenames.index = range(0, len(domain_filenames))
            impaths = [f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png' for patient, node, x, y in domain_filenames.loc[:, ['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False, name=None)]
            for idx, impath in enumerate(impaths):
                impath = osp.join(self.dataset_dir, impath)
                label = domain_filenames['tumor'][idx]
                item = Datum(impath=impath, label=int(label), domain=int(domain))
                items.append(item)

        return items