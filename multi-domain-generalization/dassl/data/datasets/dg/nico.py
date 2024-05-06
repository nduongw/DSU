import os.path as osp
import numpy as np
from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class Nico(DatasetBase):
    """Nico.
    Statistics:
        - 6 distinct domains: Autumn, Dim, Grass, Outdoor,
        Rock, Water.
        - 80 categories.
    """
    dataset_dir = 'NICO_DG'
    domains = [
        'autumn', 'dim', 'grass', 'outdoor', 'rock', 'water'
    ]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.split_dir = osp.join(self.dataset_dir, 'splits')
        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split='train')
        val = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split='test')
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, split='test')

        super().__init__(train_x=train, val=val, test=test)

    def _read_data(self, input_domains, split='train'):
        items = []

        for domain, dname in enumerate(input_domains):
            filename = dname + '_' + split + '.txt'
            split_file = osp.join(self.split_dir, filename)
            with open(split_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    last_idx = line.rfind(' ')
                    impath, label = line[:last_idx], line[last_idx:]
                    impath = impath[8:]
                    impath = osp.join(self.dataset_dir, impath)
                    label = int(label)
                    item = Datum(impath=impath, label=label, domain=domain)
                    items.append(item)

        return items