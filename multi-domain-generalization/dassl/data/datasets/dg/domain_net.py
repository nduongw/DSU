import os.path as osp

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class DomainNetDG(DatasetBase):
    """DomainNet.
    Statistics:
        - 6 distinct domains: Clipart, Infograph, Painting, Quickdraw,
        Real, Sketch.
        - Around 0.6M images.
        - 345 categories.
        - URL: http://csr.bu.edu/ftp/visda/2019/multi-source/.
    """
    dataset_dir = 'domainnet'
    domains = [
        'clipart', 'painting', 'quickdraw', 'real', 'sketch'
    ]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.split_dir = osp.join(self.dataset_dir, 'splits')
        self.reduce_rate = cfg.REDUCE
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
            counters = 0
            with open(split_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    #Due to OOM, only consider a small set of images for training and testing
                    #if reduce_rate = 1, use all data
                    if (counters % self.reduce_rate == 0):
                        line = line.strip()
                        impath, label = line.split(' ')
                        classname = impath.split('/')[1]
                        impath = osp.join(self.dataset_dir, impath)
                        label = int(label)
                        item = Datum(
                            impath=impath,
                            label=label,
                            domain=domain,
                            classname=classname
                        )
                        items.append(item)
                    counters += 1
        return items