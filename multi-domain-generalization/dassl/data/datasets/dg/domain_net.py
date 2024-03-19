import os.path as osp

from ..build import DATASET_REGISTRY
from .digits_dg import DigitsDG
from ..base_dataset import DatasetBase


@DATASET_REGISTRY.register()
class DomainNet(DatasetBase):
    """Domain Net.

    Statistics:
        - Around 15,500 images.
        - 65 classes related to office and home objects.
        - 4 domains: Art, Clipart, Product, Real World.
        - URL: http://hemanthdv.org/OfficeHome-Dataset/.

    Reference:
        - Venkateswara et al. Deep Hashing Network for Unsupervised
        Domain Adaptation. CVPR 2017.
    """
    dataset_dir = 'domainnet'
    domains = ['art', 'clipart', 'product', 'real_world']
    data_url = [
        "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"
    ]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        
        for url in urls:
            download_and_extract(url, os.path.join(self.dataset_dir, url.split("/")[-1]))

        with open(osp.join(self.dataset_dir, "domain_net_duplicates.txt"), "r") as f:
        for line in f.readlines():
            try:
                os.remove(os.path.join(full_path, line.strip()))
            except OSError:
                pass
            
        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train = self.read_data(
            self.dataset_dir, cfg.DATASET.SOURCE_DOMAINS, 'train'
        )
        val = self.read_data(
            self.dataset_dir, cfg.DATASET.SOURCE_DOMAINS, 'val'
        )
        test = self.read_data(
            self.dataset_dir, cfg.DATASET.TARGET_DOMAINS, 'all'
        )

        super().__init__(train_x=train, val=val, test=test)
    
    def _read_data(self, input_domains, split):
        items = []

        for domain, dname in enumerate(input_domains):
            dname = dname.upper()
            path = osp.join(self.dataset_dir, dname, split)
            folders = listdir_nohidden(path)
            folders.sort()

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(path, folder, '*.jpg'))

                for impath in impaths:
                    item = Datum(impath=impath, label=label, domain=domain)
                    items.append(item)

        return items
