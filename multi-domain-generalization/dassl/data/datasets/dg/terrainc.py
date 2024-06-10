import os.path as osp
from sklearn.model_selection import train_test_split
import numpy as np
import os
from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class TerraInc(DatasetBase):
    """Terra Incognita dataset.

    Statistics:
        - 4 domains: 38, 46, 100, 43.
        - 10 categories: "bird", "bobcat", "cat", "coyote", "dog", "empty", "opossum", "rabbit",
        "raccoon", "squirrel".

    Reference:
        - Sara et al. Recognition in Terra Incognita.
        ECCV 2018.
    """
    dataset_dir = 'terrainc'
    domains = ['38', '46', '100', '43']
    data_url = 'https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_all_images_sm.tar.gz'
    anotation_url = 'https://lilablobssc.blob.core.windows.net/caltechcameratraps/labels/caltech_camera_traps.json.zip'

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, 'images')
        self.split_dir = osp.join(self.dataset_dir, 'splits')

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, 'terra_incognita_images.tar.gz')
            self.download_data(self.data_url, dst, from_gdrive=True)
            dst = osp.join(root, 'caltech_camera_traps.json.zip')
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        # train = self._read_data(cfg.DATASET.SOURCE_DOMAINS, 'train')
        # val = self._read_data(cfg.DATASET.SOURCE_DOMAINS, 'crossval')
        # test = self._read_data(cfg.DATASET.TARGET_DOMAINS, 'test')

        # train_ds_domain = []
        # train_ds_label = []
        # for ele in train:
        #     train_ds_domain.append(ele.domain)
        #     train_ds_label.append(ele.label)
        # value1, count1 = np.unique(train_ds_domain, return_counts=True)
        # value2, count2 = np.unique(train_ds_label, return_counts=True)
        # print(f'Train dataset statistics| Domain {value1} - count {count1} | Class: {value2} - count {count2}')
        
        # test_ds_domain = []
        # test_ds_label = []
        # for ele in test:
        #     test_ds_domain.append(ele.domain)
        #     test_ds_label.append(ele.label)
        # value1, count1 = np.unique(test_ds_domain, return_counts=True)
        # value2, count2 = np.unique(test_ds_label, return_counts=True)
        # print(f'Test dataset statistics| Domain {value1} - count {count1} | Class: {value2} - count {count2}')
        
        # super().__init__(train_x=train, val=val, test=test)

    def _read_data(self, input_domains, split):
        items = []

        for domain, dname in enumerate(input_domains):
            if split == 'all':
                file_train = osp.join(
                    self.split_dir, dname + '_train_kfold.txt'
                )
                impath_label_list = self._read_split_pacs(file_train)
                file_val = osp.join(
                    self.split_dir, dname + '_crossval_kfold.txt'
                )
                impath_label_list += self._read_split_pacs(file_val)
            else:
                file = osp.join(
                    self.split_dir, dname + '_' + split + '_kfold.txt'
                )
                impath_label_list = self._read_split_pacs(file)

            for impath, label in impath_label_list:
                item = Datum(impath=impath, label=label, domain=domain)
                items.append(item)

        return items

    def _read_split_pacs(self, split_file):
        items = []

        with open(split_file, 'r') as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                impath, label = line.split(' ')
                if impath in self._error_paths:
                    continue
                impath = osp.join(self.image_dir, impath)
                label = int(label) - 1
                items.append((impath, label))

        return items
