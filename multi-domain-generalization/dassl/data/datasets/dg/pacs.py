import os.path as osp
import torch
import numpy as np
import os
from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class PACS(DatasetBase):
    """PACS.

    Statistics:
        - 4 domains: Photo (1,670), Art (2,048), Cartoon
        (2,344), Sketch (3,929).
        - 7 categories: dog, elephant, giraffe, guitar, horse,
        house and person.

    Reference:
        - Li et al. Deeper, broader and artier domain generalization.
        ICCV 2017.
    """
    dataset_dir = 'pacs'
    domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    data_url = 'https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE'
    # the following images contain errors and should be ignored
    _error_paths = ['sketch/dog/n02103406_4068-1.png']

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, 'images')
        self.split_dir = osp.join(self.dataset_dir, 'splits')

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, 'pacs.zip')
            self.download_data(self.data_url, dst, from_gdrive=True)
        
        # test_path = []
        
        # if osp.exists(f'{self.dataset_dir}/splits/test.txt'):
        #     store_file = open(f'{self.dataset_dir}/splits/test.txt', 'r')
        #     lines = store_file.readlines()
        #     store_file.close()
        #     for line in lines:
        #         test_path.append(line.strip())

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train = self._read_data(cfg.DATASET.SOURCE_DOMAINS, 'train')
        val = self._read_data(cfg.DATASET.SOURCE_DOMAINS, 'crossval')
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, 'test')
        
        # condition = lambda x: x in test_path
        # filter_train = list(filter(lambda x: not condition(x.impath), train))
        # filter_val = list(filter(lambda x: not condition(x.impath), val))
        # filter_test = list(filter(lambda x: not condition(x.impath), test))

        train_ds_domain = []
        train_ds_label = []
        for ele in train:
            train_ds_domain.append(ele.domain)
            train_ds_label.append(ele.label)
        value1, count1 = np.unique(train_ds_domain, return_counts=True)
        value2, count2 = np.unique(train_ds_label, return_counts=True)
        print(f'Train dataset statistics| Domain {value1} - count {count1} | Class: {value2} - count {count2}')
        
        test_ds_domain = []
        test_ds_label = []
        for ele in test:
            test_ds_domain.append(ele.domain)
            test_ds_label.append(ele.label)
        value1, count1 = np.unique(test_ds_domain, return_counts=True)
        value2, count2 = np.unique(test_ds_label, return_counts=True)
        print(f'Test dataset statistics| Domain {value1} - count {count1} | Class: {value2} - count {count2}')
        
        # super().__init__(train_x=filter_train, val=filter_val, test=filter_test)
        super().__init__(train_x=train, val=val, test=test)

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

@DATASET_REGISTRY.register()
class TotalPACS(DatasetBase):
    """PACS.

    Statistics:
        - 4 domains: Photo (1,670), Art (2,048), Cartoon
        (2,344), Sketch (3,929).
        - 7 categories: dog, elephant, giraffe, guitar, horse,
        house and person.

    Reference:
        - Li et al. Deeper, broader and artier domain generalization.
        ICCV 2017.
    """
    dataset_dir = 'pacs'
    domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    data_url = 'https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE'
    # the following images contain errors and should be ignored
    _error_paths = ['sketch/dog/n02103406_4068-1.png']

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, 'images')
        self.split_dir = osp.join(self.dataset_dir, 'splits')
        self.total = []

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, 'pacs.zip')
            self.download_data(self.data_url, dst, from_gdrive=True)
        
        # import pdb; pdb.set_trace()
        if osp.exists(f'{self.dataset_dir}/splits/test.txt'):
            os.remove(f'{self.dataset_dir}/splits/test.txt')

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train = self._read_data(cfg.DATASET.SOURCE_DOMAINS, 'train')
        val = self._read_data(cfg.DATASET.SOURCE_DOMAINS, 'crossval')
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, 'test')
        total = []
        for item in train:
            total.append(item)
        
        for item in test:
            total.append(item)
        
        for item in val:
            total.append(item)
        
        train_amount, val_amount, test_amount = len(train), len(val), len(test)
        
        train_ds, test_ds = torch.utils.data.random_split(total, [train_amount + val_amount, test_amount])
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_amount, val_amount])
        
        store_file = open(f'{self.dataset_dir}/splits/test.txt', 'a')
        train_ds_domain = []
        train_ds_label = []
        for val in train_ds:
            train_ds_domain.append(val.domain)
            train_ds_label.append(val.label)
        value1, count1 = np.unique(train_ds_domain, return_counts=True)
        value2, count2 = np.unique(train_ds_label, return_counts=True)
        print(f'Train dataset statistics| Domain {value1} - count {count1} | Class: {value2} - count {count2}')
        
        test_ds_domain = []
        test_ds_label = []
        for val in test_ds:
            test_ds_domain.append(val.domain)
            test_ds_label.append(val.label)
            store_file.write(f'{val.impath}\n')
        value1, count1 = np.unique(test_ds_domain, return_counts=True)
        value2, count2 = np.unique(test_ds_label, return_counts=True)
        print(f'Test dataset statistics| Domain {value1} - count {count1} | Class: {value2} - count {count2}')

        store_file.close()
        
        super().__init__(train_x=train_ds, val=val_ds, test=test_ds)
    
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
                if split == 'test':
                    domain = 3
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