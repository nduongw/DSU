import os
import os.path as osp

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class Spawrious(DatasetBase):
    """Spawrious dataset.
    Statistics:
    """
    dataset_dir = 'spawrious224'

    def __init__(self, cfg, combinations, type1=False):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.type1 = type1
        self.label_idx = {'bulldog': 0, 'dachshund': 1, 'labrador': 2, 'corgi': 3}
        self.location_idx = {'desert': 0, 'jungle': 1, 'dirt': 2, 'snow': 3, 'beach': 4, 'mountain': 5}
        
        self.combinations = combinations

        train = self._read_data(self.combinations['train'])
        val = self._read_data(self.combinations['test'])
        test = self._read_data(self.combinations['test'])

        super().__init__(train_x=train, val=val, test=test)
    
    # Buils combination dictionary for o2o datasets
    def build_type1_combination(self, group,test,filler):
        total = 3168
        counts = [int(0.97*total),int(0.87*total)]
        combinations = {}
        combinations['train'] = {
            ## correlated class
            ("bulldog",):[(group[0],counts[0]),(group[0],counts[1])],
            ("dachshund",):[(group[1],counts[0]),(group[1],counts[1])],
            ("labrador",):[(group[2],counts[0]),(group[2],counts[1])],
            ("corgi",):[(group[3],counts[0]),(group[3],counts[1])],
            ## filler
            ("bulldog","dachshund","labrador","corgi"):[(filler,total-counts[0]),(filler,total-counts[1])],
        }
        ## TEST
        combinations['test'] = {
            ("bulldog",):[test[0], test[0]],
            ("dachshund",):[test[1], test[1]],
            ("labrador",):[test[2], test[2]],
            ("corgi",):[test[3], test[3]],
        }
        return combinations

    # Buils combination dictionary for m2m datasets
    def build_type2_combination(self, group,test):
        total = 3168
        counts = [total,total]
        combinations = {}
        combinations['train_combinations'] = {
            ## correlated class
            ("bulldog",):[(group[0],counts[0]),(group[1],counts[1])],
            ("dachshund",):[(group[1],counts[0]),(group[0],counts[1])],
            ("labrador",):[(group[2],counts[0]),(group[3],counts[1])],
            ("corgi",):[(group[3],counts[0]),(group[2],counts[1])],
        }
        combinations['test_combinations'] = {
            ("bulldog",):[test[0], test[1]],
            ("dachshund",):[test[1], test[0]],
            ("labrador",):[test[2], test[3]],
            ("corgi",):[test[3], test[2]],
        }
        return combinations

    def _read_data(self, combinations):
        items = []

        for classes, comb_list in combinations.items():
            for ind, location_limit in enumerate(comb_list):
                if isinstance(location_limit, tuple):
                    location, limit = location_limit
                else:
                    location, limit = location_limit, None
                for cls in classes:
                    path = osp.join(self.dataset_dir, f"{0 if not self.type1 else ind}/{location}/{cls}")
                    impaths = os.listdir(path)
                    selected_impaths = impaths[:limit]
                    for impath in selected_impaths:
                        label = self.label_idx[cls]
                        impath = osp.join(path, impath)
                        item = Datum(
                            impath=impath,
                            label=label,
                            domain=self.location_idx[location],
                            classname=cls
                        )
                        items.append(item)

        return items
    
@DATASET_REGISTRY.register()
class SpawriousO2O_easy(Spawrious):
    def __init__(self, cfg):
        group = ["desert","jungle","dirt","snow"]
        test = ["dirt","snow","desert","jungle"]
        filler = "beach"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(cfg, combinations, type1=True)

@DATASET_REGISTRY.register()
class SpawriousO2O_medium(Spawrious):
    def __init__(self, cfg):
        group = ['mountain', 'beach', 'dirt', 'jungle']
        test = ['jungle', 'dirt', 'beach', 'snow']
        filler = "desert"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(cfg, combinations, type1=True)

@DATASET_REGISTRY.register()
class SpawriousO2O_hard(Spawrious):
    def __init__(self, cfg):
        group = ['jungle', 'mountain', 'snow', 'desert']
        test = ['mountain', 'snow', 'desert', 'jungle']
        filler = "beach"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(cfg, combinations, type1=True)

@DATASET_REGISTRY.register()
class SpawriousM2M_easy(Spawrious):
    def __init__(self, cfg):
        group = ['desert', 'mountain', 'dirt', 'jungle']
        test = ['dirt', 'jungle', 'mountain', 'desert']
        combinations = self.build_type2_combination(group,test)
        super().__init__(cfg, combinations) 

@DATASET_REGISTRY.register()
class SpawriousM2M_medium(Spawrious):
    def __init__(self, cfg):
        group = ['beach', 'snow', 'mountain', 'desert']
        test = ['desert', 'mountain', 'beach', 'snow']
        combinations = self.build_type2_combination(group,test)
        super().__init__(cfg, combinations) 

@DATASET_REGISTRY.register()
class SpawriousM2M_hard(Spawrious):
    def __init__(self, cfg):
        group = ["dirt","jungle","snow","beach"]
        test = ["snow","beach","dirt","jungle"]
        combinations = self.build_type2_combination(group,test)
        super().__init__(cfg, combinations) 