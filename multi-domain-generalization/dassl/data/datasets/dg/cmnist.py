import os.path as osp
from torchvision.datasets import MNIST
from ..build import DATASET_REGISTRY
from ..base_dataset import MNISTDatum, DatasetBase

def torch_bernoulli_(self, p, size):
    return (torch.rand(size) < p).float()

def torch_xor_(self, a, b):
    return (a - b).abs()

def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        classes = labels
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels, self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels, self.torch_bernoulli_(environment, len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return [MNISTDatum(impath=m, label=n, domain=p) for m, n, p in zip(x, classes, y)]
    
@DATASET_REGISTRY.register()
class ColoredMNIST(DatasetBase):
    """ColoredMNIST."""
    
    dataset_dir = 'mnist'
    domains = ['green', 'half', 'red']

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, 'images')
        self.split_dir = osp.join(self.dataset_dir, 'splits')

        if not osp.exists(self.dataset_dir):
            MNIST(self.data_dir, download=True)
        
        if cfg.DATASET.SOURCE_DOMAINS == 'green':
            envs = [0.1, 0.2, 0.9]
        elif cfg.DATASET.SOURCE_DOMAINS == 'red':
            envs = [0.9, 0.8, 0.1]
            
        original_dataset_train = MNIST(self.dataset_dir, train=True, download=True)
        original_dataset_test = MNIST(self.dataset_dir, train=False, download=True)
        
        original_images = torch.cat((original_dataset_train.data, original_dataset_test.data))
        original_labels = torch.cat((original_dataset_train.targets, original_dataset_test.targets))
        
        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]
        
        self.datasets = []
        for i in range(len(envs)):
            images = original_images[i::len(envs)]
            labels = original_labels[i::len(envs)]
            self.datasets.append(color_dataset(images, labels, envs[i]))
        
        train = self.datasets[0]
        val = self.datasets[1]
        test = self.datasets[2]
        
        super().__init__(train_x=train, val=val, test=test)
