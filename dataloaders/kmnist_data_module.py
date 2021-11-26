import pytorch_lightning as pl
from torchvision.datasets import KMNIST
from torch.utils.data import random_split, DataLoader
from dataloaders.data_transforms import compose_transforms

class KMNISTDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.kmnist_train = None
        self.kmnist_val = None
        self.kmnist_test = None
        self.config = config
        self.num_classes = 0

    def prepare_data(self):
        KMNIST(root=self.config.root_dir, train=True, download=True)
        KMNIST(root=self.config.root_dir, train=False, download=True)

    def setup(self, stage = None):
        if stage == 'fit' or stage is None:
            kmnist_full = KMNIST(self.config.root_dir, train = True, transform=compose_transforms())
            self.kmnist_train, self.kmnist_val = random_split(kmnist_full, [55000, 5000])
            self.num_classes = len(kmnist_full.classes)
        if stage == 'test' or stage is None:
            self.kmnist_test = KMNIST(self.config.root_dir, train = False, transform=compose_transforms())

    def train_dataloader(self):
        return DataLoader(self.kmnist_train, batch_size=self.config.batch_size)

    def val_dataloader(self):
        return DataLoader(self.kmnist_val, batch_size=self.config.batch_size)

    def test_dataloader(self):
        return DataLoader(self.kmnist_test, batch_size=self.config.batch_size)


