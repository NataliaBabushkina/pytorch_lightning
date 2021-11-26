import torch
from torch import nn
import pytorch_lightning as pl
from dataloaders.data_transforms import compose_transforms
from torchvision import datasets
from torch.utils.data import DataLoader

class LogReg(pl.LightningModule):
    def __init__(self, config):
        super(LogReg, self).__init__()
        self.config = config
        self.lin = nn.Linear(self.config.input_size, self.config.output_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, xb):
        return self.lin(xb.view(xb.size(0), -1))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), self.config.lr, self.config.optim_arg)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return loss

