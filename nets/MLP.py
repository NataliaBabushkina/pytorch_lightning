import torch
from torch import nn
import pytorch_lightning as pl

class MLP(pl.LightningModule):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.l1 = nn.Linear(self.config.input_size, 128)
        self.lin_relu_stack = nn.Sequential(
            nn.Linear(self.config.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.config.output_size)
        )

    def forward(self, xb):
        return self.lin_relu_stack(xb.view(xb.size(0), -1))

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
