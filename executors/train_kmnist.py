import pytorch_lightning as pl
from configs import config
from nets.MLP import MLP
from nets.LogReg import LogReg
from dataloaders.kmnist_data_module import KMNISTDataModule

if __name__ == '__main__':
    dm = KMNISTDataModule(config)
    model1 = MLP(config)
    model2 = LogReg(config)
    trainer = pl.Trainer(max_epochs=config.nrof_epochs, progress_bar_refresh_rate=20)
    trainer.fit(model1, dm)