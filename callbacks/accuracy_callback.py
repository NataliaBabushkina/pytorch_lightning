from pytorch_lightning.callbacks import Callback

class CalculatingAccuracyCallback(Callback):

    def __init__(self):
        self.accuracy = None

