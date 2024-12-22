from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS


class SketchDatamodule(LightningDataModule):
    def prepare_data(self) -> None:
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass