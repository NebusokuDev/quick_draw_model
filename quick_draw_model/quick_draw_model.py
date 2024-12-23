from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer, Adam
from torchmetrics import Accuracy


class QuikDrawModel(LightningModule):
    def __init__(self, model: Module, criterion: Module, lr=0.0001):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.lr = lr

        self.metrics = Accuracy(task="multiclass", num_classes=250)

    def forward(self, x) -> Tensor:
        return self.model(x)

    def configure_optimizer(self) -> Optimizer:
        return Adam(self.parameters(), self.lr)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> STEP_OUTPUT:
        image, label = batch

        predict = self.forward(image)
        loss = self.criterion(predict, label)

        self.metrics(predict, label)

        self.log("train_loss", loss)
        self.log("test_acc", self.metrics, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> STEP_OUTPUT:
        image, label = batch

        predict = self.forward(image)
        loss = self.criterion(predict, label)

        self.log("train_loss", loss)
        self.log("val_acc", self.metrics, on_step=True, on_epoch=True)

        self.metrics(predict, label)

        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> STEP_OUTPUT:
        image, label = batch

        predict = self.forward(image)
        loss = self.criterion(predict, label)

        self.log("train_loss", loss)
        self.log("train_acc", self.metrics, on_step=True, on_epoch=True)

        self.metrics(predict, label)

        return loss
