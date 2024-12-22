from typing import Any

from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.nn import Module
from torch.optim import Optimizer

from optimizer_builder import OptimizationBuilder


class QuikDrawModel(LightningModule):
    def __init__(self, model: Module, criterion: Module, optimization_builder: OptimizationBuilder):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimization_builder = optimization_builder

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def configure_optimizer(self) -> Optimizer:
        return self.optimization_builder.build_optimizer(self.model.parameters())

    def configure_scheduler(self, optimizer: Optimizer) -> Optimizer:
        pass

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        pass

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        pass

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        pass
