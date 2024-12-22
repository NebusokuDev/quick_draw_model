from typing import Callable

from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR, LRScheduler
from abc import ABC, abstractmethod

from test_mnist import MnistCnn


class OptimizationBuilder(ABC):
    def __init__(self, use_scheduler=True):
        self.use_scheduler = use_scheduler

    @abstractmethod
    def build_optimizer(self, model: Module) -> Optimizer:
        pass

    @abstractmethod
    def build_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        pass

    def __call__(self, model: Module) -> tuple[Optimizer, LRScheduler | None]:
        optimizer = self.build_optimizer(model)
        if not self.use_scheduler:
            return optimizer, None
        lr_scheduler = self.build_scheduler(optimizer)
        return optimizer, lr_scheduler


class AdamOptimizationBuilder(OptimizationBuilder):
    def __init__(self, lr: float = 1e-3, use_scheduler=True, step_size: int = 10, gamma: float = 0.1):
        super().__init__(use_scheduler)
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma

    def build_optimizer(self, model: Module) -> Optimizer:
        return Adam(model.parameters(), lr=self.lr)

    def build_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        return StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)


class LambdaOptimizationBuilder(OptimizationBuilder):
    def __init__(self,
                 optimizer_factory: Callable[[Module], Optimizer],
                 scheduler_factory: Callable[[Optimizer], LRScheduler],
                 use_scheduler=True):
        super().__init__(use_scheduler)
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory

    def build_optimizer(self, model: Module) -> Optimizer:
        return self.optimizer_factory(model)

    def build_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        return self.scheduler_factory(optimizer)


if __name__ == '__main__':
    # hoge_optimizer, step_lr
    builder = LambdaOptimizationBuilder(optimizer_factory=lambda param: Adam(param),
                                        scheduler_factory=lambda optimizer: StepLR(optimizer, step_size=10, gamma=0.1),
                                        )
    # adam, step_lr
    builder = AdamOptimizationBuilder()

    # どっちも同じように生成できる
    optimizer, scheduler = builder(MnistCnn())
