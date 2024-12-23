import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize
from quick_draw_model.downloader import Downloader


class HowDoHumansSketchObjects(LightningDataModule):
    URL = "https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip"

    def __init__(self, root="./data", batch_size=32, num_workers=4, transforms=None):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = None
        self.val = None
        self.downloader = Downloader(f"{self.root}/sketch_dataset", self.URL)

        self.transforms = transforms or Compose([
            ToTensor(),
            Normalize(mean=(0.5,), std=(0.5,))
        ])

    def prepare_data(self) -> None:
        self.downloader()

    def setup(self, stage=None) -> None:
        dataset = ImageFolder(self.downloader.extract_path, transform=self.transforms)
        total = len(dataset)
        train_size = int(total * 0.8)
        val_size = total - train_size
        self.train, self.val = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)
