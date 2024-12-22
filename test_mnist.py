import torch
import torch_directml
from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Linear, Flatten, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchmetrics import Accuracy


class MnistCnn(Module):
    def __init__(self, classes=10, in_channels=1):
        super().__init__()

        self.features = Sequential(
            Conv2d(in_channels, 64, 3, 1, 1),
            MaxPool2d(2, 2),
            ReLU(),
            Conv2d(64, 64, 3, 1, 1),
            MaxPool2d(2, 2),
            ReLU(),
        )

        self.classifier = Sequential(
            Flatten(),
            Linear(64 * 7 * 7, 1024),
            ReLU(),
            Linear(1024, classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    transforms.RandomAffine(20, scale=(0.2, 1.)),
                                    transforms.RandomRotation(20),
                                    ])
    train_mnist = MNIST("../data", train=True, download=True, transform=transform)
    val_mnist = MNIST("../data", train=False, download=True, transform=transform)
    train_dataloader = DataLoader(train_mnist, batch_size=64, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_mnist, batch_size=64, shuffle=True, num_workers=8)
    device = torch_directml.device()
    model = MnistCnn().to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    accuracy = Accuracy(task="multiclass", num_classes=10).to(device)

    for epoch in range(100):
        model.train()
        print(f"epoch: {epoch}")
        print("-" * 100)
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            predict = model(images)
            loss = criterion(predict, labels)
            loss.backward()

            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f"batch: {batch_idx:5}, loss: {loss.item():8.4f}, accuracy: {accuracy(predict, labels).item():6.2%}")

        print("test")
        print("-" * 100)
        for batch_idx, (images, labels) in enumerate(val_dataloader):
            with torch.no_grad():
                images, labels = images.to(device), labels.to(device)
                predict = model(images)
                loss = criterion(predict, labels)

                if batch_idx % 100 == 0:
                    print(
                        f"batch: {batch_idx:5}, loss: {loss.item():8.4f}, accuracy: {accuracy(predict, labels).item():6.2%}")
