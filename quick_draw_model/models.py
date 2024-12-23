from torch.nn import Linear
from torchinfo import summary
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def create_mobilenet(num_classes=250):
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[-1] = Linear(1024, num_classes)
    return model


def create_efficientnet(num_classes=250):
    model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
    model.classifier[-1] = Linear(1280, num_classes)
    return model


if __name__ == '__main__':
    mobilenet = create_mobilenet()
    summary(mobilenet, (4, 3, 224, 224))

    efficientnet = create_efficientnet()
    summary(efficientnet, (4, 3, 224, 224))