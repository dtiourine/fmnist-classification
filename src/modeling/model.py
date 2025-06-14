import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary


def get_model():
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    num_features = model.classifier[1].in_features
    num_classes = 10

    model.classifier = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(num_features, num_classes)
    )

    return model
