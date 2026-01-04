from torchvision import models 
import torch.nn as nn
def build_mlp_head(
    dims,
    dropout=0.5,
    use_bn=True
):
    layers = []

    for i in range(len(dims) - 1):
        in_dim = dims[i]
        out_dim = dims[i + 1]

        layers.append(nn.Linear(in_dim, out_dim))

        # Don't add BN/ReLU/Dropout after final layer
        if i < len(dims) - 2:
            if use_bn:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)
def getmodelres50(
    num_classes: int,
    head: list[int],
    dropout: float,
    bn: bool = False
):
    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1
    )

    in_features = model.fc.in_features
    dims = [in_features] + head + [num_classes]

    model.fc = build_mlp_head(
        dims=dims,
        dropout=dropout,
        use_bn=bn
    )

    return model
