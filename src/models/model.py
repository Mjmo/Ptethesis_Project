from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional
from torchvision import models
from typing import List
class TorchVisionNet(nn.Module):
    def __init__(
        self,
        name,
        num_classes,
        weights="DEFAULT",
        head=[256, 128],
        dropout=[],
        last_activation=None,
    ):
        super().__init__()
        model = getattr(models, name)(weights=weights)
        layers = list(model.children())
        last_linear = layers[-1]

        if isinstance(last_linear, nn.Sequential):
            for layer in last_linear:
                if isinstance(layer, nn.Linear):
                    last_linear = layer
                    break

        head = head.copy()
        head.insert(0, last_linear.in_features)
        head.append(num_classes)

        head_layers = [nn.Linear(head[i], head[i + 1]) for i in range(len(head) - 1)]

        if dropout:
            for idx, p in dropout:
                head_layers.insert(idx, nn.Dropout(p))

        self.base = nn.Sequential(*layers[:-1])
        self.head = nn.Sequential(*head_layers)
        self.last_activation = last_activation

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        if self.last_activation:
            x = getattr(F, self.last_activation)(x, dim=1)
        return x
def loadpretrainedmodel(weights_path:str,device:torch.device)->torch.nn.Module:
    model=TorchVisionNet(name="resnet18",head=[256,128],num_classes=50,dropout=[(1,0.5)])
    state=torch.load(weights_path,map_location=device)
    model.load_state_dict(state)
    model=model.to(device)
    return model

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
def addnewhead(model: nn.Module, dims: list[int], num_classes: int,
               dropout: float = 0.5, use_bn=True):
    in_features = model.head[0].in_features

    new_dims = [in_features] + dims + [num_classes]  

    model.head = build_mlp_head(new_dims, dropout, use_bn)
    return model
