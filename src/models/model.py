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
def add_new_head(model:torch.nn.Module,num_classes:int,head:List[int],dropout:List[int])->None:
    assert len(dropout) == len(head), "dropout and head must have same length"
    modules=[]
    in_features=model.head[0].in_features
    modules.append(nn.Linear(in_features,head[0]))
    modules.append(nn.ReLU())
    modules.append(nn.Dropout(dropout[0]))
    for i in range(len(head)-1):
        modules.append(nn.Linear(head[i],head[i+1]))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(dropout[i+1]))
    modules.append(nn.Linear(head[-1],num_classes))
    model.head=nn.Sequential(*modules)
