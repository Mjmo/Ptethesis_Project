import torch
from collections import Counter
import mlflow
def freeze_all_layers(
    model: torch.nn.Module,
    n: int,
    head_trainable: bool = True
):
    """
    Freeze all layers except:
    - head (optional)
    - last n blocks from base
    """
    for param in model.parameters():
        param.requires_grad = False


    if head_trainable:
        for name, param in model.named_parameters():
            if name.startswith("head"):
                param.requires_grad = True

    if n > 0:
        layers = ["base.7", "base.6", "base.5", "base.4", "base.2", "base.1"]
        layers_to_unfreeze = layers[:n]

        for name, param in model.named_parameters():
            if any(name.startswith(layer) for layer in layers_to_unfreeze):
                param.requires_grad = True
            
            

def compute_class_weights(dataset:torch.utils.data.Dataset):
    targets = torch.tensor(dataset.targets)
    counts = torch.bincount(targets)
    total = targets.numel()
    class_weights = total / counts.float()

    return class_weights
def initate_mlflow(path):
    mlflow.set_tracking_uri(path)