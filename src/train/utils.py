import torch
from collections import Counter
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
            
            
def get_weights(dataset:torch.utils.data.Dataset)->torch.tensor:
      all_labels=[label for _,label in dataset]
      counts=Counter(all_labels)
      print(counts)
      num_classes=len(counts)
      print(counts)
      total=sum(counts.values())
      class_weights = torch.tensor([total / counts[i] for i in range(num_classes)], dtype=torch.float32)
      return class_weights