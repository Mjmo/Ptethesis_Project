import torch
from collections import Counter
def freeze_all_layers(model:torch.nn.Module,head_trainable:bool=True):
        for name,param in model.named_parameters():
            if name.startswith("head") or name.startswith("base.7") or name.startswith("base.6") and head_trainable:
                param.requires_grad=True
            else:
                param.requires_grad=False
def get_weights(dataset:torch.utils.data.Dataset)->torch.tensor:
      all_labels=[label for _,label in dataset]
      counts=Counter(all_labels)
      num_classes=len(counts)
      print(counts)
      total=sum(counts.values())
      class_weights = torch.tensor([total / counts[i] for i in range(num_classes)], dtype=torch.float32)
      return class_weights