from datasets.datautils import OversampledAugmentedDataset,get_and_split,get_minority_classes
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from typing import Callable,Tuple

def get_data_loader(
    path:str,
    min_samples:int,
    train_aug:Callable,
    valid_aug:Callable,
    num_workers:int,
    batch_size:int,
    test_size:float=0.1
)->Tuple[DataLoader,DataLoader]:
    trainset, valset = get_and_split(path, test_size)

    train_targets = [trainset.dataset.targets[i] for i in trainset.indices]

    minority_classes = get_minority_classes(
        targets=train_targets,
        threshold=min_samples
    )

    train_dataset = OversampledAugmentedDataset(
        base_dataset=trainset,
        min_samples=min_samples,
        minority_classes=minority_classes,
        augmentations=train_aug
    )

    valset.dataset.transform = valid_aug

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader