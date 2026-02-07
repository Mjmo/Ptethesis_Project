from mydata.datautils import OversampledAugmentedDataset,get_and_split,get_minority_classes
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from typing import Callable,Tuple
from torchvision.datasets import ImageFolder
from copy import deepcopy
def get_data_loader(
    dataset:torch.utils.data.Dataset,
    min_samples:int,
    train_aug:Callable,
    valid_aug:Callable,
    num_workers:int,
    batch_size:int,
    seed:int,
    test_size:float=0.1,
)->Tuple[DataLoader,DataLoader]:
    trainset, valset = get_and_split(dataset, test_size)

    train_targets = [trainset.dataset.targets[i] for i in trainset.indices]

    minority_classes = get_minority_classes(
        targets=train_targets,
        threshold=min_samples
    )

    train_dataset = OversampledAugmentedDataset(
        base_dataset=trainset,
        min_samples=min_samples,
        minority_classes=minority_classes,
        augmentations=train_aug,seed=seed
    )

   

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_dataset=deepcopy(valset.dataset)
    val_dataset.transform=valid_aug
    val_subset = torch.utils.data.Subset(val_dataset, valset.indices)

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader
def get_dataset(folderpath:str):
    dataset=ImageFolder(folderpath,transform=None)
    return dataset