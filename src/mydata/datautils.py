from torchvision.datasets import ImageFolder
import numpy as np 
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset,Dataset
from collections import Counter
import random
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from typing import Tuple,List
random.seed(42)
"""
    Get the orginal dataset that contais all images:
    - Stratified splitting of the dataset into train and test so that the proportion of each
    class stays the same in trainset and in testset
    """
def get_and_split(dataset:torch.utils.data.Dataset,seed:int ,testsize:float=0.2,)->Tuple[Subset,Subset]:
    if not isinstance(dataset,ImageFolder):
        raise TypeError("We expects to get an Imagefolder")
    if hasattr(dataset,"targets"):
        targets = dataset.targets
    elif hasattr(dataset,"samples"):
        targets = [y for _, y in dataset.samples]
    else:
        raise TypeError("The dataset must have either target or sample attributes")
    indices = np.arange(len(targets))
    targets_np = np.array(targets)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=testsize, random_state=seed
    )
    train_idx, val_idx = next(splitter.split(indices, targets_np))

    train_data = Subset(dataset, train_idx)
    val_data   = Subset(dataset, val_idx)

    return train_data, val_data
"""
   Get classnames that have less samples than the threshold so we can augment 
   the classes in the trainset who has few samples
    """
def get_minority_classes(targets:List[int], threshold:int=50)->List[int]:
    count = Counter(targets)
    return [cls for cls, n in count.items() if n < threshold]
class OversampledAugmentedDataset(Dataset):
    def __init__(self, base_dataset:Dataset, min_samples:int, minority_classes:list[int],seed:int ,augmentations:callable):
        self.base_dataset = base_dataset
        self.augmentations = augmentations
        self.seed=seed
        if isinstance(base_dataset, Subset):
            targets = [base_dataset.dataset.targets[i] for i in base_dataset.indices]
            original_indices = base_dataset.indices
        else:
            targets = base_dataset.targets
            original_indices = list(range(len(base_dataset)))

        class_indices = {}
        for idx, label in zip(original_indices, targets):
            class_indices.setdefault(label, []).append(idx)

        self.indices = []
        self.targets=[]
        for cls, idxs in class_indices.items():
            if cls in minority_classes:
                repeats = max(min_samples - len(idxs), 0)
                rng = random.Random(self.seed)
                sampled=idxs+rng.choices(idxs,k=repeats)
            else:
                sampled=idxs
            self.indices.extend(sampled)
            self.targets.extend([cls]*len(sampled))
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if isinstance(self.base_dataset, Subset):
            img, label = self.base_dataset.dataset[self.indices[idx]]
        else:
            img, label = self.base_dataset[self.indices[idx]]
        if self.augmentations:
            img = self.augmentations(img)
        return img, label