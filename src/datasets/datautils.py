from torchvision.datasets import ImageFolder
import numpy as np 
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset,Dataset
from collections import Counter
import random
from PIL import Image
from torchvision.transforms import ToPILImage
from typing import Tuple,List
random.seed(42)
def get_and_split(path:str, testsize:float=0.2)->Tuple[Subset,Subset]:
    dataset = ImageFolder(path, transform=None)

    targets = [label for _, label in dataset.samples]
    indices = np.arange(len(targets))
    targets_np = np.array(targets)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=testsize, random_state=42
    )
    train_idx, val_idx = next(splitter.split(indices, targets_np))

    train_data = Subset(dataset, train_idx)
    val_data   = Subset(dataset, val_idx)

    return train_data, val_data
def get_minority_classes(targets:List[int], threshold:int=50)->List[int]:
    count = Counter(targets)
    return [cls for cls, n in count.items() if n < threshold]
class OversampledAugmentedDataset(Dataset):
    def __init__(self, base_dataset, min_samples, minority_classes, augmentations):
        self.base_dataset = base_dataset
        self.augmentations = augmentations

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
        for cls, idxs in class_indices.items():
            if cls in minority_classes:
                repeats = max(min_samples - len(idxs), 0)
                self.indices.extend(idxs)
                self.indices.extend(random.choices(idxs, k=repeats))
            else:
                self.indices.extend(idxs)
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if isinstance(self.base_dataset, Subset):
            img, label = self.base_dataset.dataset[self.indices[idx]]
        else:
            img, label = self.base_dataset[self.indices[idx]]

    # Convert to PIL if not already
        if not isinstance(img, Image.Image):
            img = ToPILImage()(img)

        img = self.augmentations(img)
        return img, label