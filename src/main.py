from models.model import loadpretrainedmodel,add_new_head
from datasets.datautils import get_and_split,get_minority_classes,OversampledAugmentedDataset
import torch
from datasets.gettraintest import get_data_loader
import torchvision.transforms as transforms
from train.utils import freeze_all_layers,get_weights
from EDA.display import displayrandom,compute_mean_std
from train.train import train_model
import zipfile
device = "cuda" if torch.cuda.is_available() else "cpu"
datapath = "/home/ahmad-hawa/Desktop/Pretheisis /IOW"

model = loadpretrainedmodel("/home/ahmad-hawa/Desktop/Pretheisis /best_state.pth", device=device)

# Augmentations
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0005]*3, std=[0.7142]*3)
])
valid_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0005]*3, std=[0.7142]*3)
])

# Dataset loaders
batch_size = 16
num_workers = 4
train_load, test_load = get_data_loader(
    datapath, min_samples=30, train_aug=train_transforms,
    valid_aug=valid_transform, batch_size=batch_size,
    test_size=0.5, num_workers=num_workers
)

# Add new head and freeze layers
num_classes = 23
add_new_head(model, num_classes=num_classes, head=[512,256], dropout=[1,0.5])
freeze_all_layers(model, head_trainable=True)

# Compute class weights
weights = get_weights(train_load.dataset)
weights = torch.clamp(weights, max=5.0)
criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))

# Optimizer only on trainable params
optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

# Train
history = train_model(
    model, train_load, test_load,
    criterion, optim, scheduler=None,
    num_epochs=30, device=device,
    save_path="src/train/best.pth"
)