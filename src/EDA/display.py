import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch
from torchvision.transforms import ToPILImage
import os
from PIL import Image
def unnormalize(img, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return img * std + mean
def displayrandom(dataloader):
    mean=[0.0023, 0.0024, 0.0030]
    std=[0.4607, 0.4930, 0.5443]
    images,labels=next(iter(dataloader))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i,ax in enumerate(axes.flatten()):
        img = unnormalize(images[i], mean, std)
        img = img.permute(1, 2, 0)
        ax.imshow(img)
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis("off")
    plt.savefig("sample_batch.png", dpi=150, bbox_inches="tight")
    plt.close()
def compute_mean_std(dataloader,device):
    num_pixel=0
    channel_sum=0.0
    channel_squared_sum=0.0
    for images,_ in dataloader:
        images=images.to(device)
        b,c,h,w=images.shape
        pixels=b*h*w
        channel_sum=images.sum(dim=[0,2,3])
        channel_squared_sum+=(images**2).sum(dim=[0,2,3])
        num_pixel+=pixels
    mean=channel_sum/num_pixel
    std = (channel_squared_sum / num_pixel - mean ** 2).sqrt()
    return mean,std
def get_class_names(dataset_path:str):
    return sorted(list(os.listdir(dataset_path)))
def plot_class_distribution(dataset:torch.utils.data.Dataset,classnames=None):
    labels=[label for _,label in dataset]
    counts=torch.bincount(torch.tensor(labels))
    x=range(len(counts))
    y=counts.tolist()
    if classnames:
        y=classnames
    plt.figure(figsize=(8, 5))
    plt.bar(x, y)
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_dataset_samples(dataset:torch.utils.data.Dataset,cls:dict,num_images=8,num_rows=2,figsize=(8,8)):
    images=[]
    labels=[]
    for i in range(num_images):
        img,label=dataset[i]
        if not isinstance(img,Image.Image):
           img=torchvision.transforms.ToPILImage()(img)
        images.append(img)
        labels.append(labels)
    ncol=num_rows
    num_rows = (num_images + ncol - 1) // ncol
    plt.figure(figsize)
    for i,(image,label) in enumerate(zip(image,label)):
        plt.subplot(num_rows,ncol,i+1)
        plt.imshow(image)
        plt.title(cls[label])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

