import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch
from torchvision.transforms import ToPILImage
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
