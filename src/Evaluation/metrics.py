from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import torch
import tqdm
import torch.nn as nn 
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
import os
from torch.utils.data import dataloader
from sklearn.metrics import classification_report
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
def plot_confusion_matrix(labels,preds,class_names=None,normalize=True,figsize=(6,5)):
    cm=confusion_matrix(labels,preds)
    if normalize:
        cm_to_plot=cm.astype("float")/cm.sum(axis=1)[:,np.newaxis]
    else:
        cm_to_plot=cm
    plt.figure(figsize=figsize)
    sns.heatmap(cm_to_plot,annot=True,xticklabels=class_names,yticklabels=class_names)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title("Confusion MAtrix")
    plt.show()

def validate_multiclass(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
):
    model.eval()

    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(dataloader, desc="Validating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            val_loss += loss.item() * inputs.size(0) 

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(dataloader.dataset)  
    val_acc = accuracy_score(all_labels, all_preds)

    return val_loss, val_acc, all_labels, all_preds
def show_misclassfied(model:nn.Module,dataloader:torch.utils.data.DataLoader,classes:list[str],device:torch.device,num_images:int=10):
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    with torch.no_grad():
        for images, labels in tqdm.tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            mis_idx = (preds != labels).nonzero(as_tuple=False).squeeze()

            if mis_idx.numel() == 0:
                continue

            for idx in mis_idx:
                misclassified_images.append(images[idx].cpu())
                misclassified_labels.append(labels[idx].cpu())
                misclassified_preds.append(preds[idx].cpu())
                if len(misclassified_labels) >= num_images:
                    break
            if len(misclassified_labels) >= num_images:
                break

        if len(misclassified_labels) == 0:
            print("No misclassified images found!")
            return

    for i in range(len(misclassified_labels)):
        npimg = misclassified_images[i].numpy()
        plt.figure(figsize=(4, 4))
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(f"Predicted: {classes[misclassified_preds[i].item()]} | True: {classes[misclassified_labels[i].item()]}")
        plt.axis('off')
        plt.show()
def plot_learning_curve(history: dict, save_path: str = None, show: bool = True):

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(epochs, history['train_loss'], 'o-', label='Train Loss')
    axs[0].plot(epochs, history['val_loss'], 's-', label='Validation Loss')
    axs[0].set_title("Loss over Epochs")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(epochs, history['train_acc'], 'o-', label='Train Accuracy')
    axs[1].plot(epochs, history['val_acc'], 's-', label='Validation Accuracy')
    axs[1].set_title("Accuracy over Epochs")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    if show:
        plt.show()
    
    plt.close(fig)

def save_classification_report(y_true, y_pred, class_names=None, save_path="classification_report.txt"):

    report_str = classification_report(y_true, y_pred,labels=range(43) ,target_names=class_names)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(report_str)
    