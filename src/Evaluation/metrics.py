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
import math
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
def plot_learning_curve(history:dict):
    epochs=range(1,len(history["train_loss"]+1))
    plt.figure()
    plt.plot(epochs,history["train_loss"],label="Train_loss")
    plt.plot(epochs,history["val_loss"],label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(list(epochs))
    plt.legend()
    plt.title("Learning Loss")
    plt.show()

def validate_multiclass(model:torch.nn.Module, dataloader:dataloader, criterion:torch.nn.Module, device:dataloader):
    model.eval()

    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds)

    return val_loss, val_acc, all_labels, all_preds
def print_classification_report(labels,preds,class_names):
    labels = list(range(40))
    report = classification_report(labels, preds,zero_division=True,target_names=class_names,labels=labels)
    print(report)
def show_misclassfied(model:nn.Module,dataloader:torch.utils.data.DataLoader,classes:list[str],device:torch.device,num_images:int=10):
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    # Collect misclassified images
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

def plot_multiclass_pr(y_true, y_probs, class_names=None):
    """
    Plot Precision-Recall curves for multi-class classification.

    Parameters:
    - y_true: torch.Tensor of shape (N,) with class indices
    - y_probs: torch.Tensor of shape (N, C) with probabilities (softmax outputs)
    - class_names: list of class names (length C)
    """
    num_classes = y_probs.shape[1]
    
    # Convert true labels to one-hot
    y_true_onehot = torch.nn.functional.one_hot(y_true, num_classes=num_classes)
    
    plt.figure(figsize=(10,8))
    
    for i in range(num_classes):
        # Compute PR curve
        precision, recall, _ = precision_recall_curve(y_true_onehot[:,i].numpy(), y_probs[:,i].detach().numpy())
        # Compute Average Precision (AP)
        ap = average_precision_score(y_true_onehot[:,i].numpy(), y_probs[:,i].detach().numpy())
        label = f'{class_names[i]} (AP={ap:.2f})' if class_names else f'Class {i} (AP={ap:.2f})'
        plt.plot(recall, precision, lw=2, label=label)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Multi-class)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show(  )
    
def plot_comparison_matrix(y_true,y_preds,true_names,pred_names,figure_size=(20,20)):
        cm=confusion_matrix(y_true,y_preds,normalize="true")
        plt.figure(figsize=figure_size)
        sns.heatmap(cm,annot=True,xticklabels=pred_names,yticklabels=true_names)
        plt.xlabel("True labels")
        plt.ylabel("Predicted name")
        plt.title("Testing orginal model on our data")
        plt.show()
