from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import torch
import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import dataloader
from sklearn.metrics import classification_report
def plot_confusion_matrix(labels,preds,class_names=None,normalize=True,figsize=(6,5)):
    cm=confusion_matrix(labels,preds)
    if normalize:
        cm_to_plot=cm.astype("float")/cm.sum(axis=1)[:,np.newaxis]
        fmt='.3f'
    else:
        cm_to_plot=cm
        fmt='d'
    plt.figure(figsize=figsize)
    sns.heatmap(cm_to_plot,annot=True,fmt=fmt,xticklabels=class_names,yticklabels=class_names)
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