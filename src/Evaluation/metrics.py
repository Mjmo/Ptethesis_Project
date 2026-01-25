from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import dataloader
def plot_multi_label_confusion(y_true:np.ndarray,y_predicted:np.ndarray,class_names:list):
    num_classes = len(class_names)

    for i in range(num_classes):
        cm = confusion_matrix(
            y_true[:, i],
            y_predicted[:, i]
        )

        plt.figure()
        plt.imshow(cm)
        plt.title(f"Confusion Matrix — {class_names[i]}")
        plt.colorbar()
        plt.xticks([0, 1], ["0", "1"])
        plt.yticks([0, 1], ["0", "1"])

        for x in range(2):
            for y in range(2):
                plt.text(
                    y, x,
                    cm[x, y],
                    ha="center",
                    va="center"
                )

        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

def validate_multiclass(model:torch.nn.Module, dataloader:dataloader, criterion:torch.nn.Module, device:dataloader):
    model.eval()

    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
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
