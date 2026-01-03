import torch
import torch.nn as nn
from tqdm import tqdm
def train_model(model:nn.Module,train_loader:torch.utils.data.DataLoader,val_loader:torch.utils.data.DataLoader,critreon:nn.Module,
                optimizer:torch.optim.Optimizer,scheduler=None,num_epochs:int=10,device:torch.device=torch.device("cuda"),save_path:str=None)->dict:
    model.to(device)
    history={
        "train_loss":[],"val_loss":[],"train_acc":[],"val_acc":[]
    }
    best_val_loss=float("inf")
    for epoch in range(num_epochs):
        model.train()
        running_loss=0.0
        running_correct=0
        for inputs,labels in tqdm(train_loader,desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs,labels=inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=critreon(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()*inputs.size(0)
            _,preds=torch.max(outputs,1)
            running_correct+=torch.sum(preds==labels).item()
        epoch_train_loss=running_loss/len(train_loader.dataset)
        epoch_train_acc = running_correct / len(train_loader.dataset)
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        model.eval()
        val_loss=0.0
        val_corrects=0
        with torch.no_grad():
            for inputs,labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs,labels=inputs.to(device),labels.to(device)
                outputs=model(inputs)
                loss=critreon(outputs,labels)
                val_loss+=loss.item()*inputs.size(0)
                _,preds=torch.max(outputs,1)
                val_corrects+=torch.sum(preds==labels).item()
            epoch_val_loss=val_loss/len(val_loader.dataset)
            epoch_val_acc=val_corrects/len(val_loader.dataset)
            history["val_loss"].append(epoch_val_loss)
            history["val_acc"].append(epoch_val_acc)
        if scheduler:
            scheduler.step(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        if save_path and epoch_val_loss<best_val_loss:
            best_val_loss=epoch_val_loss
            torch.save(model.state_dict,save_path)
            print(f"Saved best model at epoch {epoch+1}")
    return history