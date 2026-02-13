import torch as nn
import tqdm
import torch
import mlflow
from dataclasses import asdict
from config.expiremntconfig import ExpirementConfig
def train_one_epoch(model:nn.Module,loader:torch.utils.data.DataLoader,optimizer:torch.optim.Optimizer,criterion:nn.Module,device:torch.device):
    model.train()
    loss_sum,correct=0.0,0
    for x,y in tqdm.tqdm(loader):
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad()
        out=model(x)
        loss=criterion(out,y)
        loss.backward()
        optimizer.step()
        loss_sum+=loss.item()
        correct += (out.argmax(1) == y).sum().item()
    return (loss_sum/len(loader.dataset),correct/len(loader.dataset))
@torch.no_grad()
def evaluate_one_epoch(model:nn.Module,loader:torch.utils.data.DataLoader,criterion:nn.Module,device:torch.device,epoch:int):
    model.eval()
    val_loss=0.0,val_correct=0
    for x,y in tqdm.tqdm(loader,desc=f"validating epoch{epoch}"):
        x,y=x.to(device),y.to(device)
        out=model(x)
        loss=criterion(out,y)
        val_loss+=loss.item()*x.size(0)
        preds=out.argmax(dim=1)
        val_correct += (preds == y).sum().item()
    return(val_loss/len(loader.dataset),val_correct/len(loader.dataset))

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    num_epochs: int = 10,
    device: torch.device = torch.device("cuda"),
    save_path: str = None,
    mlflow_logging: bool = True,
) -> dict:
    model.to(device)
    history={"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}
    best_val_loss=float("inf")
    for i in tqdm.tqdm(range(num_epochs),desc=f"Training the model in epoch {i+1}/{num_epochs}"):
        train_loss,train_acc=train_one_epoch(model,train_loader,optimizer,criterion,device)
        val_loss,val_acc=evaluate_one_epoch(model,val_loader,criterion,device,i+1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch [{i+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if scheduler:
            scheduler.step(val_loss)
        if mlflow_logging:
            mlflow.log_metric("train_accuracy",train_acc)
            mlflow.log_metric("train_loss",train_loss)
            mlflow.log_metric("validation_accuracy",val_acc)
            mlflow.log_metric("validation_loss",val_loss)
        if val_loss <best_val_loss:
            best_val_loss=val_loss
            torch.save(model.state_dict(),save_path)
            print(f"Saving the model in epoch {i+1}")
        return history
def flatted_data_class(dc):
    flattend={}
    for key,value in asdict(dc).items():
        if isinstance(value,dict):
            for sub_key,sub_value in value.items():
                flattend[f"{key}.{sub_key}"]=sub_value
        else:
            flattend[key]=value

def run_expirement(
    model: nn.Module,
    expirement_name:str,
    train_loader: torch.utils.data.DataLoader,
    params:ExpirementConfig,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    num_epochs: int = 10,
    device: torch.device = torch.device("cuda"),
    save_path: str = None,
    mlflow_logging: bool = True
):
    
    mlflow.set_experiment(experiment_name=expirement_name)
    mlflow.log_params(flatted_data_class(params))
    with mlflow.start_run():
        history=train_model(model,train_loader,val_loader,criterion,optimizer,scheduler,num_epochs,device,save_path,mlflow_logging)

def train_model(model:nn.Module,train_loader:torch.utils.data.DataLoader,val_loader:torch.utils.data.DataLoader,critreon:nn.Module,
                optimizer:torch.optim.Optimizer,scheduler=None,num_epochs:int=10,device:torch.device=torch.device("cuda"),save_path:str=None)->dict:
    model.to(device)
    history={
        "train_loss":[],"val_loss":[],"train_lacc":[],"val_acc":[]
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