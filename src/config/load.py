from .dataclass import DataConfig
from .expiremntconfig import ExpirementConfig
from .trainingconfig import Training_Config
from .modelclass import ModelConfig
from pathlib import Path
import yaml
def load_config(path:str)->ExpirementConfig:
    with open(path,"r") as file:
        raw=yaml.safe_load(file)
    return ExpirementConfig(seed=raw["seed"],data=DataConfig(**raw["model"]),model=ModelConfig(**raw["model"]),trainig=Training_Config(**raw["training"]))



