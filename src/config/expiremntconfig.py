from .dataclass import DataConfig
from .modelclass import ModelConfig
from .trainingconfig import Training_Config
from dataclasses import dataclass
@dataclass
class ExpirementConfig:
    data:DataConfig
    model: ModelConfig
    training:Training_Config
    seed:int