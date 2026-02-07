from dataclasses import dataclass
@dataclass
class ModelConfig:
    pretrained_weights:str
    head_dim: list[int]
    num_classes:int
    drop_out:float
