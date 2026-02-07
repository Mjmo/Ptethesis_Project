from dataclasses import dataclass
@dataclass
class Training_Config:
    num_epochs:int
    save_path:str
    lr_head:float
    lr_base:float
