from dataclasses import dataclass
@dataclass
class DataConfig:
    train_data_path:str
    test_size: float
    batch_size:int
    min_samples:int
