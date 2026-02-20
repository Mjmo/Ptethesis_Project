import pytest
from mydata.gettraintest import get_data_loader, get_dataset
import yaml
config=None
with open("/content/Ptethesis_Project/src/configs/expiremnt_config.yaml","r") as f:
    config=yaml.safe_load(f)
    

@pytest.mark.parametrize(
    "folder_path,batchsize,testsize,minsamples",
    [
        (config["data"]["train_data_path"], 8, 0.2, 50),
        (config["data"]["train_data_path"], 16, 0.3, 100),
        (config["data"]["train_data_path"], 32, 0.4, 120),
        (config["data"]["train_data_path"], 128, 0.2, 200),
    ]
)
def test_data_leakage(folder_path, batchsize, testsize, minsamples):
    dataset = get_dataset(folder_path)

    train_loader, valid_loader = get_data_loader(
        dataset,
        minsamples,
        train_aug=lambda x: x,
        valid_aug=lambda x: x,
        batch_size=batchsize,
        test_size=testsize,
        num_workers=0,
        seed=config["seed"]
    )

    train_indices = set(train_loader.dataset.indices)
    val_indices = set(valid_loader.dataset.indices)

    assert train_indices.isdisjoint(val_indices)