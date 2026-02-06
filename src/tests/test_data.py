import pytest
from mydata.gettraintest import get_data_loader, get_dataset

@pytest.mark.parametrize(
    "folder_path,batchsize,testsize,minsamples",
    [
        ("/content/drive/MyDrive/IOW_annotated_images_202601", 8, 0.2, 50),
        ("/content/drive/MyDrive/IOW_annotated_images_202601", 16, 0.3, 100),
        ("/content/drive/MyDrive/IOW_annotated_images_202601", 32, 0.4, 120),
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
        num_workers=0
    )

    train_indices = set(train_loader.dataset.indices)
    val_indices = set(valid_loader.dataset.indices)

    assert train_indices.isdisjoint(val_indices)