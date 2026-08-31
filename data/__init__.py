"""
-------------------------------------------------
   File Name:    __init__.py.py
   Author:       Zhonghao Huang
   Date:         2019/10/22
   Description:
-------------------------------------------------
"""

from torchvision.datasets import ImageFolder

from data.datasets import FlatDirectoryImageDataset, FoldersDistributedDataset
from data.transforms import get_transform


def make_dataset(cfg, conditional=False, n_classes=None):
    """Build the dataset used by the training loop.

    Conditional training uses :class:`torchvision.datasets.ImageFolder`, whose
    subdirectory names define the integer labels. Validate the configured
    number of classes here so a mismatch is reported before the first model
    forward pass.
    """
    if conditional:
        Dataset = ImageFolder
    else:
        if cfg.folder:
            Dataset = FoldersDistributedDataset 
        else:
            Dataset = FlatDirectoryImageDataset
    
    transforms = get_transform(new_size=(cfg.resolution, cfg.resolution))
    _dataset = Dataset(cfg.img_dir, transform=transforms)

    if conditional and n_classes is not None:
        actual_classes = len(_dataset.classes)
        if actual_classes != n_classes:
            raise ValueError(
                "Conditional dataset contains {} classes, but n_classes is {}. "
                "Set n_classes to the number of class subdirectories in {}.".format(
                    actual_classes, n_classes, cfg.img_dir))

    return _dataset


def get_data_loader(dataset, batch_size, num_workers):
    """
    generate the data_loader from the given dataset
    :param dataset: dataset for training (Should be a PyTorch dataset)
                    Make sure every item is an Image, or an (Image, label)
                    pair when conditional training is enabled.
    :param batch_size: batch size of the data
    :param num_workers: num of parallel readers
    :return: dl => data_loader for the dataset
    """
    from torch.utils.data import DataLoader

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )

    return dl
