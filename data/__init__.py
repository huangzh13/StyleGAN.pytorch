"""
-------------------------------------------------
   File Name:    __init__.py.py
   Author:       Zhonghao Huang
   Date:         2019/10/22
   Description:
-------------------------------------------------
"""

from data.datasets import FlatDirectoryImageDataset, FoldersDistributedDataset
from data.transforms import get_transform


def make_dataset(cfg):
    if cfg.DATASET.FOLDER:
        Dataset = FoldersDistributedDataset
    else:
        Dataset = FlatDirectoryImageDataset

    _dataset = Dataset(data_dir=cfg.DATASET.IMG_DIR, transform=get_transform(new_size=cfg.RESOLUTION))

    return _dataset
