import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import random


def make_training_dataloader(ds):
    mura_ds = ds["train"]["mura"]
    normal_ds = ds["train"]["normal"]
    min_len = min(len(mura_ds), len(normal_ds))
    sample_num = int(1 * min_len)
    normal_ds = torch.utils.data.Subset(normal_ds, random.sample(list(range(len(normal_ds))), sample_num))
    train_ds = torch.utils.data.ConcatDataset([mura_ds, normal_ds])
    dataloader = DataLoader(train_ds,
                            batch_size=4,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True,
                            )
    return dataloader


def make_test_dataloader(ds):
    m = ds["test"]["mura"]
    n = ds["test"]["normal"]
    test_ds = torch.utils.data.ConcatDataset([m, n])
    dataloader = DataLoader(test_ds,
                            batch_size=4,
                            shuffle=False,
                            num_workers=0,
                            )
    return dataloader


def make_val_dataloader(ds):
    m = ds["val"]["mura"]
    n = ds["val"]["normal"]
    val_ds = torch.utils.data.ConcatDataset([m, n])
    dataloader = DataLoader(val_ds,
                            batch_size=4,
                            shuffle=False,
                            num_workers=0,
                            )
    return dataloader