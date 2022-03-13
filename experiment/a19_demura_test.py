#!/usr/bin/env python

import torch
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import numpy as np
import json
import cv2
import requests
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
import os
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
from torchvision.transforms.functional import InterpolationMode
import torch.optim as optim
import time
import pandas as pd
import random
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from vit_pytorch import ViT
from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics
import copy
from collections import defaultdict

from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics

import argparse
#####
from mlflow import mlflow, log_metric, log_param, log_artifacts
import sys

sys.path.append(".")
import experiment.augumentation as aug

#####


print(tf.__version__)
print(torch.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')


# transforms_resize = transforms.Resize([512, 512], interpolation=InterpolationMode.BILINEAR)

def get_data_transforms(resize=256, augumentation="wei_augumentation"):
    aug_dic = {
        "wei_augumentation": aug.wei_augumentation
    }

    return {
        "train": transforms.Compose([
            transforms.Resize([resize, resize], interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            aug_dic[augumentation](),
            transforms.ToTensor(),
        ]),
        "test": transforms.Compose([
            transforms.Resize([resize, resize], interpolation=InterpolationMode.BILINEAR),
            aug_dic[augumentation](),
            transforms.ToTensor()
        ])
    }


# In[14]:


class AI9_Dataset(Dataset):
    def __init__(self, feature, target, transform=None):
        self.X = feature
        self.Y = target
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.transform(self.X[idx]), self.Y[idx]


def get_data_info(t, l, image_info):
    res = []
    image_info = image_info[(image_info["train_type"] == t) & (image_info["label"] == l)]

    for path, img, label in zip(image_info["path"], image_info["name"], image_info["label"]):
        img_path = os.path.join(os.path.dirname(csv_path), path, img)
        res.append([img_path, label])
    X = []
    Y = []
    for d in res:
        X.append(Image.open(os.path.join(d[0])))
        Y.append(d[1])

    dataset = AI9_Dataset(feature=X,
                          target=Y,
                          transform=get_data_transforms(256, "wei_augumentation"))
    return dataset


def get_ds(image_info):
    ds = defaultdict(dict)
    for x in ["train", "test"]:
        for y in ["mura", "normal"]:
            if y == "mura":
                l = 1
            else:
                l = 0
            ds[x][y] = get_data_info(x, l, image_info)
    return ds


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


# In[25]:


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


def calc_cm(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true=labels, y_pred=preds >= 0.5).ravel()
    precision = tp / (tp + fp)
    recall = tn / (tn + fp)
    balanced_acc = metrics.balanced_accuracy_score(labels, preds >= 0.5)
    print(f"precision = {precision:.3f}, recall = {recall:.3f}, balanced_acc = {balanced_acc:.3f}")
    return precision, recall


# In[43]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wets = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_p = 0.0

    global dataloaders

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
                dataloaders["train"] = make_training_dataloader(ds)

            else:
                dataloaders.pop("train")
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            gts = []
            pred_list = []

            print("Sample...")
            data_it = []
            for inputs, labels in tqdm(dataloaders[phase]):
                data_it.append([inputs, labels])

            for d in tqdm(data_it):
                inputs, labels = d
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    labels = labels.to(torch.float32)
                    outputs = outputs.to(torch.float32)
                    outputs = torch.reshape(outputs, (-1,))
                    loss = criterion(outputs, labels)
                    preds = torch.Tensor([1 if x >= 0.5 else 0 for x in outputs]).cuda()

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                gt = labels.cpu()
                pred = preds.cpu()
                gts.extend(gt)
                pred_list.extend(pred)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # end epoch
                # del inputs
                # del labels
            if phase == "train":
                scheduler.step()
                # log_metric("learning rate", scheduler.get_lr(), step=epoch)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            p, r = calc_cm(pred_list, gts)

            log_metric("{}_epoch_loss".format(phase), epoch_loss, step=epoch)
            log_metric("{}_epoch_acc".format(phase), float(epoch_acc.cpu()), step=epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc and p >= best_p:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == "train" and epoch_acc > 0.995:
                break

        print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best test Acc: {:4f}".format(best_acc))

    log_metric("Best test Acc", best_acc)
    # log_metric("Training time",  "{:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    # mlflow.pytorch.set_log_model_display_name(display_name="a19_model")
    mlflow.pytorch.log_model(best_model_wts, "model")

    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dataset', type=str, default="./data_merged.csv")
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--parent_name', type=str, default="parent")
    parser.add_argument('--output', type=str, default="./output")
    args = parser.parse_args()

    # csv_path = "/home/tedbest/datadisk/a19/repo_to_upload/ai9_mura_dataset_2022_backup2/20220210_merged_258/data_merged.csv"
    csv_path = args.csv_dataset
    image_info = pd.read_csv(csv_path)

    ds = get_ds(image_info)

    dataloaders = {
        "train": make_training_dataloader(ds),
        "test": make_test_dataloader(ds)
    }

    # output = "./output"
    output = args.output
    os.makedirs(output, exist_ok=True)

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ["train", "test"]}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    setup_seed(42)

    ###############
    mlflow.a19_parent_run(
        experiment_name=args.experiment_name,
        parent_run_name=args.parent_name
    )
    print("#" * 30)
    print("Experiment:", args.experiment_name)
    print("Parent name:", args.parent_name)
    print("#" * 30)

    ###############
    for i in range(5):
        mlflow.a19_child_run(child_run_name="child_{}".format(i))

        mod = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', "nvidia_resnet50")
        mod.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512, bias=True),
            nn.Linear(in_features=512, out_features=16, bias=True),
            nn.Linear(in_features=16, out_features=1, bias=True),
        )

        mod = mod.to(device)
        criterion = nn.BCEWithLogitsLoss()

        EPOCHS = 5

        LR = 2e-4 * i
        optimizer = optim.Adam(mod.parameters(), lr=2e-4 * i)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(EPOCHS / 5), eta_min=2e-7)

        # model_ft = train_model(mod, criterion, optimizer, scheduler, num_epochs=EPOCHS)

        # torch.save(model_ft, os.path.join(output, "model_ft_{}.pt".format(i)))

        log_param("test", 123)
        #mlflow.pytorch.set_log_model_display_name("model_{}".format(i))
        #mlflow.pytorch.log_model(mod, "model")

        mlflow.end_run()

    mlflow.end_run()
