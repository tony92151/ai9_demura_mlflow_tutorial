import os
import time

import torchvision
from random import random, randint
from mlflow import mlflow, log_metric, log_param, log_artifacts
from tqdm import tqdm
from mlflow.tracking import MlflowClient

# Create an experiment with a name that is unique and case sensitive

import numpy as np
import torch
import mlflow.pytorch

class LinearNNModel(torch.nn.Module):
    def __init__(self):
        super(LinearNNModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


if __name__ == "__main__":
    # mlflow.set_tracking_uri('http://localhost:5000')
    print("Running mlflow_tracking.py")

    mlflow.start_run(run_name="parent")
    mlflow.start_run(run_name="child", nested=True)

    mlflow.log_param("learning rate", 0.1)
    mlflow.log_param("path", "./data")
    mlflow.log_param("some", {"n_estimators": 3, "random_state": 42})

    for i in tqdm(range(20)):
        log_metric("param1", randint(0, 100), step=i)
        time.sleep(0.1)

    log_metric("foo", random())
    log_metric("foo", random() + 1)
    log_metric("foo", random() + 2)

    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")

    log_artifacts("outputs")

    #mlflow.pytorch.set_log_model_display_name(display_name="test")
    mlflow.pytorch.log_model(LinearNNModel(), "model1")

    mlflow.end_run()
    mlflow.end_run()

    client = MlflowClient()
    # client.
