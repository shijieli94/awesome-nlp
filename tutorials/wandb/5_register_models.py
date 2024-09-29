# Model Registry Tutorial
# The model registry is a central place to house and organize all the model tasks and their associated artifacts being worked on across an org:
# - Model checkpoint management
# - Document your models with rich model cards
# - Maintain a history of all the models being used/deployed
# - Facilitate clean hand-offs and stage management of models
# - Tag and organize various model tasks
# - Set up automatic notifications when models progress
#
# This tutorial will walk through how to track the model development lifecycle for a simple image classification task.

# Log in to your W&B account
import wandb

wandb.login()

# In this tutorial, the first thing we will do is download a training dataset and log it as an artifact to be used downstream in the training job.

import sys
from pathlib import Path

# FORM VARIABLES
PROJECT_NAME = "model-registry-tutorial"
ENTITY = wandb.api.default_entity  # replace with your Team name or username

# Dataset constants
DATASET_NAME = "nature_100"
DATA_DIR = Path(sys.path[0]) / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_SRC = DATA_DIR / DATASET_NAME
IMAGES_PER_LABEL = 10
BALANCED_SPLITS = {"train": 8, "val": 1, "test": 1}
MODEL_TYPE = "squeezenet"

# Let's grab a version of our Dataset

import io
import zipfile

import requests

# Download the dataset from a bucket
src_url = f"https://storage.googleapis.com/wandb_datasets/{DATASET_NAME}.zip"
src_zip = f"{DATASET_NAME}.zip"

# Download the zip file from W&B
r = requests.get(src_url)

# Create a file object using the string data
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(path=DATA_DIR)

# We are going to generate a file containing the image

with wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type="log_datasets") as run:
    train_art = wandb.Artifact(
        name=DATASET_NAME, type="raw_images", description="nature image dataset with 10 classes, 10 images per class"
    )
    train_art.add_dir(DATA_SRC)
    wandb.log_artifact(train_art)

import math

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms


class NatureDataset(Dataset):
    def __init__(self, artifact_name_alias: str, transform=None):
        self.transform = transform

        # Pull down the artifact locally to load it into memory
        art = wandb.use_artifact(artifact_name_alias)
        self.path_at = Path(art.download())

        self.img_paths = list(DATA_SRC.rglob("*.jpg"))
        labels = [image_path.parent.name for image_path in self.img_paths]
        self.class_names = sorted(set(labels))
        self.idx_to_class = {k: v for k, v in enumerate(self.class_names)}
        self.class_to_idx = {v: k for k, v in enumerate(self.class_names)}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = Path(self.path_at) / self.img_paths[idx]

        image = Image.open(image_path)
        label = image_path.parent.name
        label = torch.tensor(self.class_to_idx[label], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


class Dataloaders:
    def __init__(self, artifact_name_alias: str, batch_size: int, input_size: int, seed: int = 42):
        self.artifact_name_alias = artifact_name_alias
        self.batch_size = batch_size
        self.input_size = input_size
        self.seed = seed

        tfms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.CenterCrop(self.input_size),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        print(f"Setting up data from artifact: {self.artifact_name_alias}")
        self.dataset = NatureDataset(artifact_name_alias=self.artifact_name_alias, transform=tfms)

        nature_length = len(self.dataset)
        train_size = math.floor(0.8 * nature_length)
        val_size = math.floor(0.2 * nature_length)
        print(f"Splitting dataset into {train_size} training samples and {val_size} validation samples")
        self.ds_train, self.ds_valid = random_split(
            self.dataset, [train_size, val_size], generator=torch.Generator().manual_seed(self.seed)
        )

        self.train = DataLoader(self.ds_train, batch_size=self.batch_size)
        self.valid = DataLoader(self.ds_valid, batch_size=self.batch_size)


import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(num_classes, feature_extract, use_pretrained=True):
    "Create a model from torchvision.models"
    model_ft = None

    # SqueezeNet
    model_ft = models.squeezenet1_0(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model_ft.num_classes = num_classes

    return model_ft, 224


class NaturePyTorchModule(torch.nn.Module):
    def __init__(self, model_name, num_classes=10, feature_extract=True, lr=0.01):
        """method used to define our model parameters"""
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_extract = feature_extract
        self.lr = lr
        self.model, self.input_size = initialize_model(num_classes=self.num_classes, feature_extract=True)

    def forward(self, x):
        """method used for inference input -> output"""
        return self.model(x)


def evaluate_model(model, val_dl, idx_to_class, class_names):
    device = torch.device("cpu")
    model.eval()
    test_loss = 0
    correct = 0
    preds = []
    actual = []

    val_table = wandb.Table(columns=["pred", "actual", "image"])

    with torch.no_grad():
        for data, target in val_dl:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            preds += list(pred.flatten().tolist())
            actual += target.numpy().tolist()
            correct += pred.eq(target.view_as(pred)).sum().item()

            for idx, img in enumerate(data):
                img = img.numpy().transpose(1, 2, 0)
                pred_class = idx_to_class[pred.numpy()[idx][0]]
                target_class = idx_to_class[target.numpy()[idx]]
                val_table.add_data(pred_class, target_class, wandb.Image(img))

    test_loss /= len(val_dl.dataset)
    accuracy = 100.0 * correct / len(val_dl.dataset)
    conf_mat = wandb.plot.confusion_matrix(y_true=actual, preds=preds, class_names=class_names)
    return test_loss, accuracy, preds, val_table, conf_mat


run = wandb.init(
    project=PROJECT_NAME,
    entity=ENTITY,
    job_type="training",
    config={"model_type": MODEL_TYPE, "lr": 1.0, "gamma": 0.75, "batch_size": 16, "epochs": 5},
)

model = NaturePyTorchModule(wandb.config["model_type"])

wandb.config["input_size"] = 224

dls = Dataloaders(
    artifact_name_alias=f"{DATASET_NAME}:latest",
    batch_size=wandb.config["batch_size"],
    input_size=wandb.config["input_size"],
)

# Train the model
learning_rate = wandb.config["lr"]
gamma = wandb.config["gamma"]
epochs = wandb.config["epochs"]

device = torch.device("cpu")
optimizer = optim.Adadelta(model.parameters(), lr=wandb.config["lr"])
scheduler = StepLR(optimizer, step_size=1, gamma=wandb.config["gamma"])

best_loss = float("inf")
best_model = None

for epoch_ndx in range(epochs):
    model.train()
    for batch_ndx, batch in enumerate(dls.train):
        data, target = batch[0].to("cpu"), batch[1].to("cpu")
        optimizer.zero_grad()
        preds = model(data)
        loss = F.cross_entropy(preds, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        ### Log your metrics ###
        wandb.log(
            {
                "train/epoch_ndx": epoch_ndx,
                "train/batch_ndx": batch_ndx,
                "train/train_loss": loss,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(f"Epoch: {epoch_ndx}, Batch: {batch_ndx}, Loss: {loss}")

    ### Evaluation at the end of each epoch ###
    test_loss, accuracy, preds, val_table, conf_mat = evaluate_model(
        model,
        dls.valid,
        dls.dataset.idx_to_class,
        dls.dataset.class_names,
    )

    is_best = test_loss < best_loss

    wandb.log(
        {
            "eval/test_loss": test_loss,
            "eval/accuracy": accuracy,
            "eval/conf_mat": conf_mat,
            "eval/val_table": val_table,
        }
    )

    ### Checkpoing your model weights ###
    torch.save(model.state_dict(), "model.pth")
    art = wandb.Artifact(
        f"nature-{wandb.run.id}",
        type="model",
        metadata={
            "format": "onnx",
            "num_classes": len(dls.dataset.class_names),
            "model_type": wandb.config["model_type"],
            "model_input_size": wandb.config["input_size"],
            "index_to_class": dls.dataset.idx_to_class,
        },
    )

    art.add_file("model.pth")

    ### Add aliases to keep track of your best checkpoints over time
    wandb.log_artifact(art, aliases=["best", "latest"] if is_best else None)
    if is_best:
        best_model = art


if ENTITY is not None:
    wandb.run.link_artifact(best_model, f"{ENTITY}/model-registry/Model Registry Tutorial", aliases=["staging"])
else:
    print("Must indicate entity where Registered Model will exist")
wandb.finish()

run = wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type="inference")
artifact = run.use_artifact(f"{ENTITY}/model-registry/Model Registry Tutorial:staging", type="model")
artifact_dir = artifact.download()
wandb.finish()
