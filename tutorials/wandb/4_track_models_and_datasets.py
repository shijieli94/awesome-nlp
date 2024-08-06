import os
import random

import torch
import torchvision
import wandb
from torch.utils.data import TensorDataset

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data parameters
num_classes = 10
input_shape = (1, 28, 28)

# drop slow mirror from list of MNIST mirrors
torchvision.datasets.MNIST.mirrors = [
    mirror for mirror in torchvision.datasets.MNIST.mirrors if not mirror.startswith("http://yann.lecun.com")
]


def load(train_size=50_000):
    """
    # Load the data
    """

    # the data, split between train and test sets
    train = torchvision.datasets.MNIST("./", train=True, download=True)
    test = torchvision.datasets.MNIST("./", train=False, download=True)
    (x_train, y_train), (x_test, y_test) = (train.data, train.targets), (test.data, test.targets)

    # split off a validation set for hyperparameter tuning
    x_train, x_val = x_train[:train_size], x_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]

    training_set = TensorDataset(x_train, y_train)
    validation_set = TensorDataset(x_val, y_val)
    test_set = TensorDataset(x_test, y_test)

    datasets = [training_set, validation_set, test_set]

    return datasets


def load_and_log():
    # 🚀 start a run, with a type to label it and a project it can call home
    with wandb.init(project="artifacts-example", job_type="load-data") as run:
        datasets = load()  # separate code for loading the datasets
        names = ["training", "validation", "test"]

        # 🏺 create our Artifact
        raw_data = wandb.Artifact(
            "mnist-raw",
            type="dataset",
            description="Raw MNIST dataset, split into train/val/test",
            metadata={"source": "torchvision.datasets.MNIST", "sizes": [len(dataset) for dataset in datasets]},
        )

        for name, data in zip(names, datasets):
            # 🐣 Store a new file in the artifact, and write something into its contents.
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ Save the artifact to W&B.
        run.log_artifact(raw_data)


load_and_log()


def preprocess(dataset, normalize=True, expand_dims=True):
    """
    ## Prepare the data
    """
    x, y = dataset.tensors

    if normalize:
        # Scale images to the [0, 1] range
        x = x.type(torch.float32) / 255

    if expand_dims:
        # Make sure images have shape (1, 28, 28)
        x = torch.unsqueeze(x, 1)

    return TensorDataset(x, y)


def preprocess_and_log(steps):
    with wandb.init(project="artifacts-example", job_type="preprocess-data") as run:
        processed_data = wandb.Artifact(
            "mnist-preprocess", type="dataset", description="Preprocessed MNIST dataset", metadata=steps
        )

        # ✔️ declare which artifact we'll be using
        raw_data_artifact = run.use_artifact("mnist-raw:latest")

        # 📥 if need be, download the artifact
        raw_dataset = raw_data_artifact.download()

        for split in ["training", "validation", "test"]:
            raw_split = read(raw_dataset, split)
            processed_dataset = preprocess(raw_split, **steps)

            with processed_data.new_file(split + ".pt", mode="wb") as file:
                x, y = processed_dataset.tensors
                torch.save((x, y), file)

        run.log_artifact(processed_data)


def read(data_dir, split):
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))

    return TensorDataset(x, y)


steps = {"normalize": True, "expand_dims": True}

preprocess_and_log(steps)

from math import floor

import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(
        self,
        hidden_layer_sizes=[32, 64],
        kernel_sizes=[3],
        activation="ReLU",
        pool_sizes=[2],
        dropout=0.5,
        num_classes=num_classes,
        input_shape=input_shape,
    ):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=hidden_layer_sizes[0], kernel_size=kernel_sizes[0]),
            getattr(nn, activation)(),
            nn.MaxPool2d(kernel_size=pool_sizes[0]),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_layer_sizes[0], out_channels=hidden_layer_sizes[-1], kernel_size=kernel_sizes[-1]
            ),
            getattr(nn, activation)(),
            nn.MaxPool2d(kernel_size=pool_sizes[-1]),
        )
        self.layer3 = nn.Sequential(nn.Flatten(), nn.Dropout(dropout))

        fc_input_dims = floor((input_shape[1] - kernel_sizes[0] + 1) / pool_sizes[0])  # layer 1 output size
        fc_input_dims = floor((fc_input_dims - kernel_sizes[-1] + 1) / pool_sizes[-1])  # layer 2 output size
        fc_input_dims = fc_input_dims * fc_input_dims * hidden_layer_sizes[-1]  # layer 3 output size

        self.fc = nn.Linear(fc_input_dims, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        return x


def build_model_and_log(config):
    with wandb.init(project="artifacts-example", job_type="initialize", config=config) as run:
        config = wandb.config

        model = ConvNet(**config)

        model_artifact = wandb.Artifact(
            "convnet", type="model", description="Simple AlexNet style CNN", metadata=dict(config)
        )

        torch.save(model.state_dict(), "initialized_model.pth")
        # ➕ another way to add a file to an Artifact
        model_artifact.add_file("initialized_model.pth")

        wandb.save("initialized_model.pth")

        run.log_artifact(model_artifact)


model_config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

build_model_and_log(model_config)

import torch.nn.functional as F


def train(model, train_loader, valid_loader, config):
    optimizer = getattr(torch.optim, config.optimizer)(model.parameters())
    model.train()
    example_ct = 0
    for epoch in range(config.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            example_ct += len(data)

            if batch_idx % config.batch_log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0%})]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

                train_log(loss, example_ct, epoch)

        # evaluate the model on the validation set at each epoch
        loss, accuracy = test(model, valid_loader)
        test_log(loss, accuracy, example_ct, epoch)


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum")  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)

    accuracy = 100.0 * correct / len(test_loader.dataset)

    return test_loss, accuracy


def train_log(loss, example_ct, epoch):
    loss = float(loss)

    # where the magic happens
    wandb.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


def test_log(loss, accuracy, example_ct, epoch):
    loss = float(loss)
    accuracy = float(accuracy)

    # where the magic happens
    wandb.log({"epoch": epoch, "validation/loss": loss, "validation/accuracy": accuracy}, step=example_ct)
    print(f"Loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}/{accuracy:.3f}")


def evaluate(model, test_loader):
    """
    ## Evaluate the trained model
    """

    loss, accuracy = test(model, test_loader)
    highest_losses, hardest_examples, true_labels, predictions = get_hardest_k_examples(model, test_loader.dataset)

    return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions


def get_hardest_k_examples(model, testing_set, k=32):
    model.eval()

    loader = DataLoader(testing_set, 1, shuffle=False)

    # get the losses and predictions for each item in the dataset
    losses = None
    predictions = None
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            pred = output.argmax(dim=1, keepdim=True)

            if losses is None:
                losses = loss.view((1, 1))
                predictions = pred
            else:
                losses = torch.cat((losses, loss.view((1, 1))), 0)
                predictions = torch.cat((predictions, pred), 0)

    argsort_loss = torch.argsort(losses, dim=0)

    highest_k_losses = losses[argsort_loss[-k:]]
    hardest_k_examples = testing_set[argsort_loss[-k:].cpu()][0]
    true_labels = testing_set[argsort_loss[-k:].cpu()][1]
    predicted_labels = predictions[argsort_loss[-k:]]

    return highest_k_losses, hardest_k_examples, true_labels, predicted_labels


from torch.utils.data import DataLoader


def train_and_log(config):
    with wandb.init(project="artifacts-example", job_type="train", config=config) as run:
        config = wandb.config

        data = run.use_artifact("mnist-preprocess:latest")
        data_dir = data.download()

        training_dataset = read(data_dir, "training")
        validation_dataset = read(data_dir, "validation")

        train_loader = DataLoader(training_dataset, batch_size=config.batch_size)
        validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size)

        model_artifact = run.use_artifact("convnet:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "initialized_model.pth")
        model_config = model_artifact.metadata
        config.update(model_config)

        model = ConvNet(**model_config)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)

        train(model, train_loader, validation_loader, config)

        model_artifact = wandb.Artifact(
            "trained-model", type="model", description="Trained NN model", metadata=dict(model_config)
        )

        torch.save(model.state_dict(), "trained_model.pth")
        model_artifact.add_file("trained_model.pth")
        wandb.save("trained_model.pth")

        run.log_artifact(model_artifact)

    return model


def evaluate_and_log(config=None):
    with wandb.init(project="artifacts-example", job_type="report", config=config) as run:
        data = run.use_artifact("mnist-preprocess:latest")
        data_dir = data.download()
        testing_set = read(data_dir, "test")

        test_loader = torch.utils.data.DataLoader(testing_set, batch_size=128, shuffle=False)

        model_artifact = run.use_artifact("trained-model:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "trained_model.pth")
        model_config = model_artifact.metadata

        model = ConvNet(**model_config)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        loss, accuracy, highest_losses, hardest_examples, true_labels, preds = evaluate(model, test_loader)

        run.summary.update({"loss": loss, "accuracy": accuracy})

        wandb.log(
            {
                "high-loss-examples": [
                    wandb.Image(hard_example, caption=str(int(pred)) + "," + str(int(label)))
                    for hard_example, pred, label in zip(hardest_examples, preds, true_labels)
                ]
            }
        )


train_config = {"batch_size": 128, "epochs": 5, "batch_log_interval": 25, "optimizer": "Adam"}

model = train_and_log(train_config)
evaluate_and_log()
