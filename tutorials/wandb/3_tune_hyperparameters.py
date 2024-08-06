import wandb

wandb.login()

# The first thing we need to define is the method for choosing new parameter values.
#
# We provide the following search methods:
#
# grid Search – Iterate over every combination of hyperparameter values. Very effective, but can be computationally costly.
# random Search – Select each new combination at random according to provided distributions. Surprisingly effective!
# bayesian Search – Create a probabilistic model of metric score as a function of the hyperparameters, and choose parameters
#       with high probability of improving the metric. Works well for small numbers of continuous parameters but scales poorly.
# We'll stick with random.

sweep_config = {"method": "random"}

# For bayesian Sweeps, you also need to tell us a bit about your metric. We need to know its name,
# so we can find it in the model outputs and we need to know whether your goal is to minimize it
# (e.g. if it's the squared error) or to maximize it (e.g. if it's the accuracy).

metric = {"name": "loss", "goal": "minimize"}

sweep_config["metric"] = metric

# Once you've picked a method to try out new values of the hyperparameters, you need to define what those parameters are.
#
# Most of the time, this step is straightforward: you just give the parameter a name and specify a list of legal values of the parameter.
#
# For example, when we choose the optimizer for our network, there's only a finite number of options. Here we stick with the
# two most popular choices, adam and sgd. Even for hyperparameters that have potentially infinite options, it usually only
# makes sense to try out a few select values, as we do here with the hidden layer_size and dropout.

parameters_dict = {
    "optimizer": {"values": ["adam", "sgd"]},
    "fc_layer_size": {"values": [128, 256, 512]},
    "dropout": {"values": [0.3, 0.4, 0.5]},
}

# It's often the case that there are hyperparameters that we don't want to vary in this Sweep, but which we still want to set in our sweep_config.
#
# In that case, we just set the value directly:

sweep_config["parameters"] = parameters_dict

parameters_dict.update({"epochs": {"value": 1}})

# For a grid search, that's all you ever need.
#
# For a random search, all the values of a parameter are equally likely to be chosen on a given run.
#
# If that just won't do, you can instead specify a named distribution, plus its parameters, like the mean mu and standard deviation sigma of a normal distribution.

parameters_dict.update(
    {
        "learning_rate": {
            # a flat distribution between 0 and 0.1
            "distribution": "uniform",
            "min": 0,
            "max": 0.1,
        },
        "batch_size": {
            # integers between 32 and 256
            # with evenly-distributed logarithms
            "distribution": "q_log_uniform_values",
            "q": 8,
            "min": 32,
            "max": 256,
        },
    }
)

# When we're finished, sweep_config is a nested dictionary that specifies exactly which parameters we're
# interested in trying and what method we're going to use to try them.

import pprint

pprint.pprint(sweep_config)

# Once you've defined the search strategy, it's time to set up something to implement it.
#
# The clockwork taskmaster in charge of our Sweep is known as the Sweep Controller. As each run completes,
# it will issue a new set of instructions describing a new run to execute. These instructions are picked up
# by agents who actually perform the runs.
#
# In a typical Sweep, the Controller lives on our machine, while the agents who complete runs live on your machine(s),
# like in the diagram below. This division of labor makes it super easy to scale up Sweeps by just adding more machines to run agents!

# We can wind up a Sweep Controller by calling wandb.sweep with the appropriate sweep_config and project name.
#
# This function returns a sweep_id that we will later user to assign agents to this Controller.

sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")

# Before we can actually execute the sweep, we need to define the training procedure that uses those values.
#
# In the functions below, we define a simple fully-connected neural network in PyTorch,
# and add the following wandb tools to log model metrics, visualize performance and output and track our experiments:

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataset(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # download MNIST training dataset
    dataset = datasets.MNIST(".", train=True, download=True, transform=transform)
    sub_dataset = torch.utils.data.Subset(dataset, indices=range(0, len(dataset), 5))
    loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size)

    return loader


def build_network(fc_layer_size, dropout):
    network = nn.Sequential(  # fully-connected, single hidden layer
        nn.Flatten(),
        nn.Linear(784, fc_layer_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(fc_layer_size, 10),
        nn.LogSoftmax(dim=1),
    )

    return network.to(device)


def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    return optimizer


def train_epoch(network, loader, optimizer):
    cumu_loss = 0
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # ➡ Forward pass
        loss = F.nll_loss(network(data), target)
        cumu_loss += loss.item()

        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})


# Now, we're ready to start sweeping! 🧹🧹🧹
#
# Sweep Controllers, like the one we made by running wandb.sweep, sit waiting for someone to ask them for a config to try out.
#
# That someone is an agent, and they are created with wandb.agent. To get going, the agent just needs to know
#
# which Sweep it's a part of (sweep_id)
# which function it's supposed to run (here, train)
# (optionally) how many configs to ask the Controller for (count)

# FYI, you can start multiple agents with the same sweep_id on different compute resources,
# and the Controller will ensure that they work together according to the strategy laid out in the sweep_config.
# This makes it trivially easy to scale your Sweeps across as many nodes as you can get ahold of!

# The cell below will launch an agent that runs train 5 times,
# usingly the randomly-generated hyperparameter values returned by the Sweep Controller.
# Execution takes under 5 minutes.

wandb.agent(sweep_id, train, count=5)
