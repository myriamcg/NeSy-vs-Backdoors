import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip
from torchvision.datasets import CIFAR10

import core

# Transforms
transform_train = Compose([ToTensor(), RandomHorizontalFlip()])
transform_test = Compose([ToTensor()])

# Load CIFAR-10 datasets
trainset = CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
testset = CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)

# Visual inspection of sample data
index = 44
x, y = trainset[index]
print(y)
for a in x[0]:
    for b in a:
        print("%-4.2f" % float(b), end=' ')
    print()

# Define backdoor pattern and weight
pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
pattern[0, -3:, -3:] = 255
weight = torch.zeros((1, 32, 32), dtype=torch.float32)
weight[0, -3:, -3:] = 1.0

# Create BadNets attack instance
badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target=0,
    poisoned_rate=0.1,
    pattern=pattern,
    weight=weight,
    schedule=None,
    seed=666
)

# Generate poisoned datasets
poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()

# Print poisoned sample
x, y = poisoned_train_dataset[index]
print(y)
for a in x[0]:
    for b in a:
        print("%-4.2f" % float(b), end=' ')
    print()

x, y = poisoned_test_dataset[index]
print(y)
for a in x[0]:
    for b in a:
        print("%-4.2f" % float(b), end=' ')
    print()


# ===========================
# Main training block (Windows-safe)
# ===========================
if __name__ == '__main__':
    # Train benign model
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': '0',
        'GPU_num': 1,

        'benign_training': True,
        'batch_size': 128,
        'num_workers': 16,  # Set to 0 if still failing due to multiprocessing

        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [150, 180],

        'epochs': 2,

        'log_iteration_interval': 100,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,

        'save_dir': 'experiments',
        'experiment_name': 'train_benign_DatasetFolder-CIFAR10'
    }

    badnets.train(schedule)

    # Train attacked model
    schedule['benign_training'] = False
    schedule['experiment_name'] = 'train_poisoned_DatasetFolder-CIFAR10'

    badnets.train(schedule)
