import os
import sys

from argparse import ArgumentParser
import time
import math
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader


def parse_args():
    parser = ArgumentParser(description='CIFAR')
    parser.add_argument('--dataset', choices=['cifar10', 'pathfinder'], required=True)
    parser.add_argument('--data_path', help='path for data file.', required=True)
    return parser.parse_args()


def dump_dataset(dataset, img_size, data_path, split, num_classes):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    try:
        os.makedirs(os.path.join(data_path, 'input'))
        os.makedirs(os.path.join(data_path, 'label'))
    except FileExistsError:
        print('File exists')

    src_path = os.path.join(data_path, 'input', split + ".src")
    label_path = os.path.join(data_path, 'label', split + ".label")

    targets = [0] * num_classes
    total = 0
    with open(src_path, 'w') as sf, open(label_path, 'w') as lf:
        for img, y in dataloader:
            img = img.view(img_size).mul(255).int().numpy().tolist()
            y = y.item()
            targets[y] += 1
            total += 1
            pixels = [str(p) for p in img]
            sf.write(' '.join(pixels) + '\n')
            lf.write(str(y) + '\n')
            sf.flush()
            lf.flush()

    print(total)
    print(targets)


def cifar10(data_path):
    dataset = datasets.CIFAR10
    num_classes = 10
    trainset = dataset(data_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Grayscale(),
                           transforms.ToTensor(),
                       ]))

    train_size = int(len(trainset) * 0.9)
    trainset, valset = torch.utils.data.random_split(
        trainset,
        (train_size, len(trainset) - train_size),
        generator=torch.Generator().manual_seed(42),
    )

    testset = dataset(data_path, train=False, download=False,
                     transform=transforms.Compose([
                         transforms.Grayscale(),
                         transforms.ToTensor(),
                     ]))

    dump_dataset(trainset, 1024, data_path, 'train', num_classes)
    dump_dataset(valset, 1024, data_path, 'valid', num_classes)
    dump_dataset(testset, 1024, data_path, 'test', num_classes)


def main(args):
    dataset = args.dataset
    data_path = args.data_path
    if dataset == 'cifar10':
        cifar10(data_path)
    else:
        pass


if __name__ == "__main__":
    args = parse_args()
    main(args)