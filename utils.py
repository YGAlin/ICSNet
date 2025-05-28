import os
import random

import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def create_dataloader(args):
    train_source = os.path.join(args.data_path, args.data_path_source, args.src)
    train_target = os.path.join(args.data_path, args.data_path_target, args.tar)
    test_target = os.path.join(args.data_path, args.data_path_test, args.test)
    if not os.path.isdir(train_source):
        raise ValueError('Null path of source train data!!!')

    train_source_dataset = datasets.ImageFolder(
        train_source,
        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]))
    train_source_loader = DataLoader(
        train_source_dataset, batch_size=args.batch_size_s, shuffle=True,drop_last=False,
        pin_memory=True, sampler=None
    )

    train_target_dataset = datasets.ImageFolder(
        train_target,
        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]))
    train_target_loader = DataLoader(
        train_target_dataset, batch_size=args.batch_size_t, shuffle=True,drop_last=False,
        pin_memory=True, sampler=None
    )

    test_target_dataset = datasets.ImageFolder(
        test_target,
        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    )
    test_target_loader = DataLoader(
        test_target_dataset, batch_size=args.batch_size_t, shuffle=True,drop_last=False,
        pin_memory=True, sampler=None
    )

    return train_source_loader, train_target_loader, test_target_loader

class IterDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iter = iter(self.dataloader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            data = next(self.iter)
        return data

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr / pow((1 + 10 * epoch / args.epochs), 0.75)
    lr_pretrain = args.lr  / pow((1 + 10 * epoch / args.epochs), 0.75)
    # 0.001 / pow((1 + 10 * epoch / epoch_total), 0.75)
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'pre-trained':
            param_group['lr'] = args.lr*0.5
        else:
            param_group['lr'] = args.lr

