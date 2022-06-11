import imageio
import matplotlib
import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import yaml
from PIL import Image
from easydict import EasyDict as edict

def Config(filename):
    with open(filename, 'r') as f:
        parser = edict(yaml.load(f, Loader=yaml.FullLoader))
    for x in parser:
        print('{}: {}'.format(x, parser[x]))
    return parser


def accuracy(preds, y, tag_pad_idx):
    max_preds = preds.argmax(dim = 1, keepdim = True)
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / y[non_pad_elements].shape[0]


def accuracy_with_print(preds, y, tag_pad_idx):
    max_preds = preds.argmax(dim = 1, keepdim = True)
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / y[non_pad_elements].shape[0]


def calculation_time(start_time, end_time):
    total_time = end_time - start_time
    mins = int(total_time / 60)
    return mins, int(total_time - (mins * 60))


def train(model, iterator, optimizer, criterion, tag_pad_idx):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        text = batch.text
        tags = batch.udtags

        optimizer.zero_grad()
        predictions = model(text)
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)

        loss = criterion(predictions, tags)

        acc = accuracy(predictions, tags, tag_pad_idx)

        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, tag_pad_idx):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text = batch.text
            tags = batch.udtags

            predictions = model(text)

            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)

            loss = criterion(predictions, tags)

            acc = accuracy(predictions, tags, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def evaluate_with_print(model, iterator, criterion, tag_pad_idx):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text = batch.text
            tags = batch.udtags

            predictions = model(text)

            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)

            loss = criterion(predictions, tags)

            acc = accuracy_with_print(predictions, tags, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)