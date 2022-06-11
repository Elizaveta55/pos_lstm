from models import *
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data
from torchtext.legacy import datasets
import numpy as np
import time
import random
import argparse

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, dest='config', default='config.yaml')
args = parser.parse_args()
config = Config(args.config)


text = data.Field(lower = True)
tags = data.Field(unk_token = None)
min_frequency = 2

fields = (("text", text), ("udtags", tags))
train_data, valid_data, test_data = datasets.UDPOS.splits(fields)

text.build_vocab(train_data,
                 min_freq = min_frequency,
                 vectors = "glove.6B.100d",
                 # vectors = "glove.twitter.27B.100d",
                 # vectors = "charngram.100d",
                 unk_init = torch.Tensor.normal_)
tags.build_vocab(train_data)
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = config.batch_size,
    device = device)
input_dim = len(text.vocab)
output_dim = len(tags.vocab)
padding_index = 2
# padding_index = text.vocab.stoi[text.pad_token]


model = LSTM(input_dim,
             config.embed_dim,
             config.hidden_dim,
             output_dim,
             config.layer,
             config.is_bidirectional,
             config.dropout,
             padding_index)

pretrained_embeddings = text.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
model.embedding.weight.data[padding_index] = torch.zeros(config.embed_dim)
optimizer = optim.Adam(model.parameters())

# tags_padding_index = tags.vocab.stoi[tags.pad_token]
tags_padding_index = 2

criterion = nn.CrossEntropyLoss(ignore_index = tags_padding_index)

model = model.to(device)
criterion = criterion.to(device)

best_valid_loss = float('inf')
for epoch in range(config.epochs):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, tags_padding_index)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, tags_padding_index)

    end_time = time.time()

    epoch_mins, epoch_secs = calculation_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model.pt')

    print(f'Epoch: {epoch + 1} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

model.load_state_dict(torch.load('model.pt'))
test_loss, test_acc = evaluate(model, test_iterator, criterion, tags_padding_index)

print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')
