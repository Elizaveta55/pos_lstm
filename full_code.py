import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data
from torchtext.legacy import datasets
import numpy as np

## We set a seed to reproduce the same results in the future
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##There are parameters which were identified as quite appropriate
batch_size = 32
layer = 2
is_bidirectional = True
epochs = 10

##Class for LSTM network with gate for bidirectional/simple
class LSTM(nn.Module):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout,
                 pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        input_embed = self.dropout(self.embedding(text))
        outputs, (_, _) = self.lstm(input_embed)
        predictions = self.fc(self.dropout(outputs))
        return predictions


def accuracy(preds, y, tag_pad_idx):
    max_preds = preds.argmax(dim = 1, keepdim = True)
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / y[non_pad_elements].shape[0]


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

def calculation_time(start_time, end_time):
    total_time = end_time - start_time
    mins = int(total_time / 60)
    return mins, int(total_time - (mins * 60))


##torchtext's application (by "fields") for UD tags is quite convinient to use
text = data.Field(lower = True)
tags = data.Field(unk_token = None)
min_frequency = 2

fields = (("text", text), ("udtags", tags))
train_data, valid_data, test_data = datasets.UDPOS.splits(fields)

## dimension for embedding.
## torchtext's field during building the vocabular requires to use some representation vector
## it already has some in-build pretrained vectors, therefore it is useful to choose one of these
## but "embed_dim" and dimention of pretrained vector should be the same
embed_dim = 100
hidden_dim = 128
dropout = 0.25

## this configuration for data limitation is complete: it uses 100% of data
## the application of 10%, ..., 90% is implemented with manual changing of "en-ud-tag.v2.dev.txt"
## and "en-ud-tag.v2.test.txt" files in .data/udpos/en-ud-v2/
print("Data limitation is ", 100, "%")
text.build_vocab(train_data,
                 min_freq = min_frequency,
                 vectors = "glove.6B.100d",
                 # vectors = "glove.twitter.27B.100d",    ## these vectors could be used
                                                          ## instead of "glove.6B.100d"
                 # vectors = "charngram.100d",
                 unk_init = torch.Tensor.normal_)
tags.build_vocab(train_data)
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = batch_size,
    device = device)
input_dim = len(text.vocab)
output_dim = len(tags.vocab)
padding_index = 2
# padding_index = text.vocab.stoi[text.pad_token]

model = LSTM(input_dim,
             embed_dim,
             hidden_dim,
             output_dim,
             layer,
             is_bidirectional,
             dropout,
             padding_index)

pretrained_embeddings = text.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
model.embedding.weight.data[padding_index] = torch.zeros(embed_dim)
# tags_padding_index = tags.vocab.stoi[tags.pad_token]
tags_padding_index = 2

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss(ignore_index = tags_padding_index)

model = model.to(device)
criterion = criterion.to(device)
best_valid_loss = float('inf')

for epoch in range(epochs):

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
