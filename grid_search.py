import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data
from torchtext.legacy import datasets
import numpy as np
import time
import random

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_sizes = [32, 64, 128]
layers = [1, 2, 4]
is_bidirectionals = [False, True]
n_epochs = [2, 5, 10, 20]
# data_limitations = [1]
data_limitations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


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
        embedded = self.dropout(self.embedding(text))
        outputs, (_, _) = self.lstm(embedded)
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


TEXT = data.Field(lower = True)
UD_TAGS = data.Field(unk_token = None)
MIN_FREQ = 2

fields = (("text", TEXT), ("udtags", UD_TAGS))
train_data, valid_data, test_data = datasets.UDPOS.splits(fields)

embed_dim = 100
hidden_dim = 128
dropout = 0.25

for data_limitation in data_limitations:
    print("Data limitation is ", data_limitation)
    TEXT.build_vocab(train_data,
                     min_freq = MIN_FREQ,
                     vectors = "glove.6B.100d",
                     # vectors = "glove.twitter.27B.100d",
                     # vectors = "charngram.100d",
                     unk_init = torch.Tensor.normal_,
                     max_size=round(len(train_data)*data_limitation))
    UD_TAGS.build_vocab(train_data)
    print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    print(f"Unique tokens in UD_TAG vocabulary: {len(UD_TAGS.vocab)}")
    for batch_size in batch_sizes:
        print("Batch size is ", batch_size)
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size = batch_size,
            device = device)
        print("train_iterator is ", len(train_iterator))
        input_dim = len(TEXT.vocab)
        output_dim = len(UD_TAGS.vocab)
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

        for layer in layers:
            print("Layer is ", layer)
            for is_bidirectional in is_bidirectionals:
                print("Bidirectional - ", is_bidirectional)
                model = LSTM(input_dim,
                             embed_dim,
                             hidden_dim,
                             output_dim,
                             layer,
                             is_bidirectional,
                             dropout,
                             PAD_IDX)


                pretrained_embeddings = TEXT.vocab.vectors
                model.embedding.weight.data.copy_(pretrained_embeddings)
                model.embedding.weight.data[PAD_IDX] = torch.zeros(embed_dim)
                optimizer = optim.Adam(model.parameters())

                TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]
                # tags_padding_index = 2

                criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)

                model = model.to(device)
                criterion = criterion.to(device)

                best_valid_loss = float('inf')
                for epochs in n_epochs:
                    print("Number of epochs is ", epochs)
                    for epoch in range(epochs):

                        start_time = time.time()

                        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, TAG_PAD_IDX)
                        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, TAG_PAD_IDX)

                        end_time = time.time()

                        epoch_mins, epoch_secs = calculation_time(start_time, end_time)

                        if valid_loss < best_valid_loss:
                            best_valid_loss = valid_loss
                            torch.save(model.state_dict(), 'tut1-model.pt')

                        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
                        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

                    model.load_state_dict(torch.load('tut1-model.pt'))
                    test_loss, test_acc = evaluate(model, test_iterator, criterion, TAG_PAD_IDX)

                    print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')
