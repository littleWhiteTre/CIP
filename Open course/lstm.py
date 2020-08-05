import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import random
import numpy as np
import copy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 50
HIDDEN = 100
EMBED = 100
BATCH_SIZE = 32
TEXT_LENS = 50
'''
return data:
[((NR, NR, NN, ...), ('戴相龙', '说', '中国'))]
'''


def read_data(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        line = f.readline()
        sentence = []
        while line:
            if line == '\n':
                line = f.readline()
                data.append(sentence)
                sentence = []
            else:
                line = line.replace('\n', '')
                vals = line.split()
                word = vals[1]
                part = vals[3]
                sentence.append((word, part))
                line = f.readline()
    data_set = []
    for i in data:
        words, tags = zip(*i)
        data_set.append([words, tags])
    return data_set


def statistic_data(data):
    words_set = set()
    tags_set = set()
    for item in data:
        words, tags = item
        for w, t in zip(words, tags):
            words_set.add(w)
            tags_set.add(t)
    words = ['<pad>']
    words.extend(list(sorted(words_set)))
    words.append('<unk>')
    tags = list(sorted(tags_set))
    words2idx = {word: idx for idx, word in enumerate(words)}
    tags2idx = {tag: idx for idx, tag in enumerate(tags)}
    return words, words2idx, tags, tags2idx


class Config:
    def __init__(self):
        self.cwd = os.getcwd()
        work_path = self.cwd.split('\\')
        if work_path[-1] != 'CIP':
            self.train_file = '../data/train.conll'
            self.dev_file = '../data/dev.conll'
            self.data_path = '../data/'
        else:
            self.train_file = './data/train.conll'
            self.dev_file = './data/dev.conll'
            self.data_path = './data/'


class My_Dataset(Dataset):
    def __init__(self, data, words, words2idx, tags, tags2idx, text_lens=50):
        self.data = data
        self.words = words
        self.words2idx = words2idx
        self.tags = tags
        self.tags2idx = tags2idx
        self.text_lens = text_lens

    def __getitem__(self, index):
        item = self.data[index]
        tags = list(item[1])
        words = list(item[0])
        tags = [self.tags2idx[i] for i in tags]
        n = len(words)
        for i in range(len(words)):
            if words[i] in self.words:
                words[i] = self.words2idx[words[i]]
            else:
                words[i] = self.words2idx['<unk>']
        if n < self.text_lens:
            tags.extend([0 for i in range(self.text_lens - n)])
            words.extend([0 for i in range(self.text_lens - n)])
        else:
            tags = tags[0:self.text_lens]
            words = tags[0:self.text_lens]
            n = self.text_lens
        return torch.tensor(tags), torch.tensor(words), torch.tensor(n)

    def __len__(self):
        return len(self.data)


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab, tags, embed_dim, hidden_size):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = len(vocab)
        self.tags_size = len(tags)
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.vocab_size,
                                      embed_dim,
                                      padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_size // 2,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_size, self.tags_size)

    def forward(self, inp, n):
        # inp: [batch, text_len]
        # n: [batch, 1]
        embed = self.embedding(inp)  # [batch, text_len, embed_dim]
        packed_inp = nn.utils.rnn.pack_padded_sequence(embed,
                                                       n,
                                                       batch_first=True,
                                                       enforce_sorted=False)
        lstm_o, (h, c) = self.lstm(packed_inp, None)
        lstm_o, _ = nn.utils.rnn.pad_packed_sequence(lstm_o, batch_first=True)
        # lstm_o: [batch, seq_lens, hidden_size]
        return self.fc(lstm_o)


def train(model, dl, epochs=5, lr=1e-3, test_dl=None, best_score=0):
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_score = 0
    for epoch in range(1, epochs + 1):
        running_loss = 0
        dl_ = copy.deepcopy(dl)
        for step, batch in enumerate(dl_):
            labels, sents, n = batch  # labels:[batch, text_lens]
            batch_size = labels.size()[0]
            pred = model(sents, n)  # [batch, text_lens, tags_size]
            optim.zero_grad()
            for i in torch.arange(batch_size):
                i_label = labels[i, :n[i]]
                i_pred = pred[i, :n[i], :]
                loss = criterion(i_pred, i_label)
                loss.backward(retain_graph=True)
            optim.step()
            running_loss += loss.item()
        print('%d running loss is %f' % (epoch, running_loss))
        if test_dl:
            acc = evalution(model, test_dl)
            print('accurancy on test dataset is %f' % (acc))
            if acc > best_score:
                best_score = acc
                torch.save(model.state_dict(), 'bilstm.model')
                print('one better model has saved.')


def evalution(model, dl):
    model.eval()
    dl_ = copy.deepcopy(dl)
    acc = 0
    correct = 0
    total = 0
    for step, batch in enumerate(dl_):
        labels, sents, n = batch  # labels:[batch, text_lens]
        batch_size = labels.size()[0]
        pred = model(sents, n)  # [batch, text_lens, tags_size]
        for i in torch.arange(batch_size):
            i_label = labels[i, :n[i]]
            i_pred = pred[i, :n[i], :]  # [n, tags_size]
            i_pred = torch.argmax(i_pred, dim=1)
            correct += sum(
                int(i_pred[j] == i_label[j]) for j in range(i_label.size()[0]))
            total += i_label.size()[0]
        batch_acc = correct / total
        acc += batch_acc
    model.train()
    return acc / step


if __name__ == "__main__":
    # ----------------------------data part---------------------------
    cfg = Config()
    train_data = read_data(cfg.train_file)
    test_data = read_data(cfg.dev_file)
    words, words2idx, tags, tags2idx = statistic_data(train_data)
    ds = My_Dataset(train_data,
                    words,
                    words2idx,
                    tags,
                    tags2idx,
                    text_lens=TEXT_LENS)
    dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True)
    test_ds = My_Dataset(test_data,
                         words,
                         words2idx,
                         tags,
                         tags2idx,
                         text_lens=TEXT_LENS)
    test_dl = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=True)
    # batch = iter(dl).next()
    # ---------------------------model part----------------------------
    model = BiLSTM_CRF(words, tags, EMBED, HIDDEN)
    if os.path.isfile('bilistm.model'):
        model.load_state_dict(torch.load('bilstm.model'))
        acc = evalution(model, test_dl)
        train(model, dl, epochs=EPOCHS, test_dl=test_dl, best_score=acc)
    else:
        train(model, dl, epochs=EPOCHS, test_dl=test_dl)
