import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import os
import random

seed = 0
torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

device = 'gpu' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
TEXT_LENS = 10
EMBED_DIM = 50
EPOCHS = 50
LR = 1e-4


def read_data(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            item = line.replace('\n', '').split('\t')
            tag = item[0]
            words = item[1].split(' ')
            data.append((tag, words))
            line = f.readline()
    return data


def statistic_data(data):
    tags = set()
    words = set()
    for item in data:
        tags.add(item[0])
        for w in item[1]:
            words.add(w)
    tags = list(sorted(tags))
    words = list(sorted(words))
    words.extend(['<pad>', '<unk>'])
    tags2idx = {tag: idx for idx, tag in enumerate(tags)}
    words2idx = {word: idx for idx, word in enumerate(words)}
    return tags, tags2idx, words, words2idx


class Config:
    def __init__(self):
        self.cwd = os.getcwd()
        work_path = self.cwd.split('\\')
        if work_path[-1] != 'CIP':
            self.train_file = '../data/HIT_big/train'
            self.dev_file = '../data/HIT_big/dev'
            self.data_path = '../data/HIT_big/'
        else:
            self.train_file = './data/HIT_big/train'
            self.dev_file = './data/HIT_big/dev'
            self.data_path = './data/HIT_big/'


class My_DataSet(Dataset):
    def __init__(self, data, tags2idx, words2idx, max_len=10):
        self.data = data
        self.tags2idx = tags2idx
        self.words2idx = words2idx
        self.max_len = max_len

    def __getitem__(self, index):
        item = self.data[index]
        tag = item[0]
        tag = self.tags2idx[tag]
        words = item[1]
        if len(words) < self.max_len:
            words.extend(['<pad>' for i in range(self.max_len - len(words))])
        else:
            words = words[0:self.max_len]
        for i in range(len(words)):
            if words[i] not in self.words2idx:
                words[i] = self.words2idx['<unk>']
            else:
                words[i] = self.words2idx[words[i]]
        return tag, words

    def __len__(self):
        return len(self.data)


class CNN_Network(nn.Module):
    def __init__(self,
                 vocab,
                 tags,
                 text_lens=10,
                 embed_dim=500,
                 kernel_nums=50,
                 kernel_size=2,
                 batch_size=32):
        super(CNN_Network, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.tags = tags
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.conv1d = nn.Conv1d(in_channels=embed_dim,
                                out_channels=kernel_nums,
                                kernel_size=kernel_size)
        self.pooling = nn.MaxPool1d(kernel_size=(text_lens - kernel_size + 1))
        self.fc = nn.Linear(kernel_nums, len(self.tags))

    def forward(self, inp, softmax=False):
        # inp: [batch_size, text_lens]
        inp = self.embedding(inp)  # [batch_size, text_lens, embedding_dim]
        inp = inp.permute(0, 2, 1)  # [batch_size, embedding_dim, text_lens]
        inp = self.conv1d(
            inp)  # [batch_size, out_channels, steps(text_lens-kernel_size+1)]
        inp = self.pooling(inp)  # [batch_size, out_chanels, 1]
        inp = torch.squeeze(inp, dim=2)
        # inp = inp.permute(0, 2, 1) #[batch_size, 1, out_chanels]
        if softmax:
            return F.softmax(self.fc(inp), dim=1)
        else:
            return self.fc(inp)  # [batch_size, tags_size]


def train(model, dl, lr, test_dl=None):
    best_score = 0
    model.to(device)
    model.train()
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        running_loss = 0
        for step, batch in enumerate(dl):
            labels = batch[0].long()
            sents = batch[1]
            batch_size = len(sents[0])
            inp = torch.zeros(batch_size, TEXT_LENS)
            for i in range(len(sents)):
                inp[:, i] = sents[i]
            labels.to(device)
            pred = model.forward(inp.long())
            loss = loss_func(pred, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss = running_loss + loss.item()
            if step % 50 == 0:
                print('%d epoch step:%d, loss is %f' %
                      (epoch, step, loss.item()))
        print('%d epoch running loss is %f' % (epoch, running_loss))
        if test_dl:
            print('evaluting model on test data...')
            acc = evalute(model, test_dl)
            print('%d epoch accurancy on test data:%f' % (epoch, acc))
            if acc > best_score:
                torch.save(model.state_dict(), 'cnn.model')
                best_score = acc
                print('one better model has saved.')


def evalute(model, dl):
    model.eval()
    acc = 0
    for step, batch in enumerate(dl):
        labels = batch[0].long()
        sents = batch[1]
        batch_size = len(sents[0])
        inp = torch.zeros(batch_size, TEXT_LENS)
        for i in range(len(sents)):
            inp[:, i] = sents[i]
        labels.to(device)
        pred = model.forward(inp.long(), softmax=True)
        pred = torch.argmax(pred, dim=1)
        acc = acc + (sum(int(pred[i] == labels[i]) for i in range(len(pred))))/batch_size
    acc = acc / step
    model.train()
    return acc

if __name__ == "__main__":
    cfg = Config()
    data = read_data(cfg.train_file)
    dev_data = read_data(cfg.dev_file)
    tags, tags2idx, words, words2idx = statistic_data(data)
    ds = My_DataSet(data,
                    tags2idx=tags2idx,
                    words2idx=words2idx,
                    max_len=TEXT_LENS)
    test_ds = My_DataSet(dev_data,
                    tags2idx=tags2idx,
                    words2idx=words2idx,
                    max_len=TEXT_LENS)
    dl = DataLoader(dataset=ds, shuffle=True, batch_size=BATCH_SIZE)
    test_dl = DataLoader(dataset=test_ds, shuffle=True, batch_size=BATCH_SIZE)
    model = CNN_Network(words,
                        tags,
                        text_lens=TEXT_LENS,
                        embed_dim=EMBED_DIM,
                        batch_size=BATCH_SIZE)
    train(model, dl, lr=LR, test_dl=test_dl)