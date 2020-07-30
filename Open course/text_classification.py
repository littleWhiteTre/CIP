import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import os
import random
import copy

seed = 0
torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
TEXT_LENS = 25
EMBED_DIM = 100
EPOCHS = 50
LR = 1e-3


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
        return torch.tensor(tag, dtype=torch.long), torch.tensor(words, dtype=torch.long)

    def __len__(self):
        return len(self.data)


class CNN_Network(nn.Module):
    def __init__(self,
                 vocab,
                 tags,
                 text_lens=25,
                 embed_dim=100,
                 kernel_nums=100,
                 kernel_size=2,
                 batch_size=32):
        super(CNN_Network, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.tags = tags
        self.embedding = nn.Embedding(self.vocab_size, embed_dim, padding_idx=self.vocab['<pad>'])
        self.conv1d = nn.Conv1d(in_channels=embed_dim,
                                out_channels=kernel_nums,
                                kernel_size=kernel_size)
        self.pooling = nn.MaxPool1d(kernel_size=(text_lens - kernel_size + 1))
        self.fc = nn.Linear(kernel_nums, len(self.tags))

    def forward(self, inp, softmax=False):
        # inp: [batch_size, text_lens]
        inp = self.embedding(inp)  # [batch_size, text_lens, embedding_dim]
        inp = inp.permute(0, 2, 1)  # [batch_size, embedding_dim, text_lens]
        inp = self.conv1d(inp)  # [batch_size, kernel_nums, steps: (text_lens-kernel_size+1)]
        inp = self.pooling(inp)  # [batch_size, out_chanels, 1]
        inp = torch.squeeze(inp, dim=2)
        # inp = inp.permute(0, 2, 1) # [batch_size, 1, out_chanels]
        if softmax:
            return F.softmax(self.fc(inp), dim=1)
        else:
            return self.fc(inp)  # [batch_size, tags_size]


def train(model, dl, lr, acc=0, test_dl=None):
    best_score = acc
    model.to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS+1):
        running_loss = 0
        dl_ = copy.deepcopy(dl)
        for step, batch in enumerate(dl_):
            labels = batch[0]
            sents = batch[1]
            inp = sents # [batch_size, text_lens]
            labels.to(device)
            inp.to(device)
            optim.zero_grad()
            pred = model.forward(inp)
            loss = loss_func(pred, labels)
            loss.backward()
            optim.step()
            running_loss = running_loss + loss.item()
            if step % 50 == 0:
                print('%d epoch step:%d, loss is %f' %
                      (epoch, step, loss.item()))
        print('%d epoch running loss is %f' % (epoch, running_loss))
        if test_dl:
            acc = evalute(model, test_dl)
            print('%d epoch accurancy on test data:%f' % (epoch, acc))
            # acc2 = evalute(model, dl)
            # print('%d epoch accurancy on train data:%f' % (epoch, acc2))
            if acc > best_score:
                torch.save(model.state_dict(), 'cnn.model')
                best_score = acc
                print('one better model has saved.')


def evalute(model, dl):
    model.eval()
    correct = 0
    total = 0
    acc = 0
    dl_ = copy.deepcopy(dl)
    for step, batch in enumerate(dl_):
        labels = batch[0]
        sents = batch[1]
        inp = sents # [batch_size, text_lens]
        labels.to(device)
        inp.to(device)
        labels.to(device)
        inp.to(device)
        pred = model.forward(inp, softmax=True) #[batch,tags_lens]
        pred = torch.argmax(pred, dim=1)#[batch, 1]
        correct += sum(int(pred[i] == labels[i]) for i in range(len(pred))) # total+=32 correct=30
        total += labels.size()[0]
    acc = correct / total
    print('%d / %d, accurancy:%f.' % (correct, total, acc))
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
    i = iter(dl)
    batch = next(i)
    model = CNN_Network(words2idx,
                        tags,
                        text_lens=TEXT_LENS,
                        embed_dim=EMBED_DIM,
                        batch_size=BATCH_SIZE)
    model.load_state_dict(torch.load('cnn.model'))
    acc = evalute(model, test_dl)
    train(model, dl, lr=LR, acc=acc ,test_dl=test_dl)