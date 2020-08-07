import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import random
import numpy as np
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 50
HIDDEN = 100
EMBED = 50
LR = 1e-3
MODEL_NAME = 'data/bilstm_crf2.model'


def read_data(file):
    '''
    return data:
    [[('戴相龙', '说', '中国'), (NR, NR, NN, ...)]]
    '''
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
    words = list(sorted(words_set))
    words.append('<unk>')
    tags = list(sorted(tags_set))
    tags.extend(['<bos>', '<eos>'])
    words2idx = {word: idx for idx, word in enumerate(words)}
    tags2idx = {tag: idx for idx, tag in enumerate(tags)}
    return words, words2idx, tags, tags2idx


def tokens2seqs(tokens, idx_dict, tag=False):
    if tag:
        return torch.tensor([idx_dict[i] for i in tokens], dtype=torch.long)
    else:
        idxs = []
        for i in range(len(tokens)):
            if tokens[i] in idx_dict:
                idxs.append(idx_dict[tokens[i]])
            else:
                idxs.append(idx_dict['<unk>'])
        return torch.tensor(idxs, dtype=torch.long)


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


class BiLSTM_CRF2(nn.Module):
    def __init__(self, vocab, tags2idx, embed_dim, hidden_size):
        super(BiLSTM_CRF2, self).__init__()
        self.vocab = vocab
        self.tags2idx = tags2idx
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(len(self.vocab), embed_dim)
        self.lstm = nn.LSTM(embed_dim,
                            hidden_size=hidden_size // 2,
                            batch_first=True,
                            num_layers=1,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_size, len(self.tags2idx) - 2)
        self.transition = nn.Parameter(
            torch.zeros(len(tags2idx), len(tags2idx)))
        self.transition.data[self.tags2idx['<eos>'], :] = -10000
        self.transition.data[:, self.tags2idx['<bos>']] = -10000

    def _get_lstm_feats(self, sents):
        '''
        Args:
            sents: [n]
        Return:
            sents_featurs:[n, tags_size-2]
        '''
        inp = self.embedding(sents).view(1, len(sents),
                                         -1)  # [1, n, embed_dim]
        lstm_o, _ = self.lstm(inp, None)  # lstm_o : [1, n, hidden_size]
        lstm_o = lstm_o.view(len(sents), self.hidden_size)  # [n, hidden_size]
        return self.fc(lstm_o)

    def _score_sents(self, emission, tags):
        '''
        Args:
            emission: [n, tags_size-2]
            tags: [n]
        Return:
            score: [1]
        '''
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tags2idx['<bos>']]),
                          tags])  #[<bos>+n]
        for i, feat in enumerate(emission):
            score += feat[tags[i + 1]] + self.transition[tags[i], tags[i + 1]]
        return score + self.transition[tags[-1], self.tags2idx['<eos>']]

    def _forward_alg(self, emission):
        '''
        Args:
            emission: [n, tags_size-2]
        Return:
            total score: [1]
        '''
        obs = emission[0] + self.transition[self.tags2idx['<bos>'], :
                                            -2]  # [tags_size - 2]
        pre = obs
        pre = torch.unsqueeze(pre, dim=1)
        for i in range(1, emission.size()[0]):
            obs = torch.unsqueeze(emission[i], dim=0)
            score = pre + obs + self.transition[:-2, :
                                                -2]  # [tags_size-2, tags_size-2]
            if i == emission.size()[0] - 1:
                score = score + torch.unsqueeze(
                    self.transition[:-2, self.tags2idx['<eos>']], dim=0)
            pre = torch.logsumexp(score, dim=0, keepdim=True).T
        total_s = torch.logsumexp(pre, dim=0)
        return total_s

    def _viterbi_decode(self, emission):
        '''
        Args:
            emission: [n, tags_size-2]
        Return:
            pred seqs: [n]
        '''
        alpha0 = []
        alpha1 = []  # lens of alpha1 should be 'n-1'.
        pre = emission[0] + self.transition[self.tags2idx['<bos>'], :
                                            -2]  # [tags_size-2]
        alpha0.append(pre)
        pre = torch.unsqueeze(pre, dim=1)  # [tags_size-2, 1]
        for i in range(1, emission.size()[0]):
            obs = torch.unsqueeze(emission[i], dim=0)
            score = pre + obs + self.transition[:-2, :-2]
            if i == emission.size()[0] - 1:
                score = score + torch.unsqueeze(
                    self.transition[:-2, self.tags2idx['<eos>']], dim=0)
            max_v, max_idx = torch.max(score, dim=0)
            alpha0.append(max_v)
            alpha1.append(max_idx)
            pre = torch.unsqueeze(max_v, dim=1)
        max_end = torch.argmax(alpha0[-1])
        best_seq = [max_end]
        for i in range(len(alpha1) - 1, -1, -1):
            pre_tag = alpha1[i][max_end]
            best_seq.append(pre_tag)
            max_end = pre_tag
        best_seq.reverse()
        return torch.tensor(best_seq)

    def neg_log_likelihood(self, sents, tags):
        emission = self._get_lstm_feats(sents)
        forward_score = self._forward_alg(emission)
        real_score = self._score_sents(emission, tags)
        return forward_score - real_score

    def forward(self, sents):
        emission = self._get_lstm_feats(sents)
        return self._viterbi_decode(emission)


def train(model, train_data, tags2idx, words2idx, test_data=None, best_acc=0):
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        for sents, tags in tqdm(train_data):
            tags = tokens2seqs(tags, tags2idx, tag=True)
            sents = tokens2seqs(sents, words2idx)
            optim.zero_grad()
            loss = model.neg_log_likelihood(sents, tags)
            loss.backward()
            optim.step()
            running_loss += loss.item()
        print('%d epoch running loss is %f' % (epoch, running_loss))
        if test_data:
            best_acc = best_acc
            acc = evalution(model, test_data, tags2idx, words2idx)
            if acc > best_acc:
                torch.save(model.state_dict(), MODEL_NAME)
                print('one better model has saved.')
                best_acc = acc


def evalution(model, test_data, tags2idx, words2idx):
    model.eval()
    total = 0
    correct = 0
    for sents, labels in test_data:
        labels = tokens2seqs(labels, tags2idx, tag=True)
        sents = tokens2seqs(sents, words2idx)
        pred = model(sents)
        correct = correct + sum(
            int(pred[i] == labels[i]) for i in range(pred.size()[0]))
        total = total + pred.size()[0]
    print('accurancy on dev dataset is %f' % (correct / total))
    return correct / total


if __name__ == "__main__":
    # -------------------------data  part-------------------------
    cfg = Config()
    train_data = read_data(cfg.train_file)
    test_data = read_data(cfg.dev_file)
    words, words2idx, tags, tags2idx = statistic_data(train_data)
    # -------------------------model part-------------------------
    model = BiLSTM_CRF2(words2idx, tags2idx, EMBED, HIDDEN)
    if os.path.isfile(MODEL_NAME):
        model.load_state_dict(torch.load(MODEL_NAME))
        print('model has loaded.')
        acc = evalution(model, test_data, tags2idx, words2idx)
        print('current model accurancy on test data is %f' % acc)
        train(model, train_data, tags2idx, words2idx, test_data, best_acc=acc)
    else:
        train(model, train_data, tags2idx, words2idx, test_data)