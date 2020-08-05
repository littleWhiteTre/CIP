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
LR = 1e-3
MODEL_NAME = 'bilstm_crf.model'
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
    tags.extend(['<bos>', '<eos>'])
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
    def __init__(self, vocab, tags2idx, embed_dim, hidden_size):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = len(vocab)
        self.tags2idx = tags2idx
        self.tags_size = len(tags2idx)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(0.5)
        self.embedding = nn.Embedding(self.vocab_size,
                                      embed_dim,
                                      padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_size // 2,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_size, self.tags_size - 2)
        self.transition = nn.Parameter(
            torch.zeros(self.tags_size,
                        self.tags_size))  #[-1]:<eos> [-2]:<bos>
        self.transition.data[self.tags2idx['<eos>'], :] = -10000
        self.transition.data[:, self.tags2idx['<bos>']] = -10000

    def _score_sentence(self, emission, tags):
        '''
        input   emission:[n, tags_size-2]
                tags    :[n]
        return  sent's score: [1]
        '''
        score = torch.zeros(1)
        tags = torch.cat(
            [torch.tensor([self.tags2idx['<bos>']], dtype=torch.long), tags])
        for i, feat in enumerate(emission):
            score = score + feat[tags[i + 1]] + self.transition[tags[i],
                                                                tags[i + 1]]
        return score + self.transition[tags[-1], self.tags2idx['<eos>']]

    def _batch_score_sentence(self, batch_emission, batch_tags, n):
        '''
        input   batch_emission:[batch, text_lens, tags_size-2]
                batch_tags    :[batch, text_lens]
                n             :[batch]
        return  batch_scores  :[batch]
        '''
        batch_size = batch_emission.size()[0]
        batch_scores = torch.zeros(batch_size)
        for i in torch.arange(batch_size):
            batch_scores[i] = self._score_sentence(batch_emission[i, :n[i], :],
                                                   batch_tags[i, :n[i]])
        return batch_scores

    def _forward_alg(self, emission):
        '''
        emission : [n, tags_size-2]
        '''
        obs = emission[0] + self.transition[self.tags2idx['<bos>'], :
                                            -2]  # [tags_size - 2]
        # total_s = torch.log(torch.sum(torch.exp(obs+self.transition[self.tags2idx['<bos>'], :-2])))
        pre = obs
        pre = torch.unsqueeze(pre, dim=1)
        for i in range(1, emission.size()[0]):
            obs = torch.unsqueeze(emission[i], dim=0)
            score = pre + obs + self.transition[:-2, :
                                                -2]  # [tags_size-2, tags_size-2]
            if i == emission.size()[0] - 1:
                score = score + torch.unsqueeze(
                    self.transition[:-2, self.tags2idx['<eos>']], dim=0)
            vmax = torch.max(score, dim=1,
                             keepdim=True).values  #[tags_size-2, 1]
            pre = torch.log(
                torch.sum(torch.exp(score - vmax), dim=1, keepdim=True)) + vmax
        vmax = torch.max(pre, dim=0, keepdim=True).values
        total_s = torch.log(torch.sum(torch.exp(pre - vmax), dim=0)) + vmax
        return total_s.squeeze(0)

    def _batch_forward_alg(self, batch_emission, n):
        '''
        input   batch_emission : [batch, text_lens, tags_size-2]
                n              : [batch]
        return  batch_total_s  : [batch]
        '''
        batch_size = batch_emission.size()[0]
        batch_total_s = torch.zeros(batch_size)
        for i in torch.arange(batch_size):
            batch_total_s[i] = self._forward_alg(
                emission=batch_emission[i, :n[i], :])
        return batch_total_s

    def _viterbi_decode(self, emission):
        '''
        input   emission : [n, tags_size-2]
        '''
        alpha0 = []
        alpha1 = []  # lens of alpha1 should be 'n-1'.
        pre = emission[0]
        alpha0.append(pre)
        pre = torch.unsqueeze(pre, dim=1)
        for i in range(1, emission.size()[0]):
            obs = torch.unsqueeze(emission[i], dim=0)
            score = pre + obs + self.transition[:-2, :-2]
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

    def _batch_viterbi_decode(self, batch_emission, n):
        '''
        batch_emission : [batch, text_lens, tags_size-2]
        n : [batch_size]
        '''
        batch_size = batch_emission.size()[0]
        text_lens = batch_emission.size()[1]
        batch_best_seqs = torch.zeros(batch_size, text_lens)
        for i in torch.arange(batch_size):
            best_seq = self._viterbi_decode(batch_emission[i, :n[i]])
            batch_best_seqs[i, :n[i]] = best_seq
        return batch_best_seqs

    def _get_lstm_out(self, inp, n):
        '''
        input   inp:[batch, text_lens]   n:[batch]
        return  [batch, text_lens, tags_size]
        '''
        embed = self.embedding(inp)  # [batch, text_len, embed_dim]
        packed_inp = nn.utils.rnn.pack_padded_sequence(embed,
                                                       n,
                                                       batch_first=True,
                                                       enforce_sorted=False)
        lstm_o, (h, c) = self.lstm(packed_inp, None)
        lstm_o, _ = nn.utils.rnn.pad_packed_sequence(lstm_o, batch_first=True)
        # lstm_o = self.dropout(lstm_o)
        # lstm_o: [batch, seq_lens, hidden_size]
        # return F.softmax(self.fc(lstm_o), dim=2)  # [batch, seq_lens, tags_size-2]
        return self.fc(lstm_o)

    def forward(self, inp, n):
        feats = self._get_lstm_out(inp, n)  # [batch, text_lens, tags_size-2]
        best_tags = self._batch_viterbi_decode(feats, n)  # [batch, text_lens]
        return best_tags

    def neg_log_likelihood(self, batch_sents, batch_tags, n):
        '''
        input:
        batch_sents:[batch, text_lens]
        batch_tags :[batch, text_lens]
        n : [batch]
        '''
        feats = self._get_lstm_out(batch_sents,
                                   n)  # [batch, seq_lens, tags_size]
        # forward_score = self._forward_alg(emission=feats[0, :n[0], :])
        # real_score = self._score_sentence(feats[0, :n[0], :],
        #                                   batch_tags[0, :n[0]])
        # loss = forward_score - real_score
        # for i in torch.arange(1, batch_sents.size()[0]):
        #     forward_score += self._forward_alg(emission=feats[i, :n[i], :])
        #     real_score += self._score_sentence(feats[i, :n[i], :],
        #                                        batch_tags[i, :n[i]])
        #     loss += forward_score - real_score
        # return loss
        forward_score = self._batch_forward_alg(feats, n)
        real_score = self._batch_score_sentence(feats, batch_tags, n)
        print('1', forward_score)
        print('2', real_score)
        return torch.sum(forward_score - real_score)


def train(model, dl, epochs=5, lr=LR, test_dl=None, best_score=0):
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_score = best_score
    for epoch in range(1, epochs + 1):
        running_loss = 0
        dl_ = copy.deepcopy(dl)
        for step, batch in enumerate(dl_):
            labels, sents, n = batch  # labels:[batch, text_lens]
            batch_size = labels.size()[0]
            optim.zero_grad()
            loss = model.neg_log_likelihood(sents, labels, n)
            loss.backward()
            optim.step()
            running_loss += loss.item()
            # print(loss.item())
        print('%d running loss is %f' % (epoch, running_loss))
        if test_dl:
            acc = evalution(model, test_dl)
            print('accurancy on test dataset is %f' % (acc))
            acc2 = evalution(model, dl)
            print('accurancy on train dataset is %f' % (acc2))
            if acc > best_score:
                best_score = acc
                torch.save(model.state_dict(), MODEL_NAME)
                print('one better model has saved.')


def evalution(model, dl):
    model.eval()
    dl_ = copy.deepcopy(dl)
    correct = 0
    total = 0
    for step, batch in enumerate(dl_):
        labels, sents, n = batch  # labels:[batch, text_lens]
        batch_size = labels.size()[0]
        pred = model(sents, n)  # [batch, text_lens]
        for i in torch.arange(batch_size):
            i_label = labels[i, :n[i]]  # [n]
            i_pred = pred[i, :n[i]]  # [n]
            correct += sum(
                int(i_pred[j] == i_label[j]) for j in range(i_label.size()[0]))
            total += i_label.size()[0]
    model.train()
    return correct / total


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
    model = BiLSTM_CRF(words, tags2idx, EMBED, HIDDEN)
    if os.path.isfile(MODEL_NAME):
        model.load_state_dict(torch.load(MODEL_NAME))
        print('model has loaded.')
        acc = evalution(model, test_dl)
        print('init accurancy is %f' % acc)
        train(model, dl, epochs=EPOCHS, test_dl=test_dl, best_score=acc)
    else:
        train(model, dl, epochs=EPOCHS, test_dl=test_dl)
    # train(model, dl, epochs=EPOCHS)
    # evalution(model, test_dl)
