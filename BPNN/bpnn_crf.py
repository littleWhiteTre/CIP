import numpy as np
import pickle
import os
import random
from tqdm import tqdm
from itertools import chain


# util functions.
def read_data(file):
    '''
    return data:
    [[('戴相龙', 'NR'), ('说', 'VV'),……]]
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
        tags, words = zip(*i)
        data_set.append((tags, words))
    return data_set


def statistic_data(data):
    words_set = set()
    tags_set = set()
    for item in data:
        w, t = item
        words_set.add(w)
        tags_set.add(t)
    words_set = list(words_set)
    words_set.append('<unk>')
    return words_set, list(tags_set)


# functions used to neural network computing.
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


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


class Cross_Entropy_Loss:
    @staticmethod
    def fn(a, y):
        return -np.sum(np.nan_to_num(y * np.log(a) + (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        return a - y

class Neg_Log_Likelihood_Loss:
    @staticmethod
    def fn(sum_p, score_real):
        return sum_p - score_real

    @staticmethod
    def delta():
        pass


class Network:
    def __init__(self, embedding_dim, hidden_size, word2idx, tag2idx, loss):
        # sizes: [embedding_dim, hidden_size, output_size]
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.tag2idx)}
        self.vocab_size = len(word2idx)
        self.tag_size = len(tag2idx)
        self.embedding = np.random.randn(self.vocab_size, embedding_dim)
        self.weights = [
            np.random.randn(hidden_size, embedding_dim),
            np.random.randn(self.tag_size, hidden_size)
        ]
        self.bias = [
            np.random.randn(hidden_size, 1),
            np.random.randn(self.tag_size, 1)
        ]
        self.transition = np.random.randn([self.tag_size, self.tag_size])
        self.loss = loss

    
    def _viterbi_decode(self, emission, words_list):
        e = emission
        t = self.transition
        n = self.tag_size
        obs = np.expand_dims(e[0], 0)
        pre = obs
        alpha1 = []
        alpha0 = []
        best_seq = []
        for i in range(1, len(words_list)):
            obs = np.expand_dims(e[i], 0)
            pre = np.repeat(pre.T, n, 1)
            obs = np.repeat(obs, n, 0)
            score = pre + obs + t
            pre = np.max(score, axis=0)
            alpha1.append(pre)
            alpha0.append(np.max(score, axis=0))
        idx = np.argmax(alpha0[-1])
        best_seq.append(idx)
        for i in range(len(alpha0)-2, -1, -1):
            pre = alpha1[i][idx]
            idx = pre
            best_seq.append(idx)
        best_seq.reverse()
        best_seq = [self.idx2tag[i] for i in best_seq]
        return best_seq


    def _forward_alg(self, emission):
        e = emission
        t = self.transition
        n = self.tag_size
        l = e.shape[0]
        obs = np.expand_dims(e[0], 0) # [1, n]
        pre = np.log(np.exp(obs)) # [1, n]
        for i in range(1, l):
            obs = np.expand_dims(e[i], 0)
            pre = np.repeat(pre.T, n, 1)
            obs = np.repeat(obs, n, 0)
            score = pre + obs + t
            pre = [np.log(np.sum(np.exp(score[j]))) for j in score.shape[0]]
            pre = np.array([pre])
        return np.log(np.sum(np.exp(pre)))
            
    
    def forward(self, inp):
        pass
    

    def save_model(self, data_path, model_name='bpnn_crf.model'):
        params = {
            'embed': self.embedding,
            'weights': self.weights,
            'bias': self.bias,
            'transition': self.transition,
            'word2idx': self.word2idx,
            'tag2idx': self.tag2idx
        }
        with open(data_path + model_name, 'wb') as f:
            pickle.dump(params, f)
        print('model has saved.')

    def load_model(self, data_path, model_name='bpnn_crf.model'):
        with open(data_path + model_name, 'rb') as f:
            params = pickle.load(f)
        self.weights = params['weights']
        self.bias = params['bias']
        self.transition = params['transition']
        self.embedding = params['embed']
        self.word2idx = params['word2idx']
        self.tag2idx = params['tag2idx']
        self.vocab_size = len(self.word2idx)
        self.tag_size = len(self.tag2idx)
        print('model has load.')


if __name__ == "__main__":
    # ------------------------------prepare data-----------------------------
    cfg = Config()
    train = read_data(cfg.train_file)
    test = read_data(cfg.dev_file)
    # --------------------------- ---model part----------- -------------------
    # model = Network(50, 300, word2idx, tag2idx, Cross_Entropy_Loss)
