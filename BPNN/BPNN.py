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
    return data


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


class Network:
    def __init__(self, embedding_dim, hidden_size, word2idx, tag2idx, loss):
        # sizes: [embedding_dim, hidden_size, output_size]
        self.word2idx = word2idx
        self.tag2idx = tag2idx
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
        self.loss = loss

    def forward(self, inp):
        # inp : one-hot number.
        a = np.expand_dims(self.embedding[inp], 1)
        for w, b in zip(self.weights, self.bias):
            z = w.dot(a) + b
            a = sigmoid(z)
        return a

    def SGD(self,
            training_data,
            epochs,
            minibatch_size,
            eta,
            test_data=None,
            shuffle=True):
        for epoch in range(epochs):
            if shuffle:
                random.shuffle(training_data)
            batchs = []
            n = len(training_data)
            running_loss = 0
            for k in range(0, n, minibatch_size):
                batch = training_data[k:k + minibatch_size]
                batchs.append(batch)
            for batch in tqdm(batchs):
                loss = self.update_batch_data(batch, eta)
                running_loss = running_loss + loss
            if test_data:
                self.evelution(test_data)
                self.evelution(training_data)
            print('%d, loss is:%f' % (epoch+1, running_loss))

    def update_batch_data(self, batch_data, eta):
        n = len(batch_data)
        delta_w = [np.zeros(i.shape) for i in self.weights]
        delta_b = [np.zeros(i.shape) for i in self.bias]
        for item in batch_data:
            word, tag = item
            word = self.word2idx[word]
            x = np.expand_dims(self.embedding[word], 1)
            y = np.zeros([self.tag_size, 1])
            y[self.tag2idx[tag]] = 1
            nabla_delta_w, nabla_delta_b = self.backward(x, y)
            delta_w = [dw + ndw for dw, ndw in zip(delta_w, nabla_delta_w)]
            delta_b = [db + ndb for db, ndb in zip(delta_b, nabla_delta_b)]
        self.weights = [
            w - (eta / n) * dw for w, dw in zip(self.weights, delta_w)
        ]
        self.bias = [
            b - (eta / n) * db for b, db in zip(self.bias, delta_b)
        ]
        loss = self.loss.fn(self.forward(word), y)
        return loss / n

    def backward(self, x, y):
        # feed forward.
        activation = x
        activations = [x]
        z_s = []
        for w, b in zip(self.weights, self.bias):
            z = w.dot(activation) + b
            z_s.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # back.
        delta = self.loss.delta(z_s[-1], activation, y)
        delta_w2 = delta.dot(
            activations[-2].transpose())  #[output, 1] [1, hidden_size]
        delta_b2 = delta
        delta_ = np.dot(self.weights[-1].transpose(), delta) * sigmoid_prime(
            z_s[0])
        delta_w1 = delta_.dot(x.transpose())  # [] * [] --> [hidden, input]
        delta_b1 = delta_
        return [delta_w1, delta_w2], [delta_b1, delta_b2]

    def evelution(self, test_data):
        words, tags = zip(*test_data)
        onehot = []
        for w in words:
            if w in self.word2idx:
                onehot.append(self.word2idx[w])
            else:
                onehot.append(self.word2idx['<unk>'])
        tags = [self.tag2idx[i] for i in tags]
        pred = [np.argmax(self.forward(i)) for i in onehot]
        result = sum(int(pred[i] == tags[i]) for i in range(len(pred)))
        print(result, '/', len(tags))


if __name__ == "__main__":
    # ------------------------------prepare data-----------------------------
    cfg = Config()
    train = read_data(cfg.train_file)
    test = read_data(cfg.dev_file)
    train = [j for i in train for j in i]
    test = [j for i in test for j in i]
    words, tags = statistic_data(train)
    word2idx = {w: idx for idx, w in enumerate(words)}
    tag2idx = {t: idx for idx, t in enumerate(tags)}
    # --------------------------- ---model part----------- -------------------
    model = Network(50, 30, word2idx, tag2idx, Cross_Entropy_Loss)
    model.SGD(train, 20, 32, 1, test_data=test)
