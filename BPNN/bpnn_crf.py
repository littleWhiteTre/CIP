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
    [((NR, NR, NN, ...), ('戴相龙', '说', '中国'))]
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
    words2idx = {word: idx for idx, word in enumerate(words)}
    tags2idx = {tag: idx for idx, tag in enumerate(tags)}
    return words, words2idx, tags, tags2idx


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
    def fn(log_sum_exp_scores, real_score):
        return log_sum_exp_scores - real_score

    @staticmethod
    def delta(sum_exp_scores, exp_real_score):
        # d-loss / d-xiyi
        return (exp_real_score / sum_exp_scores) - 1


class Network:
    def __init__(self, embedding_dim, hidden_size, words2idx, tags2idx, loss):
        # sizes: [embedding_dim, hidden_size, output_size]
        self.words2idx = words2idx
        self.tags2idx = tags2idx
        self.vocab_size = len(words2idx)
        self.tag_size = len(tags2idx)
        self.hidden_size = hidden_size
        self.embed_dim = embedding_dim
        self.embedding = np.random.randn(self.vocab_size, embedding_dim)
        self.weights = [
            np.random.randn(hidden_size, embedding_dim),
            np.random.randn(self.tag_size, hidden_size)
        ]
        self.bias = [
            np.random.randn(hidden_size, 1),
            np.random.randn(self.tag_size, 1)
        ]
        self.transition = np.random.randn(self.tag_size, self.tag_size)
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
        for i in range(len(alpha0) - 2, -1, -1):
            pre = alpha1[i][idx]
            idx = pre
            best_seq.append(idx)
        best_seq.reverse()
        # best_seq = [self.tags[i] for i in best_seq]
        return best_seq

    def _forward_alg(self, emission):
        e = emission
        t = self.transition
        n = self.tag_size
        l = e.shape[0]
        obs = np.expand_dims(e[0], 0)  # [1, n]
        pre = np.log(np.exp(obs))  # [1, n]
        for i in range(1, l):
            obs = np.expand_dims(e[i], 0)
            pre = np.repeat(pre.T, n, 1)
            obs = np.repeat(obs, n, 0)
            score = pre + obs + t
            pre = [
                np.log(np.sum(np.exp(score[j]))) for j in range(score.shape[0])
            ]
            pre = np.array([pre])
        return np.log(np.sum(np.exp(pre)))

    def _score_sentence(self, emission, tags):
        score = 0
        for idx, tag in enumerate(tags):
            emit_score = emission[idx, tag]
            if idx != 0:
                trans_score = self.transition[tags[idx - 1], tags[idx]]
            else:
                trans_score = 0
            score += (emit_score + trans_score)
        return score

    def forward(self, inp):
        # inp type is number.
        inp = np.expand_dims(self.embedding[inp], axis=1)
        a = inp
        for w, b in zip(self.weights, self.bias):
            z = w.dot(a) + b
            a = sigmoid(z)
        return a

    def SGD(self, data, batch_size, epochs=10, learning_rate=3, shuffle=True):
        for epoch in range(1, epochs+1):
            running_loss = 0
            if shuffle:
                random.shuffle(data)
            batchs = []
            for i in range(0, len(data), batch_size):
                batchs.append(data[i:i + batch_size])
            for idx, mini_batch in enumerate(batchs):
                loss = self._update_minibatch(mini_batch, learning_rate)
                # print('%d epoch %d step loss is %f' % (epoch, idx, loss))
                running_loss += loss
            print('%d epoch, running loss is %f' % (epoch, running_loss))

    def _update_minibatch(self, minibatch, learning_rate):
        delta_w = [np.zeros(i.shape) for i in self.weights]
        delta_b = [np.zeros(i.shape) for i in self.bias]
        n = len(minibatch)
        for item in minibatch:
            tags = [self.tags2idx[i] for i in item[1]]
            words = []
            for i in item[0]:
                if i in self.words2idx:
                    words.append(words2idx[i])
                else:
                    words.append(words2idx['<unk>'])
            nabla_delta_w, nabla_delta_b, delta_trans = self._backward(
                tags, words)
            delta_w = [dw + ndw for dw, ndw in zip(delta_w, nabla_delta_w)]
            delta_b = [db + ndb for db, ndb in zip(delta_b, nabla_delta_b)]
        self.weights = [
            w - (learning_rate / n) * dw
            for w, dw in zip(self.weights, delta_w)
        ]
        self.bias = [
            b - (learning_rate / n) * db for b, db in zip(self.bias, delta_b)
        ]
        self.transition = self.transition + (learning_rate / n) * delta_trans
        # log_sum_exp_scores, real_score
        emission = np.zeros([len(words), self.tag_size])
        for idx, word in enumerate(words):
            emission[idx] = np.squeeze(self.forward(word), axis=1)
        log_sum_exp_scores = self._forward_alg(emission)
        real_score = self._score_sentence(emission, tags)
        loss = self.loss.fn(log_sum_exp_scores, real_score)
        return loss

    def _backward(self, tags, words):
        # feed forward.
        p_s = np.zeros([len(words), self.tag_size
                        ])  # every word's possible on tags, emission score.
        w_activations = []
        w_z_s = []
        for idx, word in enumerate(words):
            embed = np.expand_dims(self.embedding[word], 1)
            activation = embed
            activations = [activation]
            z_s = []
            for w, b in zip(self.weights, self.bias):
                z = w.dot(activation) + b
                z_s.append(z)
                a = sigmoid(z)
                activations.append(a)
                activation = a
            p_s[idx] = np.squeeze(activation, axis=1)
            w_activations.append(activations)
            w_z_s.append(z_s)
        log_sum_scores = self._forward_alg(p_s)
        real_scores = self._score_sentence(p_s, tags)
        # backward.
        delta_trans = np.zeros([self.tag_size, self.tag_size])
        avg_delta_w2 = np.zeros([self.tag_size, self.hidden_size])
        avg_delta_w1 = np.zeros([self.hidden_size, self.embed_dim])
        avg_delta_b2 = np.zeros([self.tag_size, 1])
        avg_delta_b1 = np.zeros([self.hidden_size, 1])
        for i in range(len(words) - 1, -1, -1):
            part_acts = w_activations[i]
            part_zs = w_z_s[i]
            delta = self.loss.delta(np.exp(log_sum_scores),
                                    np.exp(real_scores))
            grad_xi_y = np.zeros([self.tag_size, 1])
            grad_xi_y[tags[i], 0] = 1
            delta = delta * grad_xi_y * sigmoid_prime(
                part_zs[-1])  # n * [n, 1] *
            delta_w2 = np.dot(delta, part_acts[-2].T)  # [y, h]
            delta_b2 = delta  # [y, 1]
            delta_ = np.dot(self.weights[-1].T, delta) * sigmoid_prime(
                part_zs[0])  #[h, 1]
            delta_w1 = np.dot(delta_, part_acts[0].T)  # [h, x]
            delta_b1 = delta_  # [h, 1]
            avg_delta_w2 += delta_w2
            avg_delta_w1 += delta_w1
            avg_delta_b2 += delta_b2
            avg_delta_b1 += delta_b1
            if i != 0:
                delta_trans[tags[i - 1], tags[i]] += 1
        avg_delta_w2 = avg_delta_w2 / len(words)
        avg_delta_w1 = avg_delta_w1 / len(words)
        avg_delta_b2 = avg_delta_b2 / len(words)
        avg_delta_b1 = avg_delta_b1 / len(words)
        return [avg_delta_w1, avg_delta_w2], [avg_delta_b1,
                                              avg_delta_b2], delta_trans

    def save_model(self, data_path, model_name='bpnn_crf.model'):
        params = {
            'embed': self.embedding,
            'weights': self.weights,
            'bias': self.bias,
            'transition': self.transition,
            'words2idx': self.words2idx,
            'tags2idx': self.tags2idx
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
        self.words2idx = params['words2idx']
        self.tags2idx = params['tags2idx']
        self.vocab_size = len(self.words2idx)
        self.tag_size = len(self.tags2idx)
        print('model has load.')


if __name__ == "__main__":
    # ------------------------------prepare data-----------------------------
    cfg = Config()
    train = read_data(cfg.train_file)
    test = read_data(cfg.dev_file)
    words, words2idx, tags, tags2idx = statistic_data(train)
    # --------------------------- ---model part----------- -------------------
    model = Network(50, 300, words2idx, tags2idx, Neg_Log_Likelihood_Loss)
    model.SGD(train, 32)
