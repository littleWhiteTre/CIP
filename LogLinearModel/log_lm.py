import numpy as np
from itertools import chain
import os
from tqdm import tqdm
import pickle


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

def evalution(dev_data, model):
    precision = 0.
    print('evaluting model.')
    for sent in tqdm(dev_data):
        words_list = [i[0] for i in sent]
        labels_list = [i[1] for i in sent]
        predict_labels = model.predict(words_list)
        right_labels = []
        for j, label in enumerate(predict_labels):
            if label == labels_list[j]:
                right_labels.append(label)
        sent_precision = len(right_labels) / len(predict_labels)
        precision = precision + sent_precision
    precision = precision / len(dev_data)
    return precision


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
        self.alpha = 0.3


class Log_lm:
    def __init__(self, data):
        self.data = data
        self._build_dicts()
        self._extract_feats()
        self.feats2idx = {feat: idx for idx, feat in enumerate(self.feats)}
        self.w = np.zeros(len(self.feats))
        self.g = np.zeros(len(self.feats))

    def _build_dicts(self):
        words, tags = zip(*(chain(*self.data)))
        words = list(set(words))
        tags = list(set(tags))
        chars = set()
        for word in words:
            for char in word:
                chars.add(char)
        chars = list(chars)
        self.words = words
        self.tags = tags
        self.chars = chars
        self.chars2idx = {char: idx for idx, char in enumerate(chars)}
        self.words2idx = {word: idx for idx, word in enumerate(words)}
        self.tags2idx = {tag: idx for idx, tag in enumerate(tags)}
        self.idx2tags = {self.tags2idx[tags]: tags for tags in self.tags2idx}

    def _extract_feats(self):
        feats = set()
        for item in self.data:
            sent, tags = zip(*item)
            sent_feats = self._sent2feats(sent, tags)
            for word_feats in sent_feats:
                for feat in word_feats:
                    feats.add(feat)
        self.feats = list(sorted(feats))

    def _word2feats(self, sent, i, tag):
        word = sent[i]
        features = ['bias', '02=' + tag + word]
        # 单词 wi 之前的特征
        if i > 0:
            word1 = sent[i - 1]
            features.extend(['03=' + tag + word1, '05=' + tag + word1[-1]])
        else:
            features.extend(['03=' + tag + 'BOS', '05=' + tag + 'S'])
        # 单词 wi 之后的特征
        if i < (len(sent) - 1):
            word1 = sent[i + 1]
            features.extend(['04=' + tag + word1, '06=' + tag + word1[0]])
        else:
            features.extend(['04=' + tag + 'EOS', '06=' + tag + 'E'])

        return features

    def _sent2feats(self, sent, tags):
        return [self._word2feats(sent, i, tags[i]) for i in range(len(sent))]

    def _feats2vec(self, feats):
        feats_vec = np.zeros(len(self.feats), dtype=np.int)
        for feat in feats:
            if feat in self.feats:
                feats_vec[self.feats2idx[feat]] = 1
        return feats_vec
        
    def _caculate_probs(self, sent, i):
        probs = np.zeros(len(self.tags))
        for j, tag in enumerate(self.tags):
            feats_vec = self._feats2vec(self._word2feats(sent, i, tag))
            probs[j] = np.exp(self.w.dot(feats_vec))
        return probs

    def train(self, epochs=1, batch_size=50):
        b = 0
        for epoch in range(epochs):
            for item in tqdm(self.data):
                sent, tags = zip(*item)
                for i, word in enumerate(sent):
                    feats_y = self._feats2vec(self._word2feats(sent, i, tags[i]))
                    probs = self._caculate_probs(sent, i)
                    for tag in self.tags:
                        feats_y_ = self._feats2vec(self._word2feats(sent, i, tag))
                        prob_y_ = probs[self.tags2idx[tag]] / np.sum(probs)
                        feats_y = feats_y - prob_y_ * feats_y_
                    self.g = self.g + feats_y
                    b = b + 1
                    if b == batch_size:
                        self.w = self.w + self.g
                        self.g = np.zeros(len(self.feats))
                        b = 0
                self.w = self.w + self.g
            # do evalution here.

    def predict(self, word_list):
        best_tags = []
        for i in range(len(word_list)):
            vals = []
            for tag in self.tags:
                word_feats = self._word2feats(word_list, i, tag)
                feats_vec = self._feats2vec(word_feats)
                score = self.w.dot(feats_vec)
                vals.append(score)
            best_tags.append(self.idx2tags[np.argmax(vals)])
        return best_tags

    def save_model(self, path):
        np.save(path + 'log_lm_weights', self.w)
        with open(path + 'feats.pkl', 'wb') as f:
            pickle.dump(self.feats, f)
        
    def load_model(self, path):
        self.w = np.load(path + 'log_lm_weights.npy')
        with open(path + 'feats.pkl', 'rb') as f:
            self.feats = pickle.load(f)
        self.feats2idx = {feat: idx for idx, feat in enumerate(self.feats)}


if __name__ == "__main__":
    cfg = Config()
    data = read_data(cfg.train_file)
    dev_data = read_data(cfg.dev_file)
    model = Log_lm(data)
    model.train()
    model.save_model(cfg.data_path)
    # sent, tags = zip(*data[0])
    # print(tags)
    # print(model.predict(sent))