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


class Linear_model:
    def __init__(self, data, use_partial=False):
        self.data = data
        self.use_partial = use_partial
        self._extract_feats()
        # self.feats    self.partical_feats
        self._build_dicts()
        # self.words    self.tags   self.chars
        # self.words2idx    self.tags2idx   self.chars2idx
        # self.idx2tags
        self.feats2idx = {feat: idx for idx, feat in enumerate(self.feats)}
        self.partical_feats2idx = {
            feat: idx
            for idx, feat in enumerate(self.partical_feats)
        }
        self.w = np.random.rand(len(self.feats))
        self.w_partial = np.random.rand(len(self.tags),
                                        len(self.partical_feats))

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
        partial_feats = set()
        for item in self.data:
            sent, tags = zip(*item)
            sent_feats = self._sent2feats(sent, tags)
            sent_partial_feats = self._sent2feats(sent, tags, partial=True)
            for word_feats in sent_feats:
                for feat in word_feats:
                    feats.add(feat)
            for partial_feat in sent_partial_feats:
                for feat in partial_feat:
                    partial_feats.add(feat)
        self.feats = list(sorted(feats))
        self.partical_feats = list(sorted(partial_feats))

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

    def _word2partial_feats(self, sent, i):
        word = sent[i]
        features = ['bias', '02=' + word]
        # 单词 wi 之前的特征
        if i > 0:
            word1 = sent[i - 1]
            features.extend(['03=' + word1, '05=' + word1[-1]])
        else:
            features.extend(['03=' + 'BOS', '05=' + 'S'])
        # 单词 wi 之后的特征
        if i < (len(sent) - 1):
            word1 = sent[i + 1]
            features.extend(['04=' + word1, '06=' + word1[0]])
        else:
            features.extend(['04=' + 'EOS', '06=' + 'E'])

        return features

    def _sent2feats(self, sent, tags, partial=False):
        if not partial:
            return [
                self._word2feats(sent, i, tags[i]) for i in range(len(sent))
            ]
        else:
            return [
                self._word2partial_feats(sent, i) for i in range(len(sent))
            ]

    def _feats2vec(self, feats, use_partial=False):
        if use_partial:
            feats_vec = np.zeros(len(self.partical_feats), dtype=np.int)
            for feat in feats:
                if feat in self.partical_feats:
                    feats_vec[self.partical_feats2idx[feat]] = 1
            return feats_vec
        else:
            feats_vec = np.zeros(len(self.feats), dtype=np.int)
            for feat in feats:
                if feat in self.feats:
                    feats_vec[self.feats2idx[feat]] = 1
            return feats_vec

    def train(self):
        for item in tqdm(self.data):
            sent, tags = zip(*item)
            for i in range(len(tags)):
                word = sent[i]
                tag = tags[i]
                pred_tag = self.predict_word(sent, i, self.use_partial)
                if pred_tag != tag:
                    if self.use_partial:
                        feats = self._feats2vec(
                            self._word2partial_feats(sent, i),
                            self.use_partial)
                        y_offset = self.tags2idx[tag]
                        pred_offset = self.tags2idx[pred_tag]
                        self.w_partial[y_offset] = self.w_partial[y_offset] + feats
                        self.w_partial[pred_offset] = self.w_partial[pred_offset] - feats
                    else:
                        feats_y = self._feats2vec(
                            self._word2feats(sent, i, tag))
                        feats_pred = self._feats2vec(
                            self._word2feats(sent, i, pred_tag))
                        self.w = self.w + feats_y - feats_pred

    def predict(self, word_list):
        best_tags = []
        for i in range(len(word_list)):
            tag = self.predict_word(word_list, i, self.use_partial)
            best_tags.append(tag)
        return best_tags

    def predict_word(self, word_list, i, use_partial=False):
        vals = []
        for tag in self.tags:
            if use_partial:
                word_feats = self._word2partial_feats(word_list, i)
                offset = self.tags2idx[tag]
                feats_vec = self._feats2vec(word_feats, use_partial)
                score = self.w_partial[offset].dot(feats_vec)
                vals.append(score)
            else:
                word_feats = self._word2feats(word_list, i, tag)
                feats_vec = self._feats2vec(word_feats)
                score = self.w.dot(feats_vec)
                vals.append(score)
        return self.tags[np.argmax(vals)]

    def save_model(self, path):
        np.save(path + 'weights', self.w)
        np.save(path + 'weights_partial', self.w_partial)
        with open(path + 'feats.pkl', 'wb') as f:
            pickle.dump(self.feats, f)
        with open(path + 'partial_feats.pkl', 'wb') as f:
            pickle.dump(self.partical_feats, f)

    def load_model(self, path):
        self.w = np.load(path + 'weights.npy')
        self.w_partial = np.load(path + 'weights_partial.npy')
        with open(path + 'feats.pkl', 'rb') as f:
            self.feats = pickle.load(f)
        with open(path + 'partial_feats.pkl', 'rb') as f:
            self.partical_feats = pickle.load(f)
        self.feats2idx = {feat: idx for idx, feat in enumerate(self.feats)}
        self.partical_feats2idx = {
            feat: idx
            for idx, feat in enumerate(self.partical_feats)
        }


if __name__ == "__main__":
    cfg = Config()
    data = read_data(cfg.train_file)
    dev_data = read_data(cfg.dev_file)
    model = Linear_model(data, use_partial=True)
    model.load_model(cfg.data_path)
    # evalution(dev_data, model)
    sent, tags = zip(*data[0])
    print(tags)
    print(model.predict(sent))
    # model.load_model(cfg.data_path)
    # precision = evalution(dev_data, model)
    # print('precision:', precision)
    #precision： 0.698
    # model.train()
    # model.save_model(cfg.data_path)