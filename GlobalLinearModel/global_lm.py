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
    print('precision:', precision)
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


class Global_lm:
    def __init__(self, data):
        self.data = data
        self._build_dicts()
        self._extract_feats()
        self.feats2idx = {feat: idx for idx, feat in enumerate(self.feats)}
        self.w = np.zeros(len(self.feats))

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
            if len(sent) == 1:
                continue
            sent_feats = self._sent2feats(sent, tags)
            for word_feats in sent_feats:
                for feat in word_feats:
                    feats.add(feat)
        self.feats = list(sorted(feats))

    def _sent2feats(self, sent, tags):
        features = []
        for i, w in enumerate(sent):
            if i == 0:
                pre_tag = '<bos>'
                word_feats = self._word2feats(sent, i, tags[i], pre_tag)
            else:
                word_feats = self._word2feats(sent, i, tags[i], tags[i-1])
            features.append(word_feats)
        return features

    def _word2feats(self, sent, i, cur_tag, pre_tag):
        feat1 = '01=' + cur_tag + pre_tag
        feat2 = '02=' + cur_tag + sent[i]
        if i == 0:
            feat3 = '03=' + cur_tag + '<bos>'
            feat5 = '05=' + cur_tag + sent[i] + '<bos>'[-1]
            feat4 = '04=' + cur_tag + sent[i + 1]
            feat6 = '06=' + cur_tag + sent[i] + sent[i + 1][0]
        elif i == len(sent) - 1:
            feat3 = '03=' + cur_tag + sent[i - 1]
            feat5 = '05=' + cur_tag + sent[i] + sent[i - 1][-1]
            feat4 = '04=' + cur_tag + '<eos>'
            feat6 = '06=' + cur_tag + sent[i] + '<eos>'[0]
        else:
            feat3 = '03=' + cur_tag + sent[i - 1]
            feat5 = '05=' + cur_tag + sent[i] + sent[i - 1][-1]
            feat4 = '04=' + cur_tag + sent[i + 1]
            feat6 = '06=' + cur_tag + sent[i] + sent[i + 1][0]
        return [feat1, feat2, feat3, feat4, feat5, feat6]

    def _feats2vec(self, feats):
        feats_vec = np.zeros(len(self.feats), dtype=np.int)
        for i in feats:
            if i in self.feats:
                feats_vec[self.feats2idx[i]] += 1
        return feats_vec

    def _feats_score(self, feats):
        idx = []
        for i in feats:
            if i in self.feats:
                idx.append(self.feats2idx[i])
        return self.w[idx].sum()

    def _viterbi_decode(self, words_list):
        max_p = np.zeros([len(words_list), len(self.tags)])
        path = np.zeros([len(words_list), len(self.tags)], dtype=np.int)
        vals = []
        for i, t in enumerate(self.tags):
            feats_0 = self._word2feats(words_list, 0, t, '<bos>')
            feats_0 = self._feats2vec(feats_0)
            score = self.w.dot(feats_0)
            # score = self._feats_score(feats_0)
            max_p[0, self.tags2idx[t]] = score
        for i in range(1, len(words_list)):
            for cur_tag in self.tags:
                vals = []
                for pre_tag in self.tags:
                    feats_i = self._word2feats(words_list, i, cur_tag, pre_tag)
                    feats_i = self._feats2vec(feats_i)
                    score = self.w.dot(feats_i) + max_p[i-1, self.tags2idx[pre_tag]]
                    # score = self._feats_score(feats_i) + max_p[i-1, self.tags2idx[pre_tag]]
                    vals.append(score)
                max_p[i, self.tags2idx[cur_tag]] = np.max(vals)
                path[i, self.tags2idx[cur_tag]] = np.argmax(vals)
        best_seq = []
        best_tag = np.argmax(max_p[-1])
        best_seq.append(best_tag)
        for state in range(len(words_list), 1, -1):
            pre_tag = path[state-1, best_tag]
            best_seq.append(pre_tag)
            best_tag = pre_tag
        best_seq.reverse()
        best_seq = [self.idx2tags[i] for i in best_seq]
        return best_seq

    def predict(self, words_list):
        return self._viterbi_decode(words_list)

    def train(self):
        print('training ...')
        for item in tqdm(self.data):
            sent, tags = zip(*item)
            pred_tags = self.predict(sent)
            if pred_tags != tags:
                crt_feats = self._sent2feats(sent, tags)
                crt_feats_vec = np.zeros(len(self.feats), dtype=np.int)
                for i in crt_feats:
                    crt_feats_vec += self._feats2vec(i)
                error_feats = self._sent2feats(sent, pred_tags)
                error_feats_vec = np.zeros(len(self.feats), dtype=np.int)
                for i in error_feats:
                    error_feats_vec += self._feats2vec(i)
                self.w = self.w + crt_feats_vec - error_feats_vec
        print('training finished.')

    def save_model(self, path, model_name='global_lm.model'):
        parameters = {'w': self.w, 'feats': self.feats, 'tags': self.tags}
        with open(path + model_name, 'wb') as f:
            pickle.dump(parameters, f)

    def load_model(self, path, model_name='global_lm.model'):
        with open(path + model_name, 'rb') as f:
            parameters = pickle.load(f)
        self.w = parameters['w']
        self.feats = parameters['feats']
        self.feats2idx = {feat: idx for idx, feat in enumerate(self.feats)}
        self.tags = parameters['tags']
        self.tags2idx = {tag: idx for idx, tag in enumerate(self.tags)}
                

if __name__ == "__main__":
    cfg = Config()
    data = read_data(cfg.train_file)
    model = Global_lm(data)
    model.train()
    model.save_model(cfg.data_path)
    # sent, tags = zip(*data[0])
    # print(sent, tags)
    # print(model._sent2feats(sent, tags))
    # print(model.predict(sent))
