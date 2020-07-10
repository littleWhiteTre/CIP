import numpy as np
import pickle
from itertools import chain
import os
from tqdm import tqdm


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


class CRF:
    def __init__(self, data):
        self.data = data
        self._build_dicts()
        self._extract_feats()
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
            if len(sent) == 1:
                continue
            sent_feats = self._sent2feats(sent, tags)
            for word_feats in sent_feats:
                for feat in word_feats:
                    feats.add(feat)
        self.feats = list(sorted(feats))
        self.feats2idx = {feat: idx for idx, feat in enumerate(self.feats)}

    def _sent2feats(self, sent, tags):
        features = []
        for i, w in enumerate(sent):
            if i == 0:
                pre_tag = '<bos>'
                word_feats = self._word2feats(sent, i, tags[i], pre_tag)
            else:
                word_feats = self._word2feats(sent, i, tags[i], tags[i - 1])
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

    def _score_sentence(self, sent, tags):
        # 给定一个句子和一个标注序列，计算score
        sent_feats = self._sent2feats(sent, tags)
        sent_feats_vec = [self._feats2vec(i) for i in sent_feats]
        score = [self.w.dot(i) for i in sent_feats_vec]
        return np.sum(score)

    def _forward_alg(self, sent, k, t):
        scores = np.zeros([len(sent), len(self.tags)])
        for tag in self.tags:
            feats_0 = self._word2feats(sent, 0, tag, '<bos>')
            feats_0 = self.w.dot(self._feats2vec(feats_0))
            scores[0, self.tags2idx[tag]] = np.exp(feats_0)
        if k == 0:
            return scores[0, self.tags2idx[t]]
        for state in range(1, len(sent)):
            for cur_tag in self.tags:
                for pre_tag in self.tags:
                    feats_i = self._word2feats(sent, state, pre_tag, cur_tag)
                    feats_i = self.w.dot(self._feats2vec(feats_i))
                    scores[state, self.tags2idx[cur_tag]] = np.exp(
                        feats_i) * scores[state - 1, self.tags2idx[pre_tag]]
            if k == state:
                return scores[k, self.tags2idx[t]]

    def _backward_alg(self, sent, k, t):
        scores = np.zeros([len(sent), len(self.tags)])
        for tag in self.tags:
            for pre_tag in self.tags:
                feats_end = self._word2feats(sent, len(sent) - 1, tag, pre_tag)
                feats_end = self.w.dot(self._feats2vec(feats_end))
                scores[-1, self.tags2idx[tag]] = np.exp(feats_end)
        if k == len(sent) - 1:
            return scores[-1, self.tags2idx[t]]
        for state in range(len(sent) - 2, -1, -1):
            for cur_tag in self.tags:
                for pre_tag in self.tags:
                    feats_i = self._word2feats(sent, state, cur_tag, pre_tag)
                    feats_i = self.w.dot(self._feats2vec(feats_i))
                    scores[state, self.tags2idx[t]] = np.exp(feats_i) * scores[
                        state + 1, self.tags2idx[pre_tag]]
            if k == state:
                return scores[k, self.tags2idx[t]]

    def _viterbi_decode(self, words_list):
        max_p = np.zeros([len(words_list), len(self.tags)])
        path = np.zeros([len(words_list), len(self.tags)], dtype=np.int)
        for i, t in enumerate(self.tags):
            feats_0 = self._word2feats(words_list, 0, t, '<bos>')
            feats_0 = self._feats2vec(feats_0)
            score = self.w.dot(feats_0)
            max_p[0, self.tags2idx[t]] = score
        for i in range(1, len(words_list)):
            for cur_tag in self.tags:
                vals = []
                for pre_tag in self.tags:
                    feats_i = self._word2feats(words_list, i, cur_tag, pre_tag)
                    feats_i = self._feats2vec(feats_i)
                    score = self.w.dot(feats_i) + max_p[i - 1,
                                                        self.tags2idx[pre_tag]]
                    vals.append(score)
                max_p[i, self.tags2idx[cur_tag]] = np.max(vals)
                path[i, self.tags2idx[cur_tag]] = np.argmax(vals)
        best_seq = []
        best_tag = np.argmax(max_p[-1])
        best_seq.append(best_tag)
        for state in range(len(words_list), 1, -1):
            pre_tag = path[state - 1, best_tag]
            best_seq.append(pre_tag)
            best_tag = pre_tag
        best_seq.reverse()
        best_seq = [self.idx2tags[i] for i in best_seq]
        return best_seq

    def train(self, epochs=1):
        for epoch in range(epochs):
            for item in tqdm(self.data):
                sent, tags = zip(*item)
                # pred_tags = self._viterbi_decode(sent)
                sent_feats = [
                    self._feats2vec(i) for i in self._sent2feats(sent, tags)
                ]
                # f(S, Y)
                sent_feats = np.sum(sent_feats, axis=0)
                z_s = [
                    self._forward_alg(sent,
                                      len(sent) - 1, t) for t in self.tags
                ]
                z_s = np.sum(z_s)
                vec_1 = np.zeros(len(self.feats))
                for i, w in enumerate(sent):
                    if i == 0:
                        for t in self.tags:
                            word_feats = self._word2feats(sent, 0, t, '<bos>')
                            word_feats = self._feats2vec(word_feats)
                            alpha = self._forward_alg(sent, i, t)
                            beta = self._backward_alg(sent, i, t)
                            exp_score = np.exp(
                                self.w.dot(
                                    self._feats2vec(
                                        self._word2feats(sent, i, t,
                                                         '<bos>'))))
                            p = (alpha * beta * exp_score) / z_s
                            vec_1 += word_feats * p
                    else:
                        for t in self.tags:
                            for pre_tag in self.tags:
                                word_feats = self._word2feats(
                                    sent, 0, t, pre_tag)
                                word_feats = self._feats2vec(word_feats)
                                alpha = self._forward_alg(sent, i - 1, pre_tag)
                                beta = self._backward_alg(sent, i, t)
                                exp_score = np.exp(
                                    self.w.dot(
                                        self._feats2vec(
                                            self._word2feats(
                                                sent, i, t, pre_tag))))
                                p = (alpha * beta * exp_score) / z_s
                                vec_1 += word_feats * p
                    self.g = self.g + (sent_feats - vec_1)
            self.w = self.w + self.g
            self.g = np.zeros(len(self.feats))

    def predict(self, words_list):
        return self._viterbi_decode(words_list)

    def save_model(self, path, model_name='crf.model'):
        parameters = {'w': self.w, 'feats': self.feats, 'tags': self.tags}
        with open(path + model_name, 'wb') as f:
            pickle.dump(parameters, f)

    def load_model(self, path, model_name='crf.model'):
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
    sent, tags = zip(*data[0])
    model = CRF(data)
    model.train()
    # model.save_model(cfg.data_path)