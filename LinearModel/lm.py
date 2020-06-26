import numpy as np
from itertools import chain
import os


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
        work_path = os.getcwd().split('\\')
        if work_path[-1] != 'CIP':
            self.train_file = '../data/train.conll'
            self.dev_file = '../data/dev.conll'
        else:    
            self.train_file = './data/train.conll'
            self.dev_file = './data/dev.conll'
        self.alpha = 0.3


class Linear_model:
    def __init__(self, data):
        self.data = data
        self.feats = self._extract_feats()
        self.feats2idx = {feat: idx for idx, feat in enumerate(self.feats)}
        self.w = np.random.rand(len(self.feats))
        # self._build_dicts()
        # self.words    self.tags   self.chars
        # self.words2idx    self.tags2idx   self.chars2idx
        # self.idx2tags

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
        return list(sorted(feats))

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
        
    def train(self):
        pass
                

    def predict(self, word_list):
        best_tags = []    
        for i in range(len(word_list)):
            tag = self.predict_word(word_list, i)
            best_tags.append(tag)
        return best_tags

    def predict_word(self, word_list, i):
        vals = []
        for tag in self.tags:
            word_feats = self._word2feats(word_list, i, tag)
            feats_vec = np.zeros(len(self.feats), dtype=np.int)
            for feat in word_feats:
                if feat in self.feats:
                    feats_vec[self.feats2idx[feat]] = 1
            score = self.w.dot(feats_vec)
            vals.append(score)
        return self.tags[np.argmax(vals)]

if __name__ == "__main__":
    cfg = Config()
    data = read_data(cfg.train_file)
    model = Linear_model(data)
    word_list, tags = zip(*data[0])
    print(model.predict(word_list))