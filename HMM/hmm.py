import numpy as np
from itertools import chain


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
    recall = 0.
    for sent in dev_data:
        words_list = [i[0] for i in sent]
        labels_list = [i[1] for i in sent]
        predict_labels = model.viterbi_decode(words_list)
        right_labels = []
        for j, label in enumerate(predict_labels):
            if label == labels_list[j]:
                right_labels.append(label)
        sent_precision = len(right_labels) / len(predict_labels)
        sent_recall = len(right_labels) / len(labels_list)
        precision = precision + sent_precision
        recall = recall + sent_recall
    precision = precision / len(dev_data)
    recall = recall / len(dev_data)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


class Config:
    def __init__(self):
        self.train_file = '../data/train.conll'
        self.dev_file = '../data/dev.conll'
        self.alpha = 0.3


class HMM:
    def __init__(self, data):
        self.data = data
        self._build_dicts()
        self.transition = np.zeros([len(self.tags), len(self.tags)])
        self.emission = np.zeros([len(self.tags), len(self.words)])

    def _build_dicts(self):
        words, tags = zip(*(chain(*self.data)))
        words = list(set(words))
        tags = list(set(tags))
        words.append('<unk>')
        tags.append('<bos>')
        tags.append('<eos>')
        self.words = words
        self.tags = tags
        self.words2idx = {word: idx for idx, word in enumerate(words)}
        self.tags2idx = {tag: idx for idx, tag in enumerate(tags)}
        self.idx2tags = {self.tags2idx[tags]: tags for tags in self.tags2idx}

    def statistic_parameters(self, alpha):
        # 统计
        for sent in data:
            first_tag = sent[0][1]
            first_word = sent[0][0]
            self.emission[self.tags2idx['<bos>'],
                          self.words2idx[first_word]] += 1
            self.transition[self.tags2idx['<bos>'],
                            self.tags2idx[first_tag]] += 1
            for i in range(len(sent) - 1):
                cur_tag = sent[i][1]
                next_tag = sent[i + 1][1]
                word = sent[i][0]
                self.transition[self.tags2idx[cur_tag],
                                self.tags2idx[next_tag]] += 1
                self.emission[self.tags2idx[cur_tag],
                              self.words2idx[word]] += 1
            end_tag = sent[-1][1]
            end_word = sent[-1][0]
            self.emission[self.tags2idx['<eos>'],
                          self.words2idx[end_word]] += 1
            self.transition[self.tags2idx['<eos>'],
                            self.tags2idx[end_tag]] += 1
        # 平滑
        for i in range(self.transition.shape[0]):
            s = np.sum(self.transition[i])
            for j in range(self.transition.shape[1]):
                self.transition[i,
                                j] = (self.transition[i, j] +
                                      alpha) / (s + alpha * (len(self.tags)))
        for i in range(self.emission.shape[0]):
            s = np.sum(self.emission[i])
            for j in range(self.emission.shape[1]):
                self.emission[i, j] = (self.emission[i, j] +
                                       alpha) / (s + alpha * (len(self.words)))

    def viterbi_decode(self, words_list):
        for i in range(len(words_list)):
            word = words_list[i]
            if word not in self.words:
                words_list[i] = '<unk>'
        best_seq = []
        max_p = np.zeros([len(words_list), len(self.tags)])
        path = np.zeros([len(words_list), len(self.tags)], dtype=np.int)
        # 初始状态
        for i, t in enumerate(self.tags2idx):
            transition_score = self.transition[self.tags2idx['<bos>'],
                                               self.tags2idx[t]]
            emission_score = self.emission[self.tags2idx[t],
                                           self.words2idx[words_list[0]]]
            max_p[0, i] = transition_score * emission_score
            path[0, i] = self.tags2idx['<bos>']
        # 迭代
        for state in range(1, len(words_list)):
            for tags in self.tags2idx:
                vals = []
                for tags_ in self.tags2idx:
                    pre_p = max_p[state - 1, self.tags2idx[tags_]]
                    transition_score = self.transition[self.tags2idx[tags_],
                                                       self.tags2idx[tags]]
                    emission_score = self.emission[
                        self.tags2idx[tags], self.words2idx[words_list[state]]]
                    score = pre_p * transition_score * emission_score
                    vals.append(score)
                max_p[state, self.tags2idx[tags]] = np.max(vals)
                path[state, self.tags2idx[tags]] = np.argmax(vals)
            # 归一化, 防止多次迭代后数值下溢
            max_p[state] = max_p[state] / np.sum(max_p[state])
        # 结束边界
        for i, t in enumerate(self.tags2idx):
            max_p[-1,
                  i] = max_p[-1, i] * self.transition[self.tags2idx[t],
                                                      self.tags2idx['<eos>']]
        # 倒序搜索
        best_tag = np.argmax(max_p[-1])
        best_seq.append(best_tag)
        for state in range(len(words_list), 1, -1):
            pre_tag = path[state - 1, best_tag]
            best_seq.append(pre_tag)
        best_seq.reverse()
        best_seq = [self.idx2tags[i] for i in best_seq]
        return best_seq


if __name__ == "__main__":
    cfg = Config()
    data = read_data(cfg.train_file)
    dev_data = read_data(cfg.dev_file)
    hmm = HMM(data)
    hmm.statistic_parameters(cfg.alpha)
    precision, recall, f1 = evalution(dev_data, hmm)
    print('%s:%f, %s:%f, %s:%f' %
          ('precision', precision, 'recall', recall, 'f1', f1))
    # precision:0.693919, recall:0.693919, f1:0.693919
