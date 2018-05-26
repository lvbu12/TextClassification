# -*- coding:utf-8 -*-
import os
import sys
import pickle
import jieba

BasePath = sys.path[0]

class NaiveBayes(object):
    
    def __init__(self, dir_name):
        self.corpus_path = dir_name
        self.labels = {}
        self.word2idx = {"UNK": 0}
        self.label_prob = {}
        self.emit_prob = {}
        self.line_size = 0.2

    def get_labels(self):
        dir_lst = os.listdir(self.corpus_path)
        for tag in dir_lst:
            self.labels[tag] = len(self.labels)
        print(self.labels)

    def get_word2idx(self, load=False, path=''):
        if load == True:
            with open(path, 'rb') as f:
                self.word2idx = pickle.load(f)
        else:
            dir_lst = os.listdir(self.corpus_path)
            for dir_ in dir_lst:
                dir_path = os.path.join(self.corpus_path, dir_)
                print('Process lines of ', dir_path)
                file_lst = os.listdir(dir_path)
                for file_ in file_lst:
                    file_path = os.path.join(dir_path, file_)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line in lines:
                            lin_seg = line.strip().split()
                            for word in lin_seg:
                                if word not in self.word2idx:
                                    self.word2idx[word] = len(self.word2idx)
                                else:
                                    continue
            
            with open(path, 'wb') as f:
                pickle.dump(self.word2idx, f)

    def get_label_prob(self, load=False, path=''):
        if load == True:
            with open(path, 'rb') as f:
                self.label_prob = pickle.load(f)
        else:
            dir_lst = os.listdir(self.corpus_path)
            tag_dict = {}
            total_file = 0.0
            for dir_ in dir_lst:
                dir_path = os.path.join(self.corpus_path, dir_)
                file_lst = os.listdir(dir_path)
                tag_dict[dir_] = len(file_lst)
                total_file += len(file_lst)
            for key, value in tag_dict.items():
                self.label_prob[key] = value / total_file
            with open(path, 'wb') as f:
                pickle.dump(self.label_prob, f)

    def get_emit_prob(self, load=False, path=''):
        if load == True:
            with open(path, 'rb') as f:
                self.emit_prob = pickle.load(f)
        else:
            emit_prob = {}
            emit_cnt = {}
            for tag, idx in self.labels.items():
                tag_dict = {}
                for word, idx_ in self.word2idx.items():
                    tag_dict[word] = 1.0
                emit_prob[tag] = tag_dict
                emit_cnt[tag] = len(self.word2idx)
            dir_lst = os.listdir(self.corpus_path)
            for dir_ in dir_lst:
                dir_path = os.path.join(self.corpus_path, dir_)
                file_lst = os.listdir(dir_path)
                for i, file_ in enumerate(file_lst):
                    if i % 10000 == 0:
                        print('Process {}th file of {}'.format(i, dir_))
                    file_path = os.path.join(dir_path, file_)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for lin in lines:
                            lin_seg = lin.strip().split()
                            emit_cnt[dir_] += len(lin_seg)
                            for word in lin_seg:
                                emit_prob[dir_][word] += 1.0
            bigger = 1.0
            for tag, total_cnt in emit_cnt.items():
                if total_cnt * 0.0001 > bigger:
                    bigger = total_cnt * 0.0001 
                else:
                    continue
            print('bigger -> ', bigger)
            for tag, emit_dict in emit_prob.items():
                cnt_lst = [cnt for word, cnt in emit_dict.items()]
                print('min of word cnt -> ', min(cnt_lst))
                print('max of word cnt -> ', max(cnt_lst))
                print('toal word cnt of {} -> {}'.format(tag, emit_cnt[tag]))
                print('average word cnt of {} -> {}'.format(tag, emit_cnt[tag] * 1.0 / len(self.word2idx)))
                prob_dict = {}
                for word, cnt in emit_dict.items():
                    prob_dict[word] = cnt * bigger / emit_cnt[tag]
                self.emit_prob[tag] = prob_dict
            with open(path, 'wb') as f:
                pickle.dump(self.emit_prob, f)

    def predict(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = lines[:int(self.line_size * len(lines))+1]
        pred_dict = {}
        for tag in self.labels:
            tag_score = self.label_prob[tag]
            for lin in lines:
                lin_seg = lin.strip().split()
                for word in lin_seg:
                    try:
                        tag_score *= self.emit_prob[tag][word]
                    except:
                        tag_score *= self.emit_prob[tag]['UNK']
            pred_dict[tag] = tag_score
        max_prob, max_tag = -1.0, None
        for tag, prob in pred_dict.items():
#            print('prob of {} -> {}'.format(tag, prob))
            if prob > max_prob:
                max_prob = prob
                max_tag = tag
            else:
                continue
        return max_tag

    def test(self, corpus_path):
        corpus_path = os.path.abspath(corpus_path)
        dir_lst = os.listdir(corpus_path)
        total, right = 0.0, 0.0
        for dir_ in dir_lst:
            dir_path = os.path.join(corpus_path, dir_)
            file_lst = os.listdir(dir_path)
            total += len(file_lst)
            for file_ in file_lst:
                file_path = os.path.join(dir_path, file_)
                if dir_ == self.predict(file_path):
                    right += 1.0
                else:
                    continue
            print('total {} files, right {} files, accuracy of model -> {} %'.format(total, right, right / total * 100))

nba = NaiveBayes(os.path.join(BasePath, 'seg_data'))
nba.get_labels()
nba.get_word2idx(load=True, path='./model/word2idx.pkl')
print('size of char2idx -> ', len(nba.word2idx))
nba.get_label_prob(load=True, path='./model/label_prob.pkl')
print(nba.label_prob)
nba.get_emit_prob(load=True, path='./model/emit_prob.pkl')

nba.test('seg_data')
