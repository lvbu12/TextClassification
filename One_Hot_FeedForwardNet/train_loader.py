# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from utils import get_chars, get_labels, flatten


class TrainData(Dataset):

    def __init__(self, filepath, char2idx, label2idx=None):

        self.data = open(filepath, 'r', encoding='utf-8').readlines()
        self.char2idx = char2idx
        self.label2idx = label2idx

    def __getitem__(self, index):
        
        *sents, label = self.data[index][:-1].split('\t')
        words_idx = [[self.char2idx[ch] for ch in sent] + [self.char2idx['SEP']] for sent in sents if len(sent) > 0]
        words_idx = list(flatten(words_idx))
        label_idx = self.label2idx[label]

        return words_idx, label_idx

    def __len__(self):
        return len(self.data)


def collate_fn(data):

    words_idx = [d[0] for d in data]
    label_idx = [d[1] for d in data]

    words_ts = [torch.tensor(idx, dtype=torch.long) for idx in words_idx]
    words_ts = [F.pad(t, pad=(0, 2100-t.size(0))).view(1, -1) for t in words_ts]
    words_t = torch.cat(words_ts, dim=0)

    label_t = torch.tensor(label_idx, dtype=torch.long)

    return words_t, label_t


if __name__ == "__main__":

    char2idx, idx2char = get_chars('corpus/chars.lst')
    label2idx, idx2label = get_chars('corpus/labels.lst')
    data = TrainData('data/train.txt', char2idx, label2idx)

#    maxlen = 0
#    for i in range(len(data)):
#        words_idx, label_idx = data[i]
#        if(maxlen < len(words_idx)):
#            maxlen = len(words_idx)
#    print('maxlen of lin = ', maxlen)
    loader = DataLoader(data, batch_size=4, shuffle=False, collate_fn=collate_fn)
    for words_t, label_t in loader:
        print(words_t)
        print(label_t)
        print(words_t.size())
        print(label_t.size())
        break
