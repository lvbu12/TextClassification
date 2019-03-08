# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from utils import get_chars, flatten


class TestData(Dataset):

    def __init__(self, filepath, char2idx):

        self.data = open(filepath, 'r', encoding='utf-8').readlines()
        self.char2idx = char2idx

    def __getitem__(self, index):
        
        sents = self.data[index][:-1].split('\t')[:-1]
        words_idx = [[self.char2idx[ch] if ch in self.char2idx else self.char2idx['UNK'] for ch in sent] + [self.char2idx['SEP']] for sent in sents if len(sent) > 0]
        words_idx = list(flatten(words_idx))

        return words_idx

    def __len__(self):
        return len(self.data)


def collate_fn(data):

    words_idx = data

    words_ts = [torch.tensor(idx, dtype=torch.long) for idx in words_idx]
    words_ts = [F.pad(t, pad=(0, 2100-t.size(0))).view(1, -1) for t in words_ts]
    words_t = torch.cat(words_ts, dim=0)

    return words_t


if __name__ == "__main__":

    char2idx, idx2char = get_chars('corpus/chars.lst')
    data = TestData('data/test.txt', char2idx)

    loader = DataLoader(data, batch_size=4, shuffle=False, collate_fn=collate_fn)
    for words_t in loader:
        print(words_t)
        print(words_t.size())
        break
