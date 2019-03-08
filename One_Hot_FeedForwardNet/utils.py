# -*- coding: utf-8 -*-
import os
import sys
import collections
import time
import math


def as_minutes(s):

    m = math.floor(s / 60)
    h = math.floor(m / 60)
    s -= m * 60
    m -= h * 60

    return '%dh %dm %ds' % (h, m, s)


def time_since(since, percent):

    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s

    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def show_model_size(model, print_fn=print):

    print_fn('model size of {} -> {}'.format(model.name, sum(p.numel() for p in model.parameters())))


def get_labels(labels_file):

    label2idx = collections.OrderedDict()
    idx = 0
    with open(labels_file, 'r', encoding='utf-8') as f:
        for lin in f:
            if lin[:-1] not in label2idx:
                label2idx[lin[:-1]] = idx
                idx += 1

    print('size of label2idx = ', len(label2idx))
    idx2label = {val: key for key, val in label2idx.items()}
    
    return label2idx, idx2label


def get_chars(chars_file):

    char2idx = collections.OrderedDict()
    idx = 0
    with open(chars_file, 'r', encoding='utf-8') as f:
        for lin in f:
            if lin[:-1] not in char2idx:
                char2idx[lin[:-1]] = idx
                idx += 1
    print('size of char2idx = ', len(char2idx))
    idx2char = {val: key for key, val in char2idx.items()}
    print('size of idx2char = ', len(idx2char))

    return char2idx, idx2char
    

def flatten(items, ignore_types=(str, bytes)):
    for x in items:
        if isinstance(x, collections.Iterable) and not isinstance(x, ignore_types):
            yield from flatten(x)
        else:
            yield x


if __name__ == "__main__":

    get_chars(sys.argv[1])
