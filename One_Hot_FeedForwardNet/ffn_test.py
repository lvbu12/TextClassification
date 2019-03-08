# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from test_loader import TestData, collate_fn
from utils import get_chars, get_labels, show_model_size
from ffn import FFN
import os
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config.from_json_file('Configs/config.json')


def test(net, words_t):

    with torch.no_grad():
        batch_size = words_t.size(0)
        out = net(words_t)
        pred = out.argmax(dim=1)

        return pred


char2idx, idx2char = get_chars(config.chars_path)
label2idx, idx2label = get_labels(config.labels_path)
test_data = TestData(config.test_path, char2idx)
test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

net = FFN(config).to(device)
print(net)
show_model_size(net)

try:
    model_path = os.path.abspath(config.load_model_path)
    net.load_state_dict(torch.load(os.path.join(model_path, '%s_%.8f_lr_%d_hidsize.pt' % (net.name, config.lr, config.hidden_size))))
    print('load pre-train model succeed.')
except Exception as e:
    print(e)
    print('load pre-train model failed.')

predfile_path = os.path.abspath(config.pred_path)
with open(predfile_path, 'w', encoding='utf-8') as f:
    for words_t in test_loader:
        words_t = words_t.to(device)
        batch_size, maxlen = words_t.size()
        pred = test(net, words_t)
        for i in range(batch_size):
            words = []
            for j in range(maxlen):
                if words_t[i, j].item() == char2idx['PAD']:
                    continue
                words.append(idx2char[words_t[i, j].item()])
            f.write(''.join(words))
            f.write('\t' + idx2label[pred[i].item()] + '\n')
