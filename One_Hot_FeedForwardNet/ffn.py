# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class FFN(nn.Module):
    """Feed-Forward Net"""
    def __init__(self, args):
        super().__init__()

        self.name = "FFN"
        self.chars_size = args.chars_size
        self.hid_fn = Linear(args.chars_size, args.hidden_size)
        self.out_fn = Linear(args.hidden_size, args.output_size)

    def forward(self, words_t):
        device = words_t.device
        batch_size, maxlen = words_t.size(0), words_t.size(1)
        one_hot = words_t.view(batch_size, maxlen, 1)
        one_hot = torch.zeros(batch_size, maxlen, self.chars_size).to(device).scatter_(2, one_hot, 1)
        one_hot = one_hot.sum(dim=1)
        hid = self.hid_fn(one_hot)
        out = self.out_fn(hid)

        return out


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)

    return m
