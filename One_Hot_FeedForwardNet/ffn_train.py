# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from config import Config
from train_loader import TrainData, collate_fn
from valid_loader import ValidData
from utils import get_chars, get_labels, show_model_size, time_since
from ffn import FFN
import os
import sys
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config.from_json_file('Configs/config.json')

def train(net, words_t, label_t, loss_fn, opt):

    net.zero_grad()
    out = net(words_t)
    loss = loss_fn(out, label_t)
    loss.backward()
    opt.step()

    return loss.item()

def valid(net, words_t, label_t, loss_fn):

    net.eval()
    with torch.no_grad():
        batch_size = words_t.size(0)
        out = net(words_t)
        loss = loss_fn(out, label_t)
        pred = out.argmax(dim=1)
        acc = pred.eq(label_t).float().sum(dim=-1) / batch_size

        return loss.item(), acc.item()


char2idx, idx2char = get_chars(config.chars_path)
label2idx, idx2label = get_labels(config.labels_path)
train_data = TrainData(config.train_path, char2idx, label2idx)
valid_data = ValidData(config.valid_path, char2idx, label2idx)
train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

net = FFN(config).to(device)
print(net)
show_model_size(net)

try:
    model_path = os.path.abspath(config.load_model_path)
    net.load_state_dict(torch.load(os.path.join(model_path, '%s_%.8f_lr_%d_hidsize.pt' % (net.name, config.lr, config.hidden_size))))
    opt = optim.Adam(net.parameters(), lr=config.cur_lr)
    print('load pre-train model succeed.')
except Exception as e:
    print(e)
    opt = optim.Adam(net.parameters(), lr=config.lr)
    print('load pre-train model failed.')

loss_fn = nn.CrossEntropyLoss()
lr_sche = lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.9, patience=30, verbose=True, min_lr=1e-6)

best_acc = config.best_acc
print('init acc = %f' % (best_acc))
sd = net.state_dict()

for epoch in range(config.epochs):
    start = time.time()
    train_loss = 0.0
    for i, (words_t, label_t) in enumerate(train_loader):
        words_t, label_t = words_t.to(device), label_t.to(device)
        loss = train(net, words_t, label_t, loss_fn, opt)
        train_loss += loss

        if (i+1) % config.print_every == 0:
            lr_sche.step(train_loss)
            print('Epoch %d, %s, (%d -- %d %%), train loss %.3f' % (epoch, time_since(start, (i+1) / len(train_loader)), i+1, (i+1) * 100 / len(train_loader), train_loss / config.print_every))
            train_loss = 0.0

        if (i+1) % config.valid_every == 0:
            valid_loss, valid_acc = 0.0, 0.0
            for words_t, label_t in valid_loader:
                words_t, label_t = words_t.to(device), label_t.to(device)
                val_l, val_a = valid(net, words_t, label_t, loss_fn)
                valid_loss += val_l
                valid_acc += val_a
            print('Epoch %d, step %d, valid loss %.3f, valid accuracy %.3f' % (epoch, i+1, valid_loss / len(valid_loader), valid_acc / len(valid_loader)))

            if best_acc < (valid_acc / len(valid_loader)):
                best_acc = (valid_acc / len(valid_loader))
                sd = net.state_dict()
                print('best acc = %.3f' % (best_acc))
                model_path = os.path.abspath(config.save_model_path)
                torch.save(sd, os.path.join(model_path, '%s_%.8f_lr_%d_hidsize.pt' % (net.name, config.lr, config.hidden_size)))

