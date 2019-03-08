# -*- coding: utf-8 -*-
import os
import sys
from utils import get_labels
import collections


def gen_txt_report(test_file, pred_file, report_file=None):

    f_test = open(test_file, 'r', encoding='utf-8')
    f_pred = open(pred_file, 'r', encoding='utf-8')
    with open(report_file, 'w', encoding='utf-8') as f:
        for (test_line, pred_line) in zip(f_test, f_pred):
            *test_s, test_label = test_line[:-1].split('\t')
            *pred_s, pred_label = pred_line[:-1].split('\t')
            if(test_label == pred_label):
                continue
            f.write('\t'.join(test_s).strip() + '\n')
            f.write(test_label + '\n')
            f.write('\t'.join(pred_s).strip() + '\n')
            f.write(pred_label + '\n\n')

def gen_csv_report(test_file, pred_file, report_file=None):

    label2idx, idx2char = get_labels('corpus/labels.lst')
    
    csv_dict = collections.OrderedDict()
    for key1 in label2idx:
        csv_dict[key1.strip()] = collections.OrderedDict()
        for key2 in label2idx:
            csv_dict[key1.strip()][key2.strip()] = 0
    #print(csv_dict)

    f_test = open(test_file, 'r', encoding='utf-8')
    f_pred = open(pred_file, 'r', encoding='utf-8')
    for (test_line, pred_line) in zip(f_test, f_pred):
        *test_s, test_label = test_line[:-1].split('\t')
        *pred_s, pred_label = pred_line[:-1].split('\t')
        csv_dict[test_label][pred_label] += 1

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(' ')
        for key in label2idx:
            f.write(',' + key)
        f.write('\n')
        for key in label2idx:
            f.write(key)
            for k in label2idx:
                f.write(',' + str(csv_dict[key][k]))
            f.write('\n')


def get_confusion_matrix(report_csv):

    confusion_matrix = collections.OrderedDict()

    with open(report_csv, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = lines[0][:-1].split(',')[1:]
        for lin in lines[1:]:
            lin_seg = lin[:-1].split(',')
            confusion_matrix[lin_seg[0]] = collections.OrderedDict()
            for i, label in enumerate(labels):
                confusion_matrix[lin_seg[0]][label] = int(lin_seg[i+1])

    return confusion_matrix


def compute_macro_F1(report_csv):

    confusion_matrix = get_confusion_matrix(report_csv)

    F1_dict = collections.OrderedDict()
    for key in confusion_matrix:
        TP = confusion_matrix[key][key]
        FNs = [p[1] for p in confusion_matrix[key].items()]
        FN = sum(FNs) - TP
        FPs = [confusion_matrix[p][key] for p in confusion_matrix]
        FP = sum(FPs) - TP
        print(key, TP, FN, FP)
        try:
            precision = TP / (TP + FP)
        except:
            precision = 0
        try:
            recall = TP / (TP + FN)
        except:
            recall = 0
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except:
            f1 = 0
        F1_dict[key] = f1
    for key in F1_dict:
        print('F1 score of {} = {:.3f}.'.format(key, F1_dict[key]))
    print('Micro_F1 = {:.3f}'.format(sum([F1_dict[k] for k in F1_dict]) / len(F1_dict)))

def compute_micro_F1(report_csv):

    confusion_matrix = get_confusion_matrix(report_csv)

    TP_dict = collections.OrderedDict()
    FN_dict = collections.OrderedDict()
    FP_dict = collections.OrderedDict()
    for key in confusion_matrix:
        TP = confusion_matrix[key][key]
        TP_dict[key] = TP
        FNs = [p[1] for p in confusion_matrix[key].items()]
        FN = sum(FNs) - TP
        FN_dict[key] = FN
        FPs = [confusion_matrix[p][key] for p in confusion_matrix]
        FP = sum(FPs) - TP
        FP_dict[key] = FP
        print(key, TP, FN, FP)
    TP = sum([TP_dict[k] for k in TP_dict])
    FN = sum([FN_dict[k] for k in FN_dict])
    FP = sum([FP_dict[k] for k in FP_dict])
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    print('Macro_F1 = {:.3f}'.format(f1))

if __name__ == "__main__":

    #gen_csv_report(sys.argv[1], sys.argv[2], sys.argv[3])
    compute_micro_F1(sys.argv[1])
    compute_macro_F1(sys.argv[1])
