# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:31:22 2019

@author: AI
"""
from math import sqrt

from sklearn.metrics import roc_auc_score
import numpy as np

def calculate_performace(num, y_pred, y_prob, y_test):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(num):
        if y_test[index] ==1:
            if y_test[index] == y_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if y_test[index] == y_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    acc = float(tp + tn)/num
    try:
        precision = float(tp)/(tp + fp)
        sensitivity = float(tp)/ (tp + fn)
        specificity = float(tn)/(tn + fp)
    except ZeroDivisionError:
        print("You can't divide by 0.")
        precision=sensitivity=specificity=MCC = 0.0001
    AUC = roc_auc_score(y_test, y_prob)
    balanced_acc = float(sensitivity+specificity)/2
    MCC=float(tp*tn-fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return precision,sensitivity,specificity,acc, balanced_acc, AUC,MCC

def transfer_label_from_prob(proba):
    label = [1 if val>=0.5 else 0 for val in proba]
    return label
'''
rna = ['1','2','3','4','5','6','7','8','9','10','11','12','13']
num_fold=3
for fold in range(num_fold):
    rna_train = np.array([x for i, x in enumerate(rna) if i % num_fold != fold])
print(rna_train)
'''