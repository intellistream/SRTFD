# -*- coding: utf-8 -*-
"""
Created on Wed May 22 20:19:17 2024

@author: Zhao Dandan
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score


def conf_matrix(y_pred, y_true, normalize=False, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)

    num1 = len(np.unique(y_true))
    # cm = np.transpose(cm)
    cm1 = cm.astype(np.float64)
    for l in range(num1):
        index = np.where(y_true == l)
        num = len(index[0])
        for h in range(num1):
            cm1[l, h] = cm1[l, h]/num

    #print(cm1)S

   # accuracy  = accuracy_score(y_true, y_pred)

    labels = ['Nor', '1', '2', '3', '4', '5', '6']
    fig, ax = plt.subplots()
    im = ax.imshow(cm1, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    #ax.set_xticklabels(labels)
    #ax.set_yticklabels(labels)

    # 设置 x 轴和 y 轴的标签
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm1.max() / 2.
    for i in range(cm1.shape[0]):
        for j in range(cm1.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.imshow(cm1, interpolation='nearest', cmap=cmap)
