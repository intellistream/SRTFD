# -*- coding: utf-8 -*-
"""
Created on Wed May 22 20:19:17 2024

@author: Zhao Dandan
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score


def conf_matrix(y_pred, y_true, name, normalize=True, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true.cpu(), y_pred.cpu())
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_percentage, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks and label them with the respective list entries
    classes = np.unique(y_true.cpu())
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix with Percentages',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'  # Format for percentage
    thresh = cm_percentage.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm_percentage[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_percentage[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(name, bbox_inches='tight')
