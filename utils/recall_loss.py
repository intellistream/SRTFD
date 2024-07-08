import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RecallLoss(nn.Module):
    """ An unofficial implementation of
        <Recall Loss for Imbalanced Image Classification and Semantic Segmentation>
        Created by: Zhang Shuai
        Email: shuaizzz666@gmail.com
        recall = TP / (TP + FN)
    Args:
        weight: An array of shape [C,]
        predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
        target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
    Return:
        diceloss
    """
    def __init__(self, weight=None):
        super(RecallLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight) # Normalized weight
        self.smooth = 1e-5

    def forward(self, input, target):
        N, C = input.size()[:2]
        _, predict = torch.max(input, 1)# # (N, C, *) ==> (N, 1, *)

        predict = predict.view(N, 1, -1) # (N, 1, *)
        target = target.view(N, 1, -1) # (N, 1, *)
        last_size = target.size(-1)

        ## convert predict & target (N, 1, *) into one hot vector (N, C, *)
        predict_onehot = torch.zeros((N, C, last_size)) # (N, 1, *) ==> (N, C, *)
        predict_onehot.scatter_(1, predict, 1) # (N, C, *)
        target_onehot = torch.zeros((N, C, last_size))# (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, target, 1) # (N, C, *)

        true_positive = torch.sum(predict_onehot * target_onehot, dim=2)  # (N, C)
        total_target = torch.sum(target_onehot, dim=2)  # (N, C)
        ## Recall = TP / (TP + FN)
        recall = (true_positive + self.smooth) / (total_target + self.smooth)  # (N, C)

        if hasattr(self, 'weight'):
            if self.weight.type() != input.type():
                self.weight = self.weight.type_as(input)
                recall = recall * self.weight * C  # (N, C)
        recall_loss = 1 - torch.mean(recall)  # 1

        return recall_loss
    
def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    #loss = (1 - p) ** gamma * input_values
    loss = (1- p) ** gamma * input_values * 10
    return loss.mean()
class FocalLoss(nn.Module):
    
    def __init__(self, weight=None, gamma=2):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none'), self.gamma)