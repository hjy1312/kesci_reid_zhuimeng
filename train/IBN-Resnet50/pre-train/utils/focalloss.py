# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 22:09:34 2019

@author: PC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

        self.size_average = size_average

    def forward(self, input, target):
        input_x=input

        target = target.view(-1,1)

        logpt = F.log_softmax(input_x,dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()


        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
