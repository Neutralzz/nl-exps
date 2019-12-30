from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class HintonLoss(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(HintonLoss, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=-1)
        p_t = F.softmax(y_t/self.T, dim=-1)
        loss = - (p_t*p_s).mean()
        return loss
