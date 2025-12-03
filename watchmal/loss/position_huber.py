'''
Author: Shuoyu Chen shuoyuchen.physics@gmail.com
Date: 2025-07-28 09:08:59
LastEditors: Shuoyu Chen shuoyuchen.physics@gmail.com
LastEditTime: 2025-07-28 09:14:49
FilePath: /schen/workspace/WatChMaL/watchmal/loss/position_huber.py
Description: 
'''
import torch
import torch.nn as nn

class PositionHuberLoss(nn.Module):
    def __init__(self, delta=1.0, eps=1e-6):
        super().__init__()
        self.delta = delta
        self.eps = eps

    def forward(self, pred, target):
        error_vector = pred - target
        euclidean_distance = torch.linalg.vector_norm(error_vector, dim=-1)
        abs_error = euclidean_distance
        quadratic = torch.minimum(abs_error, torch.tensor(self.delta, device=abs_error.device))
        linear = abs_error - quadratic
        loss_per_sample = 0.5 * quadratic.pow(2) + self.delta * linear
        return loss_per_sample.mean()