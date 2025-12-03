'''
Author: Shuoyu Chen shuoyuchen.physics@gmail.com
Date: 2025-07-16 11:05:40
LastEditors: Shuoyu Chen shuoyuchen.physics@gmail.com
LastEditTime: 2025-07-27 20:43:33
FilePath: /schen/workspace/WatChMaL/watchmal/loss/uncertainty_loss.py
Description: 
'''
import torch
import torch.nn as nn
import math
class UncertaintyLoss(nn.Module):
    def __init__(self, task_losses: dict,  initial_variances: dict = None):
        super().__init__()
        self.task_losses = nn.ModuleDict(task_losses)
        self.tasks = list(task_losses.keys())
        num_tasks = len(self.tasks)
        if initial_variances is None:
            initial_log_vars = torch.zeros(num_tasks)
        else:
            initial_log_vars_list = []
            for task in self.tasks:
                variance = initial_variances.get(task, 1.0)
                initial_log_vars_list.append(math.log(variance + 1e-8))
            initial_log_vars = torch.tensor(initial_log_vars_list)
        self.log_vars = nn.Parameter(initial_log_vars)

    def forward(self, preds: dict, targets: dict):
        total_loss = 0
        log_dict = {}
        for i, task in enumerate(self.tasks):
            pred = preds[task]
            target = targets[task]
            log_var = self.log_vars[i]
            precision = torch.exp(-log_var)
            base_loss_func = self.task_losses[task]
            base_loss = base_loss_func(pred, target)
            task_loss = precision * base_loss +  log_var
            total_loss += task_loss
            log_dict[f'loss_{task}'] = base_loss
            log_dict[f'variance_{task}'] = torch.exp(log_var)
        return total_loss, log_dict
