'''
Author: Shuoyu Chen shuoyuchen.physics@gmail.com
Date: 2025-07-22 19:45:09
LastEditors: Shuoyu Chen shuoyuchen.physics@gmail.com
LastEditTime: 2025-07-27 21:16:17
FilePath: /schen/workspace/WatChMaL/watchmal/model/SwinTransformer.py
Description: 
'''
"""
Here is a Swin Transformer model.
"""

import torch
import torch.nn as nn
import timm


class SwinRegressor(nn.Module):
    def __init__(
        self,
        model_name="swin_tiny_patch4_window7_224",
        pretrained=False,
        img_size=(192, 192),
        in_chans=2,
        num_output_channels=3,
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_output_channels,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
        )
        self.output_dim = num_output_channels

    def forward(self, x):
        out = self.vit(x)
        return out



class MultiTaskSwin(nn.Module):
    def __init__(
        self,
        model_name="swin_tiny_patch4_window7_224",
        pretrained=False,
        img_size=(192, 192),
        in_chans=2,
        drop_path_rate=0.0,
        task_output_dims={'positions': 3, 'directions': 3, 'energies': 1}
    ):
        super().__init__()
        
        self.task_names = list(task_output_dims.keys())
        self.task_dims = list(task_output_dims.values())
        total_output_channels = sum(self.task_dims)
        self.regressor = SwinRegressor(
            model_name=model_name,
            pretrained=pretrained,
            img_size=img_size,
            in_chans=in_chans,
            num_output_channels=total_output_channels, 
            drop_path_rate=drop_path_rate
        )
    def forward(self, x):
        combined_output = self.regressor(x)
        split_outputs = torch.split(combined_output, self.task_dims, dim=1)
        outputs = {
            task_name: tensor
            for task_name, tensor in zip(self.task_names, split_outputs)
        }
        return outputs
    
