"""
DINO-style self-supervised models built on the existing SwinT backbone.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

from watchmal.model.swintSI import SwinTransformerWithPreStage


class DINOHead(nn.Module):
    """
    Projection/prototype head used only during self-supervised pre-training.
    """

    def __init__(
        self,
        in_dim,
        hidden_dim=2048,
        bottleneck_dim=256,
        out_dim=1024,
        nlayers=3,
        norm_last_layer=True,
    ):
        super().__init__()
        if nlayers < 1:
            raise ValueError("DINOHead requires nlayers >= 1")

        layers = []
        dim = in_dim
        for _ in range(nlayers - 1):
            layers.extend([
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
            ])
            dim = hidden_dim
        layers.append(nn.Linear(dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.norm_last_layer = norm_last_layer
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        if self.norm_last_layer:
            weight = F.normalize(self.last_layer.weight, dim=1)
            return F.linear(x, weight)
        return self.last_layer(x)


class SwinDINO(nn.Module):
    """
    Student/teacher SwinT model for DINO pre-training.
    """

    def __init__(
        self,
        img_size=(192, 192),
        in_chans=2,
        pre_stage_patch_size=2,
        pre_stage_embed_dim=48,
        pre_stage_depth=8,
        pre_stage_num_heads=6,
        pre_stage_window_size=4,
        embed_dim=96,
        depths=(4, 4, 10, 2),
        num_heads=(3, 6, 12, 24),
        window_size=6,
        ape=True,
        dino_hidden_dim=2048,
        dino_bottleneck_dim=256,
        dino_out_dim=1024,
        dino_nlayers=3,
        dino_norm_last_layer=True,
        **backbone_kwargs,
    ):
        super().__init__()
        backbone_config = dict(
            img_size=img_size,
            in_chans=in_chans,
            pre_stage_patch_size=pre_stage_patch_size,
            pre_stage_embed_dim=pre_stage_embed_dim,
            pre_stage_depth=pre_stage_depth,
            pre_stage_num_heads=pre_stage_num_heads,
            pre_stage_window_size=pre_stage_window_size,
            embed_dim=embed_dim,
            depths=list(depths),
            num_heads=list(num_heads),
            window_size=window_size,
            ape=ape,
            **backbone_kwargs,
        )
        self.student_backbone = SwinTransformerWithPreStage(**backbone_config)
        self.student_head = DINOHead(
            in_dim=self.student_backbone.num_features,
            hidden_dim=dino_hidden_dim,
            bottleneck_dim=dino_bottleneck_dim,
            out_dim=dino_out_dim,
            nlayers=dino_nlayers,
            norm_last_layer=dino_norm_last_layer,
        )

        self.teacher_backbone = deepcopy(self.student_backbone)
        self.teacher_head = deepcopy(self.student_head)
        self.freeze_teacher()

    def freeze_teacher(self):
        for parameter in self.teacher_backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.teacher_head.parameters():
            parameter.requires_grad = False

    def forward_student(self, x):
        return self.student_head(self.student_backbone(x))

    @torch.no_grad()
    def forward_teacher(self, x):
        return self.teacher_head(self.teacher_backbone(x))

    @torch.no_grad()
    def update_teacher(self, momentum):
        for student_param, teacher_param in zip(
            self.student_backbone.parameters(), self.teacher_backbone.parameters()
        ):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
        for student_param, teacher_param in zip(
            self.student_head.parameters(), self.teacher_head.parameters()
        ):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)

    def forward(self, x):
        return self.forward_student(x)


class SwinDINOLinearProbe(nn.Module):
    """
    Frozen-backbone probe for checking whether DINO features contain target information.
    """

    def __init__(
        self,
        dino_checkpoint=None,
        freeze_backbone=True,
        num_output_channels=3,
        img_size=(192, 192),
        in_chans=2,
        pre_stage_patch_size=2,
        pre_stage_embed_dim=48,
        pre_stage_depth=8,
        pre_stage_num_heads=6,
        pre_stage_window_size=4,
        embed_dim=96,
        depths=(4, 4, 10, 2),
        num_heads=(3, 6, 12, 24),
        window_size=6,
        ape=True,
        **backbone_kwargs,
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.backbone = SwinTransformerWithPreStage(
            img_size=img_size,
            in_chans=in_chans,
            pre_stage_patch_size=pre_stage_patch_size,
            pre_stage_embed_dim=pre_stage_embed_dim,
            pre_stage_depth=pre_stage_depth,
            pre_stage_num_heads=pre_stage_num_heads,
            pre_stage_window_size=pre_stage_window_size,
            embed_dim=embed_dim,
            depths=list(depths),
            num_heads=list(num_heads),
            window_size=window_size,
            ape=ape,
            **backbone_kwargs,
        )
        if dino_checkpoint is not None:
            self.load_dino_backbone(dino_checkpoint)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.head = nn.Linear(self.backbone.num_features, num_output_channels)
        nn.init.zeros_(self.head.bias)
        self.output_dim = num_output_channels

    def load_dino_backbone(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
        backbone_state = {}
        prefix = "student_backbone."
        for key, value in state_dict.items():
            if key.startswith(prefix):
                backbone_state[key[len(prefix):]] = value
        if not backbone_state:
            raise ValueError(f"No student_backbone weights found in DINO checkpoint: {checkpoint_path}")
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected keys while loading DINO backbone: {unexpected}")
        if missing:
            raise ValueError(f"Missing keys while loading DINO backbone: {missing}")

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        return self.head(features)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self
