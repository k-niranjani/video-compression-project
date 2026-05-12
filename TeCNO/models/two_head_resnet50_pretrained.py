from argparse import ArgumentParser

import torch
import torch.nn as nn
from torchvision import models


class TwoHeadResNet50Pretrained(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.out_features = getattr(hparams, "out_features", 7)
        self.num_tool_classes = getattr(hparams, "num_tool_classes", 7)
        self.dropout = getattr(hparams, "dropout", 0.5)
        self.freeze_backbone = getattr(hparams, "freeze_backbone", False)
        self.freeze_until_layer4 = getattr(hparams, "freeze_until_layer4", False)

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif self.freeze_until_layer4:
            for name, p in self.backbone.named_parameters():
                p.requires_grad = False
                if name.startswith("layer4"):
                    p.requires_grad = True

        self.feature_dropout = nn.Dropout(self.dropout)
        self.phase_head = nn.Linear(in_features, self.out_features)
        self.tool_head = nn.Linear(in_features, self.num_tool_classes)

    def forward(self, x):
        stem = self.backbone(x)
        stem = self.feature_dropout(stem)

        p_phase = self.phase_head(stem)
        p_tool = self.tool_head(stem)

        return stem, p_phase, p_tool

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("TwoHeadResNet50Pretrained")
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--num_tool_classes", type=int, default=7)
        parser.add_argument("--freeze_backbone", action="store_true")
        parser.add_argument("--freeze_until_layer4", action="store_true")
        return parent_parser