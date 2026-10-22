from __future__ import annotations
import torch
import torch.nn as nn
import cv_config as cfg
import timm
from cv_models import SingleEncoderOnlyModel
from classification_head import MLPFusionHead

TIMM_BACKBONES = {"convnext_tiny", "swin_tiny"}


class TimmEncoderOnlyBranch(nn.Module):

    def __init__(
        self, encoder_name: str, in_channels: int = 3, pretrain: str | None = "imagenet"
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrain is not None,
            in_chans=in_channels,
            num_classes=0,
            global_pool="avg",
        )
        self.out_channels = int(self.encoder.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TimmSingleEncoderOnlyModel(nn.Module):

    def __init__(
        self,
        modality: str,
        encoder_name: str,
        num_classes: int = 3,
        pretrain: str | None = "imagenet",
    ):
        super().__init__()
        self.modality = modality
        self.branch = TimmEncoderOnlyBranch(
            encoder_name, in_channels=3, pretrain=pretrain
        )
        self.classifier = MLPFusionHead(
            input_dim=self.branch.out_channels,
            hidden_dim=cfg.FUSION_HIDDEN_DIM,
            num_classes=num_classes,
            dropout=0.5,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.branch(x)
        return {"cls_pred": self.classifier(features), "feature_fusion": features}


def build_single_model(
    modality: str,
    backbone: str,
    num_classes: int = 3,
    pretrain: str | None = "imagenet",
) -> nn.Module:
    encoder_name = cfg.BACKBONES[backbone]["encoder_name"]
    if backbone in TIMM_BACKBONES:
        return TimmSingleEncoderOnlyModel(
            modality=modality,
            encoder_name=encoder_name,
            num_classes=num_classes,
            pretrain=pretrain,
        )
    return SingleEncoderOnlyModel(
        modality=modality,
        encoder_name=encoder_name,
        num_classes=num_classes,
        pretrain=pretrain,
    )
