import torch
import torch.nn as nn
import cv_config as cfg
import segmentation_models_pytorch as smp
import timm
from classification_head import MLPFusionHead


class EncoderOnlyBranch(nn.Module):

    def __init__(
        self, encoder_name: str, in_channels: int = 3, pretrain: str = "imagenet"
    ):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=in_channels, depth=5, weights=pretrain
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    @property
    def out_channels(self) -> int:
        return int(self.encoder.out_channels[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.flatten(self.gap(features[-1]))


class TimmEncoderOnlyBranch(nn.Module):

    def __init__(
        self, encoder_name: str, in_channels: int = 3, pretrain: str | None = "imagenet"
    ):
        super().__init__()
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


def build_encoder_only_branch(
    encoder_name: str, in_channels: int = 3, pretrain: str | None = "imagenet"
) -> nn.Module:
    if encoder_name in {"convnext_tiny", "swin_tiny_patch4_window7_224"}:
        return TimmEncoderOnlyBranch(
            encoder_name=encoder_name, in_channels=in_channels, pretrain=pretrain
        )
    return EncoderOnlyBranch(
        encoder_name=encoder_name, in_channels=in_channels, pretrain=pretrain
    )


class SingleEncoderOnlyModel(nn.Module):

    def __init__(
        self,
        modality: str,
        encoder_name: str,
        num_classes: int = 3,
        pretrain: str = "imagenet",
    ):
        super().__init__()
        self.modality = modality
        self.branch = EncoderOnlyBranch(
            encoder_name=encoder_name, in_channels=3, pretrain=pretrain
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


class FlexibleMultimodalModel(nn.Module):

    def __init__(
        self,
        modalities: tuple[str, ...],
        backbone_by_modality: dict[str, str],
        num_classes: int = 3,
        pretrain: str = "imagenet",
        fusion_hidden_dim: int = 1024,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        self.branches = nn.ModuleDict(
            {
                modality: build_encoder_only_branch(
                    backbone_by_modality[modality], in_channels=3, pretrain=pretrain
                )
                for modality in self.modalities
            }
        )
        fusion_input_dim = sum(
            (int(self.branches[m].out_channels) for m in self.modalities)
        )
        self.fusion_head = MLPFusionHead(
            input_dim=fusion_input_dim,
            hidden_dim=fusion_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = []
        for idx, modality in enumerate(self.modalities):
            xi = x[:, idx * 3 : (idx + 1) * 3, :, :]
            feat = self.branches[modality](xi)
            features.append(feat)
        fused = torch.cat(features, dim=1)
        return {"cls_pred": self.fusion_head(fused), "feature_fusion": fused}


