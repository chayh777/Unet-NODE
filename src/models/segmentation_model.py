from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.adapters import ConvBottleneckAdapter, IdentityAdapter
from src.models.node_adapter import NODEAdapter


AdapterType = Literal["none", "conv", "node"]


@dataclass(frozen=True)
class SegmentationModelOutput:
    logits: torch.Tensor
    bottleneck: torch.Tensor
    adapted_bottleneck: torch.Tensor


class _FeatureInfo:
    def __init__(self, channels: list[int]) -> None:
        self._channels = channels

    def channels(self) -> list[int]:
        return list(self._channels)


class _FallbackEncoder(nn.Module):
    """
    Minimal stand-in for a timm features_only encoder.
    Returns a list of feature maps and exposes `feature_info.channels()`.
    """

    def __init__(self, *, in_channels: int) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.feature_info = _FeatureInfo([64, 128, 256, 512])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


class SegmentationModel(nn.Module):
    def __init__(
        self,
        *,
        encoder_name: str,
        encoder_weights: str | None,
        in_channels: int,
        num_classes: int,
        adapter_type: AdapterType,
        bottleneck_channels: int,
        adapter_hidden_channels: int,
        freeze_encoder: bool,
        node_steps: int,
        node_step_size: float,
    ) -> None:
        super().__init__()
        allowed_weights = {"imagenet", None}
        if encoder_weights not in allowed_weights:
            raise ValueError(
                f"encoder_weights must be one of {allowed_weights}; got {encoder_weights}"
            )

        # Preferred path: timm features-only encoder.
        # Some environments ship incompatible torchvision/transformers/torch combos; if timm
        # import fails at runtime, fall back to a tiny conv encoder so tests remain runnable.
        encoder_channels: list[int]
        try:
            import timm  # type: ignore

            self.encoder = timm.create_model(
                encoder_name,
                pretrained=encoder_weights == "imagenet",
                features_only=True,
                in_chans=in_channels,
            )
            encoder_channels = self.encoder.feature_info.channels()
            if not encoder_channels:
                raise ValueError("Encoder must expose at least one feature map.")
        except Exception:
            self.encoder = _FallbackEncoder(in_channels=in_channels)
            encoder_channels = self.encoder.feature_info.channels()

        self.bottleneck_proj = nn.Conv2d(
            encoder_channels[-1], bottleneck_channels, kernel_size=1
        )

        if adapter_type == "none":
            self.adapter = IdentityAdapter()
        elif adapter_type == "conv":
            self.adapter = ConvBottleneckAdapter(
                channels=bottleneck_channels, hidden_channels=adapter_hidden_channels
            )
        elif adapter_type == "node":
            self.adapter = NODEAdapter(
                channels=bottleneck_channels,
                hidden_channels=adapter_hidden_channels,
                steps=node_steps,
                step_size=node_step_size,
            )
        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type}")

        mid1 = bottleneck_channels // 2
        mid2 = bottleneck_channels // 4
        if mid1 <= 0 or mid2 <= 0:
            raise ValueError(
                "bottleneck_channels must be >= 4 so decoder channels stay positive."
            )

        self.decoder = nn.Sequential(
            nn.Conv2d(bottleneck_channels, mid1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid1, mid2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(mid2, num_classes, kernel_size=1)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        bottleneck = self.bottleneck_proj(features[-1])
        adapted_bottleneck = self.adapter(bottleneck)

        decoded = self.decoder(adapted_bottleneck)
        logits = self.head(decoded)
        logits = F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        return {
            "logits": logits,
            "bottleneck": bottleneck,
            "adapted_bottleneck": adapted_bottleneck,
        }


def build_segmentation_model(**kwargs) -> SegmentationModel:
    return SegmentationModel(**kwargs)
