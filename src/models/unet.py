from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUNet(nn.Module):
    """
    U-Net-like wrapper using a timm encoder and skip-concatenation decoder.
    Provides a fixed 1024-channel bottleneck projection for visualization.
    """

    class Output(NamedTuple):
        logits: torch.Tensor
        bottleneck: torch.Tensor

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        num_classes: int = 2,
    ):
        super().__init__()
        try:
            import timm
        except ModuleNotFoundError as exc:
            raise ImportError(
                "SimpleUNet depends on timm. Please install it (pip install timm)."
            ) from exc

        allowed_weights = {"imagenet", None}
        if encoder_weights not in allowed_weights:
            raise ValueError(
                f"encoder_weights must be one of {allowed_weights}; got {encoder_weights}"
            )

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=encoder_weights == "imagenet",
            features_only=True,
            in_chans=in_channels,
        )

        encoder_channels = self.encoder.feature_info.channels()
        if len(encoder_channels) < 2:
            raise ValueError("Encoder must expose at least one skip feature map.")

        self.skip_channels = encoder_channels[:-1]
        bottleneck_in = encoder_channels[-1]

        self.bottleneck_proj = nn.Conv2d(bottleneck_in, 1024, kernel_size=1)

        decoder_blocks: list[nn.Module] = []
        prev_ch = 1024
        for skip_ch in reversed(self.skip_channels):
            block = nn.Sequential(
                nn.Conv2d(prev_ch + skip_ch, skip_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(skip_ch, skip_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            decoder_blocks.append(block)
            prev_ch = skip_ch

        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.output_head = nn.Conv2d(prev_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> "SimpleUNet.Output":
        """
        Returns segmentation logits and the fixed 1024-channel bottleneck tensor.
        """
        features = self.encoder(x)
        bottleneck = self.bottleneck_proj(features[-1])
        current = bottleneck

        skip_features = features[:-1]
        if len(skip_features) != len(self.decoder_blocks):
            raise ValueError("Encoder skip features count mismatch with decoder blocks.")

        for skip_feature, block in zip(reversed(skip_features), self.decoder_blocks):
            current = F.interpolate(
                current, size=skip_feature.shape[-2:], mode="bilinear", align_corners=False
            )
            current = torch.cat([current, skip_feature], dim=1)
            current = block(current)

        logits = self.output_head(
            F.interpolate(current, size=x.shape[-2:], mode="bilinear", align_corners=False)
        )

        return SimpleUNet.Output(logits=logits, bottleneck=bottleneck)
