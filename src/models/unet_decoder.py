from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetUpBlock(nn.Module):
    def __init__(self, *, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class StandardUNetDecoder(nn.Module):
    def __init__(
        self,
        *,
        encoder_channels: list[int],
        bottleneck_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()
        if len(encoder_channels) < 2:
            raise ValueError("StandardUNetDecoder requires at least two encoder stages.")

        skip_channels = list(encoder_channels[:-1])[::-1]
        if not skip_channels:
            raise ValueError("StandardUNetDecoder requires at least one skip feature.")

        block_out_channels: list[int] = []
        current = bottleneck_channels
        for idx in range(len(skip_channels)):
            proposed = max(current // 2, output_channels)
            if idx == len(skip_channels) - 1:
                proposed = output_channels
            block_out_channels.append(proposed)
            current = proposed

        blocks = []
        in_channels = bottleneck_channels
        for skip_ch, out_ch in zip(skip_channels, block_out_channels):
            blocks.append(
                UNetUpBlock(
                    in_channels=in_channels,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                )
            )
            in_channels = out_ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, bottleneck: torch.Tensor, *, skip_features: list[torch.Tensor]) -> torch.Tensor:
        if len(skip_features) != len(self.blocks):
            raise ValueError(
                f"Expected {len(self.blocks)} skip features, got {len(skip_features)}."
            )
        x = bottleneck
        for block, skip in zip(self.blocks, skip_features):
            x = block(x, skip)
        return x
