import torch
import torch.nn as nn


def build_conv_bottleneck_block(channels: int, hidden_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, hidden_channels, kernel_size=1),
        # Stateless BN avoids running-mean/var drift when the block is used recurrently (NODE).
        nn.BatchNorm2d(hidden_channels, track_running_stats=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_channels, channels, kernel_size=1),
    )


class IdentityAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ConvBottleneckAdapter(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.net = build_conv_bottleneck_block(channels=channels, hidden_channels=hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

