import torch
import torch.nn as nn
from typing import Literal


AdapterInit = Literal["default", "zero_last_layer"]


def _zero_init_final_conv(block: nn.Sequential) -> None:
    final_conv = next((m for m in reversed(block) if isinstance(m, nn.Conv2d)), None)
    if final_conv is None:
        raise ValueError("Cannot zero-initialize final Conv2d: block has no Conv2d layers")

    nn.init.zeros_(final_conv.weight)
    if final_conv.bias is not None:
        nn.init.zeros_(final_conv.bias)


def build_conv_bottleneck_block(
    channels: int,
    hidden_channels: int,
    init: AdapterInit = "default",
) -> nn.Sequential:
    block = nn.Sequential(
        nn.Conv2d(channels, hidden_channels, kernel_size=1),
        # Stateless BN avoids running-mean/var drift when the block is used recurrently (NODE).
        nn.BatchNorm2d(hidden_channels, track_running_stats=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_channels, channels, kernel_size=1),
    )

    if init == "zero_last_layer":
        _zero_init_final_conv(block)
    elif init != "default":
        raise ValueError(f"Unsupported adapter init: {init!r}")

    return block


class IdentityAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ConvBottleneckAdapter(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, init: AdapterInit = "default") -> None:
        super().__init__()
        self.net = build_conv_bottleneck_block(
            channels=channels,
            hidden_channels=hidden_channels,
            init=init,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

