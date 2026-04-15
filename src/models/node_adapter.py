import torch
import torch.nn as nn

from .adapters import build_conv_bottleneck_block


class ODEFunction(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.net = build_conv_bottleneck_block(channels=channels, hidden_channels=hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NODEAdapter(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, steps: int, step_size: float) -> None:
        super().__init__()
        self.func = ODEFunction(channels=channels, hidden_channels=hidden_channels)
        self.steps = steps
        self.step_size = step_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        for _ in range(self.steps):
            z = z + self.step_size * self.func(z)
        return z

