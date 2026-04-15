import torch
import torch.nn as nn


class ODEFunction(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

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

