import torch
import torch.nn as nn

from .adapters import AdapterInit, build_conv_bottleneck_block


class ODEFunction(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        init: AdapterInit = "default",
    ) -> None:
        super().__init__()
        self.net = build_conv_bottleneck_block(
            channels=channels,
            hidden_channels=hidden_channels,
            init=init,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NODEAdapter(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        steps: int,
        step_size: float,
        init: AdapterInit = "default",
        solver: str = "euler",
    ) -> None:
        super().__init__()
        if solver not in ("euler", "rk4"):
            raise ValueError(f"solver must be 'euler' or 'rk4', got {solver!r}")
        self.func = ODEFunction(
            channels=channels,
            hidden_channels=hidden_channels,
            init=init,
        )
        self.steps = steps
        self.step_size = step_size
        self.solver = solver

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.solver == "rk4":
            return self._forward_rk4(x)
        return self._forward_euler(x)

    def _forward_euler(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        for _ in range(self.steps):
            z = z + self.step_size * self.func(z)
        return z

    def _forward_rk4(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        h = self.step_size
        for _ in range(self.steps):
            k1 = self.func(z)
            k2 = self.func(z + 0.5 * h * k1)
            k3 = self.func(z + 0.5 * h * k2)
            k4 = self.func(z + h * k3)
            z = z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return z

