from __future__ import annotations

from typing import Literal, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import AdapterInit, ConvBottleneckAdapter, IdentityAdapter
from .node_adapter import NODEAdapter
from .unet_decoder import StandardUNetDecoder


AdapterType = Literal["none", "conv", "node"]
AdapterPlacement = Literal["bottleneck", "output"]
ModelArchitecture = Literal["legacy_no_skip", "standard_unet"]


class SegmentationModelOutput(NamedTuple):
    logits: torch.Tensor
    bottleneck: torch.Tensor
    adapted_bottleneck: torch.Tensor
    output_adapter_activation: torch.Tensor | None = None
    node_diagnostics: dict[str, list[torch.Tensor]] | None = None


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


class _TimmDependencyError(RuntimeError):
    """Raised when timm cannot be imported due to dependency/environment issues."""

    def __init__(self, message: str, *, cause: BaseException) -> None:
        super().__init__(message)
        self.__cause__ = cause


def _is_timm_dependency_error(exc: BaseException) -> bool:
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return True
    # Common "version mismatch" style failures during import in this environment.
    if isinstance(exc, AttributeError):
        msg = str(exc)
        if "torch.utils._pytree" in msg or "register_pytree_node" in msg:
            return True
    return False


def _build_timm_encoder(
    *,
    encoder_name: str,
    encoder_weights: str | None,
    in_channels: int,
) -> tuple[nn.Module, list[int]]:
    """
    Build a timm encoder in features_only mode.

    Only raises _TimmDependencyError for import/dependency problems.
    Any create_model/config errors propagate as-is.
    """
    try:
        import timm  # type: ignore
    except Exception as exc:  # noqa: BLE001 - we reclassify known dependency failures
        if _is_timm_dependency_error(exc):
            raise _TimmDependencyError(
                "Failed to import timm due to dependency/environment mismatch.", cause=exc
            ) from exc
        raise

    encoder = timm.create_model(
        encoder_name,
        pretrained=encoder_weights == "imagenet",
        features_only=True,
        in_chans=in_channels,
    )
    encoder_channels = encoder.feature_info.channels()
    if not encoder_channels:
        raise ValueError("Encoder must expose at least one feature map.")
    return encoder, encoder_channels


class SegmentationModel(nn.Module):
    Output = SegmentationModelOutput

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
        adapter_placement: AdapterPlacement = "bottleneck",
        node_solver: str = "euler",
        adapter_init: AdapterInit = "default",
        architecture: ModelArchitecture = "standard_unet",
    ) -> None:
        super().__init__()
        allowed_weights = {"imagenet", None}
        if encoder_weights not in allowed_weights:
            raise ValueError(
                f"encoder_weights must be one of {allowed_weights}; got {encoder_weights}"
            )
        allowed_adapter_placements = {"bottleneck", "output"}
        if adapter_placement not in allowed_adapter_placements:
            raise ValueError(
                "adapter_placement must be one of "
                f"{allowed_adapter_placements}; got {adapter_placement}"
            )
        allowed_architectures = {"legacy_no_skip", "standard_unet"}
        if architecture not in allowed_architectures:
            raise ValueError(
                f"architecture must be one of {allowed_architectures}; got {architecture}"
            )
        self.architecture = architecture

        try:
            encoder, encoder_channels = _build_timm_encoder(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
            )
            self.encoder = encoder
        except _TimmDependencyError:
            self.encoder = _FallbackEncoder(in_channels=in_channels)
            encoder_channels = self.encoder.feature_info.channels()

        self.bottleneck_proj = nn.Conv2d(
            encoder_channels[-1], bottleneck_channels, kernel_size=1
        )

        mid1 = bottleneck_channels // 2
        mid2 = bottleneck_channels // 4
        if mid1 <= 0 or mid2 <= 0:
            raise ValueError(
                "bottleneck_channels must be >= 4 so decoder channels stay positive."
            )

        if adapter_placement == "bottleneck":
            self.adapter = self._build_adapter(
                adapter_type=adapter_type,
                channels=bottleneck_channels,
                hidden_channels=adapter_hidden_channels,
                node_steps=node_steps,
                node_step_size=node_step_size,
                node_solver=node_solver,
                adapter_init=adapter_init,
            )
            self.output_adapter = IdentityAdapter()
        else:
            self.adapter = IdentityAdapter()
            self.output_adapter = self._build_adapter(
                adapter_type=adapter_type,
                channels=encoder_channels[0]
                if architecture == "standard_unet"
                else mid2,
                hidden_channels=adapter_hidden_channels,
                node_steps=node_steps,
                node_step_size=node_step_size,
                node_solver=node_solver,
                adapter_init=adapter_init,
            )

        if architecture == "legacy_no_skip":
            decoder_output_channels = mid2
            self.decoder = nn.Sequential(
                nn.Conv2d(bottleneck_channels, mid1, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid1, mid2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        else:
            decoder_output_channels = encoder_channels[0]
            self.decoder = StandardUNetDecoder(
                encoder_channels=encoder_channels,
                bottleneck_channels=bottleneck_channels,
                output_channels=decoder_output_channels,
            )
        self.head = nn.Conv2d(decoder_output_channels, num_classes, kernel_size=1)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _build_adapter(
        self,
        *,
        adapter_type: AdapterType,
        channels: int,
        hidden_channels: int,
        node_steps: int,
        node_step_size: float,
        node_solver: str,
        adapter_init: AdapterInit,
    ) -> nn.Module:
        if adapter_type == "none":
            return IdentityAdapter()
        if adapter_type == "conv":
            return ConvBottleneckAdapter(
                channels=channels,
                hidden_channels=hidden_channels,
                init=adapter_init,
            )
        if adapter_type == "node":
            return NODEAdapter(
                channels=channels,
                hidden_channels=hidden_channels,
                steps=node_steps,
                step_size=node_step_size,
                init=adapter_init,
                solver=node_solver,
            )
        raise ValueError(f"Unknown adapter_type: {adapter_type}")

    def forward(self, x: torch.Tensor) -> "SegmentationModel.Output":
        features = self.encoder(x)
        bottleneck = self.bottleneck_proj(features[-1])
        adapted_bottleneck = self.adapter(bottleneck)

        if self.architecture == "legacy_no_skip":
            decoded = self.decoder(adapted_bottleneck)
        else:
            decoded = self.decoder(
                adapted_bottleneck,
                skip_features=list(features[:-1])[::-1],
            )
        decoded = self.output_adapter(decoded)
        output_adapter_activation = (
            decoded if not isinstance(self.output_adapter, IdentityAdapter) else None
        )
        node_diagnostics = None
        if hasattr(self.adapter, "last_diagnostics"):
            node_diagnostics = getattr(self.adapter, "last_diagnostics")
        if node_diagnostics is None and hasattr(self.output_adapter, "last_diagnostics"):
            node_diagnostics = getattr(self.output_adapter, "last_diagnostics")
        logits = self.head(decoded)
        logits = F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        return SegmentationModel.Output(
            logits=logits,
            bottleneck=bottleneck,
            adapted_bottleneck=adapted_bottleneck,
            output_adapter_activation=output_adapter_activation,
            node_diagnostics=node_diagnostics,
        )


def build_segmentation_model(**kwargs) -> SegmentationModel:
    return SegmentationModel(**kwargs)
