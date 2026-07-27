from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .data import SplitSlices


MAD_TO_SIGMA = 1.4826


def _nanmedian(values: Tensor, dim: int) -> Tensor:
    return torch.nanmedian(values, dim=dim).values


def _safe_center_and_scale(
    values: Tensor,
    absolute_floor: float,
    relative_floor: float,
) -> tuple[Tensor, Tensor]:
    center = _nanmedian(values, dim=0)
    center = torch.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0)
    mad = _nanmedian(torch.abs(values - center), dim=0)
    raw_scale = MAD_TO_SIGMA * mad
    minimum = torch.clamp(
        torch.abs(center) * relative_floor, min=absolute_floor
    )
    scale = torch.where(
        torch.isfinite(raw_scale) & (raw_scale > minimum),
        raw_scale,
        minimum,
    )
    return center, scale


@dataclass
class RobustScaler:
    absolute_floor: float = 1e-6
    relative_floor: float = 1e-6
    center: Tensor | None = None
    scale: Tensor | None = None

    def fit(self, values: Tensor) -> "RobustScaler":
        self.center, self.scale = _safe_center_and_scale(
            values, self.absolute_floor, self.relative_floor
        )
        return self

    def transform(self, values: Tensor) -> Tensor:
        if self.center is None or self.scale is None:
            raise RuntimeError("RobustScaler must be fitted before transform.")
        return (values - self.center) / self.scale

    def state_dict(self) -> dict[str, Any]:
        if self.center is None or self.scale is None:
            raise RuntimeError("Cannot serialize an unfitted RobustScaler.")
        return {
            "absolute_floor": self.absolute_floor,
            "relative_floor": self.relative_floor,
            "center": self.center.cpu(),
            "scale": self.scale.cpu(),
        }


class SeasonalMedianMAD:
    def __init__(
        self,
        period: int = 288,
        absolute_floor: float = 1e-6,
        relative_floor: float = 1e-6,
    ) -> None:
        if period < 2:
            raise ValueError("period must be at least two.")
        self.period = period
        self.absolute_floor = absolute_floor
        self.relative_floor = relative_floor
        self.center: Tensor | None = None
        self.scale: Tensor | None = None

    def fit(self, train_values: Tensor, offset: int = 0) -> "SeasonalMedianMAD":
        if train_values.ndim != 2:
            raise ValueError("train_values must have shape [time, series].")
        global_center, global_scale = _safe_center_and_scale(
            train_values, self.absolute_floor, self.relative_floor
        )

        centers: list[Tensor] = []
        scales: list[Tensor] = []
        time_index = torch.arange(train_values.shape[0])
        for phase in range(self.period):
            phase_values = train_values[
                (time_index + offset).remainder(self.period) == phase
            ]
            if phase_values.shape[0] == 0:
                centers.append(global_center)
                scales.append(global_scale)
                continue

            phase_center = _nanmedian(phase_values, dim=0)
            phase_center = torch.where(
                torch.isfinite(phase_center), phase_center, global_center
            )
            phase_mad = _nanmedian(
                torch.abs(phase_values - phase_center), dim=0
            )
            phase_scale = MAD_TO_SIGMA * phase_mad
            phase_scale = torch.where(
                torch.isfinite(phase_scale)
                & (phase_scale > self.absolute_floor),
                phase_scale,
                global_scale,
            )
            minimum = torch.clamp(
                torch.abs(phase_center) * self.relative_floor,
                min=self.absolute_floor,
            )
            phase_scale = torch.maximum(phase_scale, minimum)
            centers.append(phase_center)
            scales.append(phase_scale)

        self.center = torch.stack(centers)
        self.scale = torch.stack(scales)
        return self

    def transform(self, values: Tensor, offset: int = 0) -> Tensor:
        if self.center is None or self.scale is None:
            raise RuntimeError("SeasonalMedianMAD must be fitted before transform.")
        phases = (
            torch.arange(values.shape[0], device=values.device) + offset
        ).remainder(self.period)
        center = self.center.to(values.device)[phases]
        scale = self.scale.to(values.device)[phases]
        return (values - center) / scale

    def state_dict(self) -> dict[str, Any]:
        if self.center is None or self.scale is None:
            raise RuntimeError("Cannot serialize an unfitted baseline.")
        return {
            "name": "seasonal_mad",
            "period": self.period,
            "absolute_floor": self.absolute_floor,
            "relative_floor": self.relative_floor,
            "center": self.center.cpu(),
            "scale": self.scale.cpu(),
        }


class AdditiveHoltWinters:
    """Causal additive Holt-Winters filter with missing-value-safe updates."""

    def __init__(
        self,
        period: int = 288,
        alpha: float = 0.2,
        beta: float = 0.02,
        gamma: float = 0.05,
    ) -> None:
        if period < 2:
            raise ValueError("period must be at least two.")
        for name, value in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        self.period = period
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.level: Tensor | None = None
        self.trend: Tensor | None = None
        self.seasonal: Tensor | None = None

    def initialize(self, values: Tensor) -> None:
        if values.shape[0] < 2 * self.period:
            raise ValueError(
                f"Holt-Winters needs at least {2 * self.period} training rows."
            )
        first = values[: self.period]
        second = values[self.period : 2 * self.period]

        first_level = _nanmedian(first, dim=0)
        overall_level = _nanmedian(values, dim=0)
        first_level = torch.where(
            torch.isfinite(first_level), first_level, overall_level
        )
        first_level = torch.nan_to_num(first_level, nan=0.0)

        seasonal = first - first_level
        seasonal = torch.nan_to_num(seasonal, nan=0.0, posinf=0.0, neginf=0.0)

        paired = torch.isfinite(first) & torch.isfinite(second)
        per_step_change = torch.where(
            paired,
            (second - first) / self.period,
            torch.nan,
        )
        trend = _nanmedian(per_step_change, dim=0)
        trend = torch.nan_to_num(trend, nan=0.0, posinf=0.0, neginf=0.0)

        second_level = _nanmedian(second - seasonal, dim=0)
        second_level = torch.where(
            torch.isfinite(second_level),
            second_level,
            first_level + self.period * trend,
        )
        self.level = second_level
        self.trend = trend
        self.seasonal = seasonal

    def _check_initialized(self) -> tuple[Tensor, Tensor, Tensor]:
        if self.level is None or self.trend is None or self.seasonal is None:
            raise RuntimeError("Holt-Winters has not been initialized.")
        return self.level, self.trend, self.seasonal

    def transform(
        self,
        values: Tensor,
        offset: int,
        update: bool = True,
    ) -> Tensor:
        level, trend, seasonal = self._check_initialized()
        residuals = torch.full_like(values, torch.nan)

        for local_time in range(values.shape[0]):
            phase = (offset + local_time) % self.period
            observation = values[local_time]
            observed = torch.isfinite(observation)
            old_seasonal = seasonal[phase]

            level_prediction = level + trend
            forecast = level_prediction + old_seasonal
            residuals[local_time] = torch.where(
                observed, observation - forecast, torch.nan
            )

            if update:
                updated_level = (
                    self.alpha * (observation - old_seasonal)
                    + (1.0 - self.alpha) * level_prediction
                )
                updated_trend = (
                    self.beta * (updated_level - level)
                    + (1.0 - self.beta) * trend
                )
                updated_seasonal = (
                    self.gamma * (observation - updated_level)
                    + (1.0 - self.gamma) * old_seasonal
                )
                level = torch.where(observed, updated_level, level_prediction)
                trend = torch.where(observed, updated_trend, trend)
                seasonal[phase] = torch.where(
                    observed, updated_seasonal, old_seasonal
                )

        if update:
            self.level = level
            self.trend = trend
            self.seasonal = seasonal
        return residuals

    def fit_transform(self, train_values: Tensor) -> Tensor:
        self.initialize(train_values)
        burn_in = 2 * self.period
        residuals = torch.full_like(train_values, torch.nan)
        residuals[burn_in:] = self.transform(
            train_values[burn_in:], offset=burn_in, update=True
        )
        return residuals

    def state_dict(self) -> dict[str, Any]:
        level, trend, seasonal = self._check_initialized()
        return {
            "name": "holt_winters",
            "period": self.period,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "level": level.cpu(),
            "trend": trend.cpu(),
            "seasonal": seasonal.cpu(),
        }


@dataclass
class BaselineOutput:
    features: Tensor
    state: dict[str, Any]


def build_baseline_features(
    values: Tensor,
    splits: SplitSlices,
    method: str,
    period: int = 288,
    alpha: float = 0.2,
    beta: float = 0.02,
    gamma: float = 0.05,
) -> BaselineOutput:
    train_values = values[splits.train]

    if method == "seasonal_mad":
        baseline = SeasonalMedianMAD(period=period).fit(train_values)
        return BaselineOutput(
            features=baseline.transform(values),
            state=baseline.state_dict(),
        )

    if method == "holt_winters":
        baseline = AdditiveHoltWinters(
            period=period, alpha=alpha, beta=beta, gamma=gamma
        )
        train_residual = baseline.fit_transform(train_values)
        validation_residual = baseline.transform(
            values[splits.validation],
            offset=splits.validation.start,
            update=True,
        )
        test_residual = baseline.transform(
            values[splits.test],
            offset=splits.test.start,
            update=True,
        )
        residuals = torch.cat(
            (train_residual, validation_residual, test_residual), dim=0
        )
        scaler = RobustScaler().fit(train_residual)
        state = baseline.state_dict()
        state["residual_scaler"] = scaler.state_dict()
        return BaselineOutput(
            features=scaler.transform(residuals),
            state=state,
        )

    if method == "none":
        scaler = RobustScaler().fit(train_values)
        return BaselineOutput(
            features=scaler.transform(values),
            state={"name": "none", "scaler": scaler.state_dict()},
        )

    raise ValueError(f"Unknown baseline method: {method}")
