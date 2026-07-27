from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import MetadataFeatures


@dataclass(frozen=True)
class PanelFactorConfig:
    rank: int = 32
    period: int = 288
    dropout: float = 0.05
    smoothness_weight: float = 0.01
    daily_weight: float = 0.01
    residual_loading_weight: float = 0.001
    collective_weight: float = 0.5

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


class MetadataPanelFactor(nn.Module):
    """Robust low-rank model for a short, very wide time-series panel."""

    def __init__(
        self,
        train_length: int,
        series_count: int,
        config: PanelFactorConfig,
        metadata: MetadataFeatures | None = None,
    ) -> None:
        super().__init__()
        if config.rank < 1:
            raise ValueError("rank must be positive.")
        self.config = config
        self.train_length = train_length
        self.series_count = series_count

        self.time_factors = nn.Embedding(train_length, config.rank)
        self.time_bias = nn.Embedding(train_length, 1)
        self.series_residual = nn.Embedding(series_count, config.rank)
        self.series_bias = nn.Embedding(series_count, 1)
        self.global_bias = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(config.dropout)

        self.metadata_embeddings = nn.ModuleList()
        if metadata is not None:
            for cardinality in metadata.cardinalities:
                self.metadata_embeddings.append(
                    nn.Embedding(cardinality, config.rank)
                )
            self.metadata_numeric = nn.Linear(
                metadata.numeric.shape[1], config.rank, bias=False
            )
            self.register_buffer(
                "metadata_categorical",
                metadata.categorical,
                persistent=True,
            )
            self.register_buffer(
                "metadata_values",
                metadata.numeric,
                persistent=True,
            )
        else:
            self.metadata_numeric = None
            self.register_buffer(
                "metadata_categorical", None, persistent=True
            )
            self.register_buffer("metadata_values", None, persistent=True)

        nn.init.normal_(self.time_factors.weight, std=0.05)
        nn.init.normal_(self.series_residual.weight, std=0.02)
        nn.init.zeros_(self.time_bias.weight)
        nn.init.zeros_(self.series_bias.weight)
        for embedding in self.metadata_embeddings:
            nn.init.normal_(embedding.weight, std=0.02)

    def series_loadings(self) -> Tensor:
        loadings = self.series_residual.weight
        if (
            self.metadata_categorical is not None
            and self.metadata_values is not None
            and self.metadata_numeric is not None
        ):
            components = [loadings]
            for column, embedding in enumerate(self.metadata_embeddings):
                components.append(
                    embedding(self.metadata_categorical[:, column])
                )
            components.append(self.metadata_numeric(self.metadata_values))
            loadings = torch.stack(components).mean(dim=0)
        return self.dropout(loadings)

    def reconstruct_train(self, time_indices: Tensor) -> Tensor:
        factors = self.time_factors(time_indices)
        loadings = self.series_loadings()
        return (
            factors @ loadings.transpose(0, 1)
            + self.time_bias(time_indices)
            + self.series_bias.weight.transpose(0, 1)
            + self.global_bias
        )

    def temporal_regularization(self) -> Tensor:
        factors = self.time_factors.weight
        time_bias = self.time_bias.weight
        smoothness = (
            (factors[1:] - factors[:-1]).square().mean()
            + (time_bias[1:] - time_bias[:-1]).square().mean()
        )
        daily = factors.new_zeros(())
        if factors.shape[0] > self.config.period:
            daily = (
                factors[self.config.period :]
                - factors[: -self.config.period]
            ).square().mean() + (
                time_bias[self.config.period :]
                - time_bias[: -self.config.period]
            ).square().mean()
        residual_loading = self.series_residual.weight.square().mean()
        return (
            self.config.smoothness_weight * smoothness
            + self.config.daily_weight * daily
            + self.config.residual_loading_weight * residual_loading
        )


def robust_reconstruction_loss(
    prediction: Tensor,
    target: Tensor,
    observed: Tensor,
    delta: float = 1.0,
) -> Tensor:
    if not torch.any(observed):
        return prediction.sum() * 0.0
    return F.huber_loss(
        prediction[observed],
        target[observed],
        delta=delta,
    )


def reconstruct_from_factors(
    factors: Tensor,
    time_bias: Tensor,
    loadings: Tensor,
    series_bias: Tensor,
    global_bias: Tensor,
) -> Tensor:
    return (
        factors @ loadings.transpose(0, 1)
        + time_bias
        + series_bias.transpose(0, 1)
        + global_bias
    )


def infer_split_factors(
    values: Tensor,
    model: MetadataPanelFactor,
    initial_factor: Tensor,
    reference_factors: Tensor | None = None,
    reference_time_bias: Tensor | None = None,
    steps: int = 60,
    learning_rate: float = 0.05,
) -> tuple[Tensor, Tensor, Tensor, list[float]]:
    """Infer split-local factors while keeping learned series relations fixed."""
    if steps < 1:
        raise ValueError("steps must be positive.")
    device = next(model.parameters()).device
    target = values.to(device)
    observed = torch.isfinite(target)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    model.eval()
    loadings = model.series_loadings().detach()
    series_bias = model.series_bias.weight.detach()
    global_bias = model.global_bias.detach()
    if reference_factors is not None:
        reference = reference_factors.detach().to(device)
        expected_shape = (values.shape[0], model.config.rank)
        if tuple(reference.shape) != expected_shape:
            raise ValueError(
                "reference_factors must have shape "
                f"{expected_shape}, got {tuple(reference.shape)}."
            )
        factor_start = reference.clone()
    else:
        reference = None
        factor_start = (
            initial_factor.detach()
            .to(device)
            .expand(values.shape[0], -1)
            .clone()
        )
    factors = nn.Parameter(factor_start)
    if reference_time_bias is not None:
        bias_reference = reference_time_bias.detach().to(device)
        expected_bias_shape = (values.shape[0], 1)
        if tuple(bias_reference.shape) != expected_bias_shape:
            raise ValueError(
                "reference_time_bias must have shape "
                f"{expected_bias_shape}, got {tuple(bias_reference.shape)}."
            )
        bias_start = bias_reference.clone()
    else:
        bias_reference = None
        bias_start = torch.zeros((values.shape[0], 1), device=device)
    time_bias = nn.Parameter(bias_start)
    optimizer = torch.optim.Adam((factors, time_bias), lr=learning_rate)
    history: list[float] = []

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = reconstruct_from_factors(
            factors,
            time_bias,
            loadings,
            series_bias,
            global_bias,
        )
        reconstruction = robust_reconstruction_loss(
            prediction, target, observed
        )
        smoothness = (
            (factors[1:] - factors[:-1]).square().mean()
            if len(factors) > 1
            else factors.new_zeros(())
        )
        daily = (
            (factors - reference).square().mean()
            if reference is not None
            else factors.new_zeros(())
        )
        if bias_reference is not None:
            daily = daily + (time_bias - bias_reference).square().mean()
        loss = (
            reconstruction
            + model.config.smoothness_weight * smoothness
            + model.config.daily_weight * daily
        )
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))

    with torch.no_grad():
        prediction = reconstruct_from_factors(
            factors,
            time_bias,
            loadings,
            series_bias,
            global_bias,
        )
    return (
        factors.detach(),
        time_bias.detach(),
        prediction.detach().cpu(),
        history,
    )


@torch.inference_mode()
def factor_effect_scores(
    factors: Tensor,
    reference_factors: Tensor,
    loadings: Tensor,
    time_bias: Tensor | None = None,
    reference_time_bias: Tensor | None = None,
) -> Tensor:
    """Map an unexpected shared-state change back to every series."""
    if factors.shape != reference_factors.shape:
        raise ValueError("factors and reference_factors must have equal shapes.")
    effect = (factors - reference_factors) @ loadings.transpose(0, 1)
    if (time_bias is None) != (reference_time_bias is None):
        raise ValueError(
            "time_bias and reference_time_bias must be provided together."
        )
    if time_bias is not None and reference_time_bias is not None:
        if time_bias.shape != reference_time_bias.shape:
            raise ValueError(
                "time_bias and reference_time_bias must have equal shapes."
            )
        effect = effect + time_bias - reference_time_bias
    return torch.abs(effect)
