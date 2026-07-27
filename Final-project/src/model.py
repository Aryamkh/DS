from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import MetadataFeatures


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "gru"
    input_size: int = 2
    hidden_size: int = 64
    layers: int = 1
    dropout: float = 0.0
    tcn_blocks: int = 4
    group_embedding_dim: int = 4
    query_embedding_dim: int = 16
    label_embedding_dim: int = 8

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class MetadataEncoder(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        metadata: MetadataFeatures | None,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList()
        self.output_size = 0

        if metadata is None:
            self.register_buffer("categorical", None, persistent=True)
            self.register_buffer("numeric", None, persistent=True)
            return

        dimensions = (
            config.group_embedding_dim,
            config.query_embedding_dim,
            config.label_embedding_dim,
        )
        if len(metadata.cardinalities) != len(dimensions):
            raise ValueError("Unexpected metadata categorical width.")
        for cardinality, dimension in zip(
            metadata.cardinalities, dimensions, strict=True
        ):
            self.embeddings.append(nn.Embedding(cardinality, dimension))
        self.register_buffer(
            "categorical", metadata.categorical, persistent=True
        )
        self.register_buffer("numeric", metadata.numeric, persistent=True)
        self.output_size = sum(dimensions) + metadata.numeric.shape[1]

    def forward(self, series_index: Tensor) -> Tensor | None:
        if self.categorical is None or self.numeric is None:
            return None
        tags = self.categorical[series_index]
        embedded = [
            embedding(tags[:, column])
            for column, embedding in enumerate(self.embeddings)
        ]
        return torch.cat((*embedded, self.numeric[series_index]), dim=-1)


class GlobalResidualGRU(nn.Module):
    """A small forecasting model shared by every metric series."""

    def __init__(
        self,
        config: ModelConfig,
        metadata: MetadataFeatures | None = None,
    ) -> None:
        super().__init__()
        if config.hidden_size < 1 or config.layers < 1:
            raise ValueError("hidden_size and layers must be positive.")

        effective_dropout = config.dropout if config.layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.metadata = MetadataEncoder(config, metadata)

        head_width = max(16, config.hidden_size // 2)
        self.head = nn.Sequential(
            nn.Linear(
                config.hidden_size + self.metadata.output_size, head_width
            ),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(head_width, 1),
        )

    def forward(self, inputs: Tensor, series_index: Tensor) -> Tensor:
        encoded, _ = self.gru(inputs)
        features = encoded[:, -1]
        metadata_features = self.metadata(series_index)
        if metadata_features is not None:
            features = torch.cat((features, metadata_features), dim=-1)
        return self.head(features)


class CausalResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.convolution = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.normalization = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        encoded = F.pad(inputs, (self.left_padding, 0))
        encoded = self.convolution(encoded)
        encoded = encoded.transpose(1, 2)
        encoded = self.normalization(encoded)
        encoded = F.silu(encoded).transpose(1, 2)
        return residual + self.dropout(encoded)


class ContextualResidualTCN(nn.Module):
    """Temporal convolution with cross-series context and metadata gating."""

    def __init__(
        self,
        config: ModelConfig,
        metadata: MetadataFeatures | None = None,
    ) -> None:
        super().__init__()
        if config.hidden_size < 1 or config.tcn_blocks < 1:
            raise ValueError("hidden_size and tcn_blocks must be positive.")
        self.input_projection = nn.Linear(
            config.input_size, config.hidden_size
        )
        self.blocks = nn.ModuleList(
            CausalResidualBlock(
                hidden_size=config.hidden_size,
                kernel_size=3,
                dilation=2**block,
                dropout=config.dropout,
            )
            for block in range(config.tcn_blocks)
        )
        self.attention = nn.Linear(config.hidden_size, 1)
        self.temporal_norm = nn.LayerNorm(config.hidden_size)
        self.metadata = MetadataEncoder(config, metadata)

        if self.metadata.output_size:
            self.metadata_projection = nn.Sequential(
                nn.Linear(self.metadata.output_size, config.hidden_size),
                nn.SiLU(),
            )
            self.metadata_gate = nn.Linear(2 * config.hidden_size, config.hidden_size)
        else:
            self.metadata_projection = None
            self.metadata_gate = None

        head_width = max(32, config.hidden_size // 2)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, head_width),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(head_width, 1),
        )

    def forward(self, inputs: Tensor, series_index: Tensor) -> Tensor:
        sequence = self.input_projection(inputs)
        encoded = sequence.transpose(1, 2)
        for block in self.blocks:
            encoded = block(encoded)
        encoded = encoded.transpose(1, 2)

        weights = torch.softmax(self.attention(encoded).squeeze(-1), dim=1)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        temporal = self.temporal_norm(encoded[:, -1] + pooled)

        metadata_features = self.metadata(series_index)
        if (
            metadata_features is not None
            and self.metadata_projection is not None
            and self.metadata_gate is not None
        ):
            metadata_state = self.metadata_projection(metadata_features)
            gate = torch.sigmoid(
                self.metadata_gate(torch.cat((temporal, metadata_state), dim=-1))
            )
            temporal = gate * temporal + (1.0 - gate) * metadata_state

        return self.head(temporal)


def build_model(
    config: ModelConfig,
    metadata: MetadataFeatures | None = None,
) -> nn.Module:
    if config.architecture == "gru":
        return GlobalResidualGRU(config, metadata=metadata)
    if config.architecture == "context_tcn":
        return ContextualResidualTCN(config, metadata=metadata)
    raise ValueError(f"Unknown architecture: {config.architecture}")
