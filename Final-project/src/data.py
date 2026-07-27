from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SplitSlices:
    train: slice
    validation: slice
    test: slice

    def as_dict(self) -> dict[str, tuple[int, int]]:
        return {
            "train": (self.train.start, self.train.stop),
            "validation": (self.validation.start, self.validation.stop),
            "test": (self.test.start, self.test.stop),
        }


@dataclass
class WideSeries:
    timestamps: Tensor
    values: Tensor
    columns: list[str]


@dataclass
class MetadataFeatures:
    categorical: Tensor
    numeric: Tensor
    cardinalities: tuple[int, ...]
    categorical_names: tuple[str, ...]


@dataclass
class ContextFeatures:
    global_time: Tensor
    per_series: Tensor | None
    names: tuple[str, ...]


@dataclass(frozen=True)
class SeriesFilterResult:
    keep: Tensor
    report: dict[str, Any]


def chronological_splits(
    length: int,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> SplitSlices:
    if length < 3:
        raise ValueError("At least three timestamps are required.")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between zero and one.")
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0, 1).")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be below one.")

    train_end = int(length * train_ratio)
    validation_end = train_end + int(length * validation_ratio)
    if train_end == 0 or validation_end == train_end or validation_end == length:
        raise ValueError("The requested split produces an empty partition.")

    return SplitSlices(
        train=slice(0, train_end),
        validation=slice(train_end, validation_end),
        test=slice(validation_end, length),
    )


def load_wide_csv(
    path: str | Path,
    timestamp_column: str = "timestamp",
    expected_step_seconds: int = 300,
) -> WideSeries:
    path = Path(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if timestamp_column not in header:
        raise ValueError(f"{timestamp_column!r} is not present in {path}.")

    dtypes: dict[str, str] = {column: "float32" for column in header}
    dtypes[timestamp_column] = "int64"
    frame = pd.read_csv(path, dtype=dtypes)

    timestamps_np = frame.pop(timestamp_column).to_numpy(dtype=np.int64, copy=True)
    if len(timestamps_np) < 2:
        raise ValueError("The input must contain at least two timestamps.")
    steps = np.diff(timestamps_np)
    if np.any(steps <= 0):
        raise ValueError("Timestamps must be strictly increasing.")
    if expected_step_seconds and np.any(steps != expected_step_seconds):
        unique_steps = np.unique(steps).tolist()
        raise ValueError(
            f"Expected {expected_step_seconds}-second spacing; found {unique_steps}."
        )

    values_np = frame.to_numpy(dtype=np.float32, copy=True)
    values_np[np.isinf(values_np)] = np.nan
    return WideSeries(
        timestamps=torch.from_numpy(timestamps_np),
        values=torch.from_numpy(values_np),
        columns=frame.columns.tolist(),
    )


def select_usable_series(
    values: Tensor,
    train_slice: slice,
    min_train_observations: int,
) -> Tensor:
    train_values = values[train_slice]
    observed_count = torch.isfinite(train_values).sum(dim=0)
    keep = observed_count >= min_train_observations
    if not torch.any(keep):
        raise ValueError(
            "No series has enough finite training observations. "
            "Lower --min-train-observations."
        )
    return keep


def filter_useful_series(
    values: Tensor,
    train_slice: slice,
    min_train_observations: int = 288,
    min_train_coverage: float = 0.8,
    min_informative_observations: int = 12,
    absolute_tolerance: float = 1e-8,
    relative_tolerance: float = 1e-6,
) -> SeriesFilterResult:
    """Remove unlearnable series using training data only.

    A useful series has sufficient finite coverage, a non-trivial range, and
    enough values meaningfully different from its training median. The last
    condition removes almost-always-constant counters without discarding a
    normally sparse but repeatedly active signal.
    """
    if values.ndim != 2:
        raise ValueError("values must have shape [time, series].")
    if min_train_observations < 1:
        raise ValueError("min_train_observations must be positive.")
    if not 0.0 <= min_train_coverage <= 1.0:
        raise ValueError("min_train_coverage must be in [0, 1].")
    if min_informative_observations < 1:
        raise ValueError("min_informative_observations must be positive.")
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("filter tolerances cannot be negative.")

    train = values[train_slice]
    observed = torch.isfinite(train)
    observed_count = observed.sum(dim=0)
    required_count = max(
        min_train_observations,
        int(np.ceil(train.shape[0] * min_train_coverage)),
    )
    enough_observations = observed_count >= required_count

    positive_infinity = torch.full_like(train, torch.inf)
    negative_infinity = torch.full_like(train, -torch.inf)
    minimum = torch.where(observed, train, positive_infinity).amin(dim=0)
    maximum = torch.where(observed, train, negative_infinity).amax(dim=0)
    center = torch.nanmedian(train, dim=0).values
    tolerance = torch.maximum(
        torch.full_like(center, absolute_tolerance),
        center.abs() * relative_tolerance,
    )
    value_range = maximum - minimum
    variable = torch.isfinite(value_range) & (value_range > tolerance)

    deviations = torch.where(
        observed,
        (train - center).abs(),
        torch.zeros_like(train),
    )
    informative_count = (deviations > tolerance).sum(dim=0)
    informative = informative_count >= min_informative_observations

    maximum_absolute = torch.where(
        observed, train.abs(), torch.zeros_like(train)
    ).amax(dim=0)
    all_zero = enough_observations & (
        maximum_absolute <= absolute_tolerance
    )
    constant_or_near_constant = (
        enough_observations & ~all_zero & ~variable
    )
    too_few_informative = (
        enough_observations & variable & ~informative
    )
    keep = enough_observations & variable & informative
    if not torch.any(keep):
        raise ValueError(
            "The quality filter removed every series. Lower the coverage or "
            "informative-observation requirement."
        )

    report: dict[str, Any] = {
        "input_series": values.shape[1],
        "retained_series": int(keep.sum()),
        "removed_series": int((~keep).sum()),
        "retained_percent": 100.0 * float(keep.float().mean()),
        "minimum_required_train_observations": required_count,
        "minimum_train_coverage": min_train_coverage,
        "minimum_informative_observations": min_informative_observations,
        "removed_reasons": {
            "insufficient_train_observations": int(
                (~enough_observations).sum()
            ),
            "always_zero": int(all_zero.sum()),
            "constant_or_numerically_flat": int(
                constant_or_near_constant.sum()
            ),
            "too_few_informative_values": int(
                too_few_informative.sum()
            ),
        },
    }
    return SeriesFilterResult(keep=keep, report=report)


def load_metadata_features(
    path: str | Path,
    keep_series: Tensor,
    expected_series_count: int,
) -> MetadataFeatures:
    frame = pd.read_csv(path)
    required = {
        "series_index",
        "group_tag",
        "query_tag",
        "label_tag",
        "sample_count",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    if len(frame) != expected_series_count:
        raise ValueError(
            f"Metadata has {len(frame)} rows, expected {expected_series_count}."
        )

    expected_indices = np.arange(1, expected_series_count + 1)
    if not np.array_equal(frame["series_index"].to_numpy(), expected_indices):
        raise ValueError("Metadata rows are not in one-based series order.")

    keep_np = keep_series.cpu().numpy()
    selected = frame.loc[keep_np].reset_index(drop=True)
    categorical_names = ("group_tag", "query_tag", "label_tag")
    categorical_np = selected.loc[:, categorical_names].to_numpy(
        dtype=np.int64, copy=True
    )
    cardinalities = tuple(
        int(frame[column].max()) + 1 for column in categorical_names
    )

    sample_count = np.log1p(
        selected["sample_count"].to_numpy(dtype=np.float32, copy=True)
    )
    sample_std = float(sample_count.std())
    if sample_std < 1e-6:
        sample_std = 1.0
    sample_count = (sample_count - float(sample_count.mean())) / sample_std

    return MetadataFeatures(
        categorical=torch.from_numpy(categorical_np),
        numeric=torch.from_numpy(sample_count[:, None]),
        cardinalities=cardinalities,
        categorical_names=categorical_names,
    )


def _leave_one_out_group_mean(
    values: Tensor,
    observed: Tensor,
    tags: Tensor,
    cardinality: int,
    fallback: Tensor,
) -> Tensor:
    """Return each series' peer mean without including the series itself."""
    time_steps, series_count = values.shape
    expanded_tags = tags.unsqueeze(0).expand(time_steps, series_count)
    filled = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    group_sum = torch.zeros(
        (time_steps, cardinality), dtype=values.dtype, device=values.device
    )
    group_count = torch.zeros_like(group_sum)
    group_sum.scatter_add_(1, expanded_tags, filled)
    group_count.scatter_add_(1, expanded_tags, observed.to(values.dtype))

    peer_sum = group_sum[:, tags] - filled
    peer_count = group_count[:, tags] - observed.to(values.dtype)
    return torch.where(
        peer_count > 0,
        peer_sum / peer_count.clamp_min(1.0),
        fallback.expand(-1, series_count),
    )


def build_context_features(
    values: Tensor,
    timestamps: Tensor,
    metadata: MetadataFeatures | None,
) -> ContextFeatures:
    """Build causal cross-series and calendar inputs for contextual models."""
    if values.ndim != 2:
        raise ValueError("values must have shape [time, series].")
    if timestamps.shape[0] != values.shape[0]:
        raise ValueError("timestamps and values have different lengths.")

    observed = torch.isfinite(values)
    filled = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    count = observed.sum(dim=1, keepdim=True).clamp_min(1)
    count_float = count.to(values.dtype)
    global_mean = filled.sum(dim=1, keepdim=True) / count_float
    centered = torch.where(observed, values - global_mean, 0.0)
    global_std = torch.sqrt(
        centered.square().sum(dim=1, keepdim=True) / count_float
    )
    global_abs_mean = filled.abs().sum(dim=1, keepdim=True) / count_float
    observed_fraction = count_float / values.shape[1]

    seconds_in_day = timestamps.remainder(86_400).to(values.dtype)
    angle = 2.0 * torch.pi * seconds_in_day / 86_400.0
    global_time = torch.cat(
        (
            global_mean,
            global_std,
            global_abs_mean,
            observed_fraction,
            torch.sin(angle).unsqueeze(1),
            torch.cos(angle).unsqueeze(1),
        ),
        dim=1,
    )
    names = (
        "global_mean",
        "global_std",
        "global_abs_mean",
        "observed_fraction",
        "time_of_day_sin",
        "time_of_day_cos",
    )

    per_series = None
    if metadata is not None:
        peer_context: list[Tensor] = []
        for column, cardinality in enumerate(metadata.cardinalities):
            peer_context.append(
                _leave_one_out_group_mean(
                    values,
                    observed,
                    metadata.categorical[:, column],
                    cardinality,
                    global_mean,
                )
            )
        per_series = torch.stack(peer_context, dim=-1)
        names += tuple(f"{name}_peer_mean" for name in metadata.categorical_names)

    return ContextFeatures(
        global_time=global_time,
        per_series=per_series,
        names=names,
    )


class WindowDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Samples series/time pairs without materializing millions of windows."""

    def __init__(
        self,
        values: Tensor,
        target_slice: slice,
        window: int,
        max_samples: int,
        random_sampling: bool,
        seed: int = 17,
        context: ContextFeatures | None = None,
    ) -> None:
        if values.ndim != 2:
            raise ValueError("values must have shape [time, series].")
        if window < 1:
            raise ValueError("window must be positive.")
        if max_samples < 1:
            raise ValueError("max_samples must be positive.")

        start = max(int(target_slice.start), window)
        stop = int(target_slice.stop)
        if start >= stop:
            raise ValueError("The split is too short for the requested window.")

        self.values = values
        self.observed = torch.isfinite(values)
        self.start = start
        self.stop = stop
        self.window = window
        self.series_count = values.shape[1]
        self.total_pairs = (stop - start) * self.series_count
        self.length = min(max_samples, self.total_pairs)
        self.random_sampling = random_sampling
        self.seed = seed
        self.epoch = 0
        self.context = context
        self.input_size = 2
        if context is not None:
            if context.global_time.shape[0] != values.shape[0]:
                raise ValueError("Global context has the wrong time dimension.")
            self.input_size += context.global_time.shape[1]
            if context.per_series is not None:
                if context.per_series.shape[:2] != values.shape:
                    raise ValueError("Per-series context has the wrong shape.")
                self.input_size += context.per_series.shape[2]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def _flat_index(self, index: int) -> int:
        if self.random_sampling:
            mask = (1 << 64) - 1
            value = index + self.seed + self.epoch * self.length
            value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
            value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
            return (value ^ (value >> 31)) % self.total_pairs
        return (index * self.total_pairs) // self.length

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        flat_index = self._flat_index(index)
        time_offset, series_index = divmod(flat_index, self.series_count)
        target_time = self.start + time_offset

        history = self.values[
            target_time - self.window : target_time, series_index
        ]
        history_observed = self.observed[
            target_time - self.window : target_time, series_index
        ]
        input_parts = [
            torch.stack(
                (
                    torch.nan_to_num(history, nan=0.0, posinf=0.0, neginf=0.0),
                    history_observed.to(torch.float32),
                ),
                dim=-1,
            )
        ]
        if self.context is not None:
            history_slice = slice(target_time - self.window, target_time)
            input_parts.append(self.context.global_time[history_slice])
            if self.context.per_series is not None:
                input_parts.append(
                    self.context.per_series[history_slice, series_index]
                )
        inputs = torch.cat(input_parts, dim=-1)

        target = self.values[target_time, series_index]
        target_observed = self.observed[target_time, series_index]
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        return (
            inputs,
            target.unsqueeze(0),
            target_observed.unsqueeze(0),
            torch.tensor(series_index, dtype=torch.long),
        )
