from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .baselines import build_baseline_features
from .data import (
    MetadataFeatures,
    chronological_splits,
    filter_useful_series,
    load_metadata_features,
    load_wide_csv,
)
from .panel_factor import (
    MetadataPanelFactor,
    PanelFactorConfig,
    factor_effect_scores,
    infer_split_factors,
    robust_reconstruction_loss,
)
from .reporting import (
    error_metrics,
    save_model_anomaly_report,
    save_panel_factor_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a metadata-aware robust wide-panel factor model."
    )
    parser.add_argument("--csv", type=Path, default=Path("final.csv"))
    parser.add_argument("--metadata", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/panel_factor")
    )
    parser.add_argument(
        "--baseline",
        choices=("seasonal_mad", "holt_winters", "none"),
        default="seasonal_mad",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--period", type=int, default=288)
    parser.add_argument("--min-train-observations", type=int, default=288)
    parser.add_argument("--min-train-coverage", type=float, default=0.8)
    parser.add_argument(
        "--min-informative-observations", type=int, default=12
    )
    parser.add_argument("--clip-value", type=float, default=20.0)

    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--time-batch-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--smoothness-weight", type=float, default=0.01)
    parser.add_argument("--daily-weight", type=float, default=0.01)
    parser.add_argument("--residual-loading-weight", type=float, default=0.001)
    parser.add_argument("--collective-weight", type=float, default=0.5)
    parser.add_argument("--inference-steps", type=int, default=60)
    parser.add_argument("--inference-learning-rate", type=float, default=0.05)
    parser.add_argument("--threshold-quantile", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "rank",
        "epochs",
        "time_batch_size",
        "inference_steps",
        "period",
        "min_train_observations",
        "min_informative_observations",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.learning_rate <= 0 or args.inference_learning_rate <= 0:
        raise ValueError("Learning rates must be positive.")
    if not 0.0 <= args.min_train_coverage <= 1.0:
        raise ValueError("--min-train-coverage must be in [0, 1].")
    for name in (
        "dropout",
        "smoothness_weight",
        "daily_weight",
        "residual_loading_weight",
        "collective_weight",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative.")
    if args.dropout >= 1:
        raise ValueError("--dropout must be below one.")
    if not 0.0 < args.threshold_quantile < 1.0:
        raise ValueError("--threshold-quantile must be between zero and one.")


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def combined_anomaly_scores(
    prediction: Tensor,
    target: Tensor,
    collective_scores: Tensor,
    collective_weight: float,
) -> Tensor:
    observed = torch.isfinite(target)
    target_filled = torch.nan_to_num(
        target, nan=0.0, posinf=0.0, neginf=0.0
    )
    scores = torch.full_like(target, torch.nan)
    cell_scores = (
        (prediction - target_filled).abs()
        + collective_weight * collective_scores
    )
    scores[observed] = cell_scores[observed]
    return scores


def daily_references(
    factors: Tensor,
    period: int,
) -> Tensor:
    """Use the previous step initially, then the same phase one day earlier."""
    references = factors.clone()
    if len(factors) > 1:
        references[1:] = factors[:-1]
    if len(factors) > period:
        references[period:] = factors[:-period]
    return references


@torch.inference_mode()
def reconstruct_training(
    model: MetadataPanelFactor,
    batch_size: int,
) -> Tensor:
    device = next(model.parameters()).device
    predictions: list[Tensor] = []
    model.eval()
    for start in range(0, model.train_length, batch_size):
        indices = torch.arange(
            start,
            min(start + batch_size, model.train_length),
            device=device,
        )
        predictions.append(model.reconstruct_train(indices).cpu())
    return torch.cat(predictions)


def train_model(
    model: MetadataPanelFactor,
    train_values: Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    use_amp: bool,
) -> list[dict[str, float | int]]:
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float | int]] = []
    batch_count = math.ceil(len(train_values) / batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_values))
        reconstruction_sum = 0.0
        observed_count = 0
        for start in range(0, len(order), batch_size):
            time_indices = order[start : start + batch_size]
            target = train_values[time_indices].to(
                device, non_blocking=device.type == "cuda"
            )
            observed = torch.isfinite(target)
            target = torch.nan_to_num(
                target, nan=0.0, posinf=0.0, neginf=0.0
            )
            device_indices = time_indices.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=use_amp
            ):
                prediction = model.reconstruct_train(device_indices)
                reconstruction = robust_reconstruction_loss(
                    prediction, target, observed
                )
                regularization = model.temporal_regularization() / batch_count
                loss = reconstruction + regularization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            count = int(observed.sum())
            reconstruction_sum += float(reconstruction.detach()) * count
            observed_count += count

        row: dict[str, float | int] = {
            "epoch": epoch,
            "reconstruction_loss": reconstruction_sum
            / max(observed_count, 1),
            "regularization": float(
                model.temporal_regularization().detach()
            ),
        }
        history.append(row)
        print(json.dumps(row))
    return history


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = choose_device(args.device)
    use_amp = args.amp and device.type == "cuda"

    loaded = load_wide_csv(args.csv)
    splits = chronological_splits(
        len(loaded.timestamps),
        args.train_ratio,
        args.validation_ratio,
    )
    filter_result = filter_useful_series(
        loaded.values,
        splits.train,
        min_train_observations=args.min_train_observations,
        min_train_coverage=args.min_train_coverage,
        min_informative_observations=args.min_informative_observations,
    )
    keep = filter_result.keep
    raw_values = loaded.values[:, keep]
    columns = [
        column
        for column, selected in zip(
            loaded.columns, keep.tolist(), strict=True
        )
        if selected
    ]
    metadata: MetadataFeatures | None = None
    if args.metadata is not None:
        metadata = load_metadata_features(
            args.metadata,
            keep_series=keep,
            expected_series_count=len(loaded.columns),
        )

    baseline = build_baseline_features(
        raw_values,
        splits,
        method=args.baseline,
        period=args.period,
    )
    features = torch.clamp(
        baseline.features, min=-args.clip_value, max=args.clip_value
    )
    config = PanelFactorConfig(
        rank=args.rank,
        period=args.period,
        dropout=args.dropout,
        smoothness_weight=args.smoothness_weight,
        daily_weight=args.daily_weight,
        residual_loading_weight=args.residual_loading_weight,
        collective_weight=args.collective_weight,
    )
    model = MetadataPanelFactor(
        train_length=splits.train.stop,
        series_count=features.shape[1],
        config=config,
        metadata=metadata,
    ).to(device)
    print(
        f"device={device} rows={features.shape[0]} "
        f"series={features.shape[1]} rank={args.rank} "
        f"metadata={metadata is not None}"
    )
    print(json.dumps({"series_filter": filter_result.report}))
    history = train_model(
        model,
        features[splits.train],
        epochs=args.epochs,
        batch_size=args.time_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_amp=use_amp,
    )

    train_prediction = reconstruct_training(model, args.time_batch_size)
    model.eval()
    train_factors = model.time_factors.weight.detach().cpu()
    train_time_bias = model.time_bias.weight.detach().cpu()
    loadings = model.series_loadings().detach().cpu()
    train_reference = daily_references(train_factors, args.period)
    train_bias_reference = daily_references(train_time_bias, args.period)
    train_collective = factor_effect_scores(
        train_factors,
        train_reference,
        loadings,
        train_time_bias,
        train_bias_reference,
    )
    train_scores = combined_anomaly_scores(
        train_prediction,
        features[splits.train],
        train_collective,
        args.collective_weight,
    )

    validation_global_indices = torch.arange(
        splits.validation.start, splits.validation.stop
    )
    validation_reference = train_factors[
        validation_global_indices - args.period
    ]
    validation_bias_reference = train_time_bias[
        validation_global_indices - args.period
    ]
    initial_factor = validation_reference[0]
    (
        validation_factors,
        validation_time_bias,
        validation_prediction,
        validation_history,
    ) = infer_split_factors(
        features[splits.validation],
        model,
        initial_factor=initial_factor,
        reference_factors=validation_reference,
        reference_time_bias=validation_bias_reference,
        steps=args.inference_steps,
        learning_rate=args.inference_learning_rate,
    )
    known_factors = torch.cat(
        (train_factors, validation_factors.cpu()), dim=0
    )
    known_time_bias = torch.cat(
        (train_time_bias, validation_time_bias.cpu()), dim=0
    )
    test_global_indices = torch.arange(splits.test.start, splits.test.stop)
    test_reference = known_factors[test_global_indices - args.period]
    test_bias_reference = known_time_bias[
        test_global_indices - args.period
    ]
    (
        test_factors,
        test_time_bias,
        test_prediction,
        test_history,
    ) = infer_split_factors(
        features[splits.test],
        model,
        initial_factor=test_reference[0],
        reference_factors=test_reference,
        reference_time_bias=test_bias_reference,
        steps=args.inference_steps,
        learning_rate=args.inference_learning_rate,
    )
    validation_collective = factor_effect_scores(
        validation_factors.cpu(),
        validation_reference,
        loadings,
        validation_time_bias.cpu(),
        validation_bias_reference,
    )
    validation_scores = combined_anomaly_scores(
        validation_prediction,
        features[splits.validation],
        validation_collective,
        args.collective_weight,
    )
    test_collective = factor_effect_scores(
        test_factors.cpu(),
        test_reference,
        loadings,
        test_time_bias.cpu(),
        test_bias_reference,
    )
    test_scores = combined_anomaly_scores(
        test_prediction,
        features[splits.test],
        test_collective,
        args.collective_weight,
    )
    full_scores = torch.cat(
        (train_scores, validation_scores, test_scores), dim=0
    )
    finite_validation = validation_scores[
        torch.isfinite(validation_scores)
    ]
    threshold = float(
        torch.quantile(finite_validation, args.threshold_quantile)
    )

    split_metrics = {
        "train": error_metrics(train_scores),
        "validation": error_metrics(validation_scores),
        "test": error_metrics(test_scores),
    }
    split_metrics["test"]["threshold_exceedance_rate"] = float(
        (
            test_scores[torch.isfinite(test_scores)] > threshold
        ).float().mean()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    anomaly_summary = save_model_anomaly_report(
        args.output_dir,
        scores=full_scores,
        threshold=threshold,
        timestamps=loaded.timestamps,
        raw_values=raw_values,
        series_columns=columns,
        validation_slice=splits.validation,
        test_slice=splits.test,
        series_filter_report=filter_result.report,
    )
    metrics: dict[str, Any] = {
        "model": "metadata_panel_factor",
        "config": config.as_dict(),
        "series_filter": filter_result.report,
        "split_metrics": split_metrics,
        "threshold": threshold,
        "history": history,
        "validation_inference_loss": validation_history,
        "test_inference_loss": test_history,
        "full_grid_anomaly": anomaly_summary,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    save_panel_factor_report(
        args.output_dir,
        history=history,
        split_metrics=split_metrics,
        validation_inference_loss=validation_history,
        test_inference_loss=test_history,
    )
    torch.save(
        {
            "model_state": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
            "config": config.as_dict(),
            "baseline_state": baseline.state,
            "series_columns": columns,
            "splits": splits.as_dict(),
            "series_filter": filter_result.report,
            "threshold": threshold,
            "arguments": {
                **vars(args),
                "csv": str(args.csv),
                "metadata": (
                    str(args.metadata) if args.metadata is not None else None
                ),
                "output_dir": str(args.output_dir),
            },
        },
        args.output_dir / "checkpoint.pt",
    )
    torch.save(
        {
            "scores": full_scores,
            "threshold": threshold,
            "timestamps": loaded.timestamps,
            "series_columns": columns,
            "series_filter": filter_result.report,
        },
        args.output_dir / "full_grid_scores.pt",
    )
    print(json.dumps({"threshold": threshold, "splits": split_metrics}))


if __name__ == "__main__":
    main()
