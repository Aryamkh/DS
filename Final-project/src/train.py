from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .baselines import build_baseline_features
from .data import (
    ContextFeatures,
    MetadataFeatures,
    SplitSlices,
    WindowDataset,
    build_context_features,
    chronological_splits,
    filter_useful_series,
    load_metadata_features,
    load_wide_csv,
)
from .model import ModelConfig, build_model
from .reporting import save_model_anomaly_report, save_training_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a global PyTorch residual forecaster."
    )
    parser.add_argument("--csv", type=Path, default=Path("final.csv"))
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/run"))
    parser.add_argument(
        "--baseline",
        choices=("seasonal_mad", "holt_winters", "none"),
        default="seasonal_mad",
    )
    parser.add_argument(
        "--architecture",
        choices=("gru", "context_tcn"),
        default="gru",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--period", type=int, default=288)
    parser.add_argument("--min-train-observations", type=int, default=288)
    parser.add_argument("--min-train-coverage", type=float, default=0.8)
    parser.add_argument(
        "--min-informative-observations", type=int, default=12
    )
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--clip-value", type=float, default=20.0)

    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--tcn-blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--samples-per-epoch", type=int, default=100_000)
    parser.add_argument("--evaluation-samples", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--full-score",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Score the time x series grid and write anomaly-rate reports.",
    )
    parser.add_argument("--score-batch-size", type=int, default=1024)
    parser.add_argument(
        "--score-max-time-points",
        type=int,
        default=0,
        help="Zero scores every timestamp; positive values are for smoke tests.",
    )
    parser.add_argument(
        "--score-max-series",
        type=int,
        default=0,
        help="Zero scores every series; positive values are for smoke tests.",
    )

    parser.add_argument("--hw-alpha", type=float, default=0.2)
    parser.add_argument("--hw-beta", type=float, default=0.02)
    parser.add_argument("--hw-gamma", type=float, default=0.05)
    parser.add_argument("--threshold-quantile", type=float, default=0.995)
    parser.add_argument("--num-workers", type=int, default=0)
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
        help="Use automatic mixed precision on CUDA.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "period",
        "min_train_observations",
        "min_informative_observations",
        "window",
        "batch_size",
        "samples_per_epoch",
        "evaluation_samples",
        "epochs",
        "patience",
        "tcn_blocks",
        "score_batch_size",
    )
    for name in positive_names:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if not 0.0 <= args.min_train_coverage <= 1.0:
        raise ValueError("--min-train-coverage must be in [0, 1].")
    if args.clip_value <= 0.0:
        raise ValueError("--clip-value must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")
    if args.score_max_time_points < 0 or args.score_max_series < 0:
        raise ValueError("Scoring limits cannot be negative.")
    if not 0.0 < args.threshold_quantile < 1.0:
        raise ValueError("--threshold-quantile must be between zero and one.")


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return device
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


def make_loader(
    dataset: WindowDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def move_batch(
    batch: tuple[Tensor, Tensor, Tensor, Tensor],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return tuple(
        item.to(device, non_blocking=device.type == "cuda") for item in batch
    )  # type: ignore[return-value]


def run_training_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    gradient_clip: float,
    use_amp: bool,
) -> float:
    model.train()
    loss_sum = 0.0
    observed_count = 0

    for batch in loader:
        inputs, target, observed, series_index = move_batch(batch, device)
        valid = observed.squeeze(-1)
        if not torch.any(valid):
            continue

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            prediction = model(inputs, series_index)
            loss = nn.functional.smooth_l1_loss(
                prediction[valid], target[valid]
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        count = int(valid.sum())
        loss_sum += float(loss.detach()) * count
        observed_count += count

    return loss_sum / max(observed_count, 1)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]],
    device: torch.device,
    use_amp: bool,
) -> tuple[dict[str, float], Tensor]:
    model.eval()
    absolute_errors: list[Tensor] = []
    squared_error_sum = 0.0
    observed_count = 0

    for batch in loader:
        inputs, target, observed, series_index = move_batch(batch, device)
        valid = observed.squeeze(-1)
        if not torch.any(valid):
            continue
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            prediction = model(inputs, series_index)
        errors = prediction[valid].float() - target[valid].float()
        absolute_errors.append(errors.abs().cpu())
        squared_error_sum += float(errors.square().sum())
        observed_count += errors.numel()

    if not absolute_errors:
        raise RuntimeError("No finite targets were available for evaluation.")
    absolute = torch.cat(absolute_errors)
    metrics = {
        "mae": float(absolute.mean()),
        "rmse": math.sqrt(squared_error_sum / observed_count),
        "samples": float(observed_count),
    }
    return metrics, absolute


def cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _evenly_spaced_indices(length: int, maximum: int) -> Tensor:
    if maximum <= 0 or maximum >= length:
        return torch.arange(length, dtype=torch.long)
    return torch.linspace(0, length - 1, steps=maximum).round().long().unique()


@torch.inference_mode()
def score_full_grid(
    model: nn.Module,
    values: Tensor,
    context: ContextFeatures | None,
    window: int,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
    max_time_points: int = 0,
    max_series: int = 0,
) -> Tensor:
    """Vectorized inference over time and series without per-window Python IO."""
    model.eval()
    scores = torch.full_like(values, torch.nan)
    relative_times = _evenly_spaced_indices(
        values.shape[0] - window, max_time_points
    )
    target_times = relative_times + window
    series_indices = _evenly_spaced_indices(values.shape[1], max_series)

    for target_time_tensor in target_times:
        target_time = int(target_time_tensor)
        history_slice = slice(target_time - window, target_time)
        for start in range(0, len(series_indices), batch_size):
            batch_indices = series_indices[start : start + batch_size]
            history = values[history_slice, batch_indices].transpose(0, 1)
            observed_history = torch.isfinite(history)
            input_parts = [
                torch.stack(
                    (
                        torch.nan_to_num(
                            history, nan=0.0, posinf=0.0, neginf=0.0
                        ),
                        observed_history.to(torch.float32),
                    ),
                    dim=-1,
                )
            ]
            if context is not None:
                global_history = context.global_time[history_slice]
                input_parts.append(
                    global_history.unsqueeze(0).expand(
                        len(batch_indices), -1, -1
                    )
                )
                if context.per_series is not None:
                    input_parts.append(
                        context.per_series[
                            history_slice, batch_indices
                        ].permute(1, 0, 2)
                    )
            inputs = torch.cat(input_parts, dim=-1).to(
                device, non_blocking=device.type == "cuda"
            )
            device_indices = batch_indices.to(
                device, non_blocking=device.type == "cuda"
            )
            with torch.amp.autocast(
                device_type=device.type, enabled=use_amp
            ):
                prediction = model(inputs, device_indices).squeeze(-1)

            target = values[target_time, batch_indices]
            observed_target = torch.isfinite(target)
            absolute_error = (
                prediction.float().cpu()
                - torch.nan_to_num(
                    target, nan=0.0, posinf=0.0, neginf=0.0
                )
            ).abs()
            scores[target_time, batch_indices] = torch.where(
                observed_target, absolute_error, torch.nan
            )
    return scores


def prepare_data(
    args: argparse.Namespace,
) -> tuple[
    Tensor,
    Tensor,
    list[str],
    Tensor,
    SplitSlices,
    MetadataFeatures | None,
    dict[str, Any],
    ContextFeatures | None,
    dict[str, Any],
]:
    loaded = load_wide_csv(args.csv)
    splits = chronological_splits(
        len(loaded.timestamps), args.train_ratio, args.validation_ratio
    )
    filter_result = filter_useful_series(
        loaded.values,
        splits.train,
        min_train_observations=args.min_train_observations,
        min_train_coverage=args.min_train_coverage,
        min_informative_observations=args.min_informative_observations,
    )
    keep = filter_result.keep
    values = loaded.values[:, keep]
    columns = [
        column for column, selected in zip(loaded.columns, keep.tolist(), strict=True)
        if selected
    ]

    metadata = None
    if args.metadata is not None:
        metadata = load_metadata_features(
            args.metadata,
            keep_series=keep,
            expected_series_count=len(loaded.columns),
        )

    baseline = build_baseline_features(
        values,
        splits,
        method=args.baseline,
        period=args.period,
        alpha=args.hw_alpha,
        beta=args.hw_beta,
        gamma=args.hw_gamma,
    )
    features = torch.clamp(
        baseline.features, min=-args.clip_value, max=args.clip_value
    )
    context = None
    if args.architecture == "context_tcn":
        context = build_context_features(
            features,
            loaded.timestamps,
            metadata,
        )
    return (
        features,
        values,
        columns,
        loaded.timestamps,
        splits,
        metadata,
        baseline.state,
        context,
        filter_result.report,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = choose_device(args.device)
    use_amp = args.amp and device.type == "cuda"

    (
        features,
        raw_values,
        columns,
        timestamps,
        splits,
        metadata,
        baseline_state,
        context,
        series_filter_report,
    ) = prepare_data(args)
    train_dataset = WindowDataset(
        features,
        splits.train,
        window=args.window,
        max_samples=args.samples_per_epoch,
        random_sampling=True,
        seed=args.seed,
        context=context,
    )
    validation_dataset = WindowDataset(
        features,
        splits.validation,
        window=args.window,
        max_samples=args.evaluation_samples,
        random_sampling=False,
        seed=args.seed,
        context=context,
    )
    test_dataset = WindowDataset(
        features,
        splits.test,
        window=args.window,
        max_samples=args.evaluation_samples,
        random_sampling=False,
        seed=args.seed,
        context=context,
    )
    train_loader = make_loader(
        train_dataset, args.batch_size, args.num_workers, device
    )
    validation_loader = make_loader(
        validation_dataset, args.batch_size, args.num_workers, device
    )
    test_loader = make_loader(
        test_dataset, args.batch_size, args.num_workers, device
    )

    model_config = ModelConfig(
        architecture=args.architecture,
        input_size=train_dataset.input_size,
        hidden_size=args.hidden_size,
        layers=args.layers,
        dropout=args.dropout,
        tcn_blocks=args.tcn_blocks,
    )
    model = build_model(model_config, metadata=metadata).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_state: dict[str, Tensor] | None = None
    best_validation_mae = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    print(
        f"device={device} rows={features.shape[0]} series={features.shape[1]} "
        f"splits={splits.as_dict()} metadata={metadata is not None} "
        f"architecture={args.architecture} input_size={train_dataset.input_size}"
    )
    print(json.dumps({"series_filter": series_filter_report}))
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_loss = run_training_epoch(
            model,
            train_loader,
            optimizer,
            amp_scaler,
            device,
            args.gradient_clip,
            use_amp,
        )
        validation_metrics, _ = evaluate(
            model, validation_loader, device, use_amp
        )
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_mae": validation_metrics["mae"],
            "validation_rmse": validation_metrics["rmse"],
        }
        history.append(row)
        print(json.dumps(row))

        if validation_metrics["mae"] < best_validation_mae:
            best_validation_mae = validation_metrics["mae"]
            best_epoch = epoch
            best_state = cpu_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early_stopping epoch={epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    validation_metrics, validation_errors = evaluate(
        model, validation_loader, device, use_amp
    )
    test_metrics, test_errors = evaluate(model, test_loader, device, use_amp)
    train_metrics, _ = evaluate(model, train_loader, device, use_amp)
    threshold = float(torch.quantile(validation_errors, args.threshold_quantile))
    grid_scores = None
    grid_summary = None
    if args.full_score:
        grid_scores = score_full_grid(
            model,
            features,
            context,
            window=args.window,
            device=device,
            batch_size=args.score_batch_size,
            use_amp=use_amp,
            max_time_points=args.score_max_time_points,
            max_series=args.score_max_series,
        )
        finite_grid_validation = grid_scores[splits.validation]
        finite_grid_validation = finite_grid_validation[
            torch.isfinite(finite_grid_validation)
        ]
        if finite_grid_validation.numel():
            threshold = float(
                torch.quantile(
                    finite_grid_validation, args.threshold_quantile
                )
            )
    test_metrics["anomaly_threshold"] = threshold
    test_metrics["threshold_exceedance_rate"] = float(
        (test_errors > threshold).float().mean()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved_arguments = vars(args).copy()
    saved_arguments["csv"] = str(args.csv)
    saved_arguments["metadata"] = (
        str(args.metadata) if args.metadata is not None else None
    )
    saved_arguments["output_dir"] = str(args.output_dir)
    checkpoint = {
        "model_state": best_state,
        "model_config": model_config.as_dict(),
        "baseline_state": baseline_state,
        "series_columns": columns,
        "timestamps": {
            "first": int(timestamps[0]),
            "last": int(timestamps[-1]),
        },
        "splits": splits.as_dict(),
        "series_filter": series_filter_report,
        "arguments": saved_arguments,
        "best_epoch": best_epoch,
        "threshold": {
            "quantile": args.threshold_quantile,
            "value": threshold,
        },
        "context_feature_names": context.names if context is not None else (),
    }
    torch.save(checkpoint, args.output_dir / "checkpoint.pt")

    metrics = {
        "best_epoch": best_epoch,
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
        "series_filter": series_filter_report,
    }
    if grid_scores is not None:
        grid_summary = save_model_anomaly_report(
            args.output_dir,
            scores=grid_scores,
            threshold=threshold,
            timestamps=timestamps,
            raw_values=raw_values,
            series_columns=columns,
            validation_slice=splits.validation,
            test_slice=splits.test,
            series_filter_report=series_filter_report,
        )
        metrics["full_grid_anomaly"] = grid_summary
        torch.save(
            {
                "scores": grid_scores,
                "threshold": threshold,
                "timestamps": timestamps,
                "series_columns": columns,
                "series_filter": series_filter_report,
            },
            args.output_dir / "full_grid_scores.pt",
        )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    save_training_report(
        args.output_dir,
        history=history,
        split_metrics={
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
        },
        validation_errors=validation_errors,
        test_errors=test_errors,
        threshold=threshold,
    )
    print(json.dumps({"best_epoch": best_epoch, "test": test_metrics}))


if __name__ == "__main__":
    main()
