from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor


def _finite_numpy(values: Tensor | Sequence[float]) -> np.ndarray:
    if isinstance(values, Tensor):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def error_metrics(errors: Tensor | Sequence[float]) -> dict[str, float | int]:
    absolute = _finite_numpy(errors)
    if absolute.size == 0:
        return {"count": 0}
    return {
        "count": int(absolute.size),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(absolute)))),
        "median_absolute_error": float(np.quantile(absolute, 0.50)),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "p99_absolute_error": float(np.quantile(absolute, 0.99)),
        "maximum_absolute_error": float(absolute.max()),
    }


def score_metrics(scores: Tensor) -> dict[str, float | int]:
    finite = _finite_numpy(scores)
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "median": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "maximum": float(finite.max()),
    }


def _save_figure(figure: Any, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    figure.clf()


def save_training_report(
    output_dir: str | Path,
    history: list[dict[str, float | int]],
    split_metrics: dict[str, dict[str, float]],
    validation_errors: Tensor,
    test_errors: Tensor,
    threshold: float,
) -> Path:
    """Write JSON, Markdown, and PNG diagnostics for one training run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_dir = Path(output_dir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    validation_summary = error_metrics(validation_errors)
    test_summary = error_metrics(test_errors)
    summary: dict[str, Any] = {
        "split_metrics": split_metrics,
        "validation_error_summary": validation_summary,
        "test_error_summary": test_summary,
        "anomaly_threshold": threshold,
        "history": history,
    }

    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o")
    axes[0].set_ylabel("Smooth L1 train loss")
    axes[0].set_title("Training history")
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        epochs,
        [row["validation_mae"] for row in history],
        marker="o",
        label="validation MAE",
    )
    axes[1].plot(
        epochs,
        [row["validation_rmse"] for row in history],
        marker="o",
        label="validation RMSE",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    _save_figure(figure, report_dir / "loss_curve.png")
    plt.close(figure)

    labels = list(split_metrics)
    x = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        x - width / 2,
        [split_metrics[label]["mae"] for label in labels],
        width,
        label="MAE",
    )
    axis.bar(
        x + width / 2,
        [split_metrics[label]["rmse"] for label in labels],
        width,
        label="RMSE",
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Standardized residual error")
    axis.set_title("Split metrics")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, report_dir / "split_metrics.png")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for axis, errors, label in (
        (axes[0], validation_errors, "Validation"),
        (axes[1], test_errors, "Test"),
    ):
        finite = _finite_numpy(errors)
        if finite.size:
            finite = finite[:200_000]
            axis.hist(np.log1p(finite), bins=60, color="#4472c4", alpha=0.85)
        axis.set_title(f"{label} absolute errors")
        axis.set_xlabel("log(1 + absolute error)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Count")
    _save_figure(figure, report_dir / "error_histograms.png")
    plt.close(figure)

    (report_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report_lines = [
        "# Training report",
        "",
        f"Anomaly threshold (validation quantile): `{threshold:.6g}`",
        "",
        "| Split | MAE | RMSE | Count |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, metrics in split_metrics.items():
        report_lines.append(
            f"| {label} | {metrics['mae']:.6g} | "
            f"{metrics['rmse']:.6g} | {int(metrics['samples'])} |"
        )
    report_lines += [
        "",
        "## Error metrics",
        "",
        "| Split | Median AE | P95 AE | P99 AE | Max AE |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| validation | {validation_summary.get('median_absolute_error', 0):.6g} | "
        f"{validation_summary.get('p95_absolute_error', 0):.6g} | "
        f"{validation_summary.get('p99_absolute_error', 0):.6g} | "
        f"{validation_summary.get('maximum_absolute_error', 0):.6g} |",
        f"| test | {test_summary.get('median_absolute_error', 0):.6g} | "
        f"{test_summary.get('p95_absolute_error', 0):.6g} | "
        f"{test_summary.get('p99_absolute_error', 0):.6g} | "
        f"{test_summary.get('maximum_absolute_error', 0):.6g} |",
        "",
        "## Figures",
        "",
        "- `loss_curve.png` — train loss and validation MAE/RMSE.",
        "- `split_metrics.png` — MAE/RMSE by chronological split.",
        "- `error_histograms.png` — log-scaled validation/test error distributions.",
    ]
    (report_dir / "report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return report_dir


def save_baseline_report(
    output_dir: str | Path,
    summaries: dict[str, dict[str, dict[str, float | int]]],
    scores_by_method: dict[str, Tensor],
    thresholds: dict[str, float],
    timestamps: Tensor,
    validation_slice: slice,
    test_slice: slice,
    series_filter_report: dict[str, Any] | None = None,
) -> Path:
    """Write baseline summary JSON/Markdown and score distribution plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "methods": summaries,
        "thresholds": thresholds,
    }
    if series_filter_report is not None:
        summary["series_filter"] = series_filter_report

    timestamp_values = timestamps.detach().cpu().numpy().astype(np.int64)
    anomaly_rates: dict[str, np.ndarray] = {}
    anomaly_counts: dict[str, np.ndarray] = {}
    finite_counts: dict[str, np.ndarray] = {}
    for method, scores in scores_by_method.items():
        finite = torch.isfinite(scores)
        flagged = finite & (scores > thresholds[method])
        finite_count = finite.sum(dim=1)
        flagged_count = flagged.sum(dim=1)
        rate = torch.full_like(finite_count, torch.nan, dtype=torch.float32)
        valid = finite_count > 0
        rate[valid] = 100.0 * flagged_count[valid] / finite_count[valid]
        anomaly_rates[method] = rate.cpu().numpy()
        anomaly_counts[method] = flagged_count.cpu().numpy()
        finite_counts[method] = finite_count.cpu().numpy()

    time_summary: dict[str, Any] = {}
    for method, rates in anomaly_rates.items():
        finite_rates = rates[np.isfinite(rates)]
        peak_index = int(np.nanargmax(rates)) if finite_rates.size else None
        time_summary[method] = {
            "mean_percent": float(np.nanmean(rates)),
            "median_percent": float(np.nanmedian(rates)),
            "p95_percent": float(np.nanquantile(rates, 0.95)),
            "maximum_percent": float(np.nanmax(rates)),
            "peak_timestamp": (
                int(timestamp_values[peak_index]) if peak_index is not None else None
            ),
        }
    summary["full_week_anomaly_percent"] = time_summary

    time_series_payload = {
        "timestamps": timestamp_values.tolist(),
        "methods": {
            method: {
                "anomaly_percent": anomaly_rates[method].tolist(),
                "anomalous_series_count": anomaly_counts[method].tolist(),
                "finite_series_count": finite_counts[method].tolist(),
                "threshold": thresholds[method],
            }
            for method in scores_by_method
        },
    }
    (report_dir / "anomaly_rate_by_time.json").write_text(
        json.dumps(time_series_payload) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, len(scores_by_method), figsize=(6 * len(scores_by_method), 4.5))
    axes = np.atleast_1d(axes)
    for axis, (method, scores) in zip(axes, scores_by_method.items(), strict=True):
        validation = _finite_numpy(scores[validation_slice])[:200_000]
        test = _finite_numpy(scores[test_slice])[:200_000]
        if validation.size:
            axis.hist(np.log1p(validation), bins=60, alpha=0.6, label="validation")
        if test.size:
            axis.hist(np.log1p(test), bins=60, alpha=0.6, label="test")
        axis.axvline(
            np.log1p(thresholds[method]), color="black", linestyle="--", label="threshold"
        )
        axis.set_title(method)
        axis.set_xlabel("log(1 + absolute score)")
        axis.set_ylabel("Count")
        axis.legend()
        axis.grid(alpha=0.2)
    _save_figure(figure, report_dir / "baseline_score_histograms.png")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 4.8))
    time_axis = timestamp_values.astype("datetime64[s]")
    for method, rates in anomaly_rates.items():
        axis.plot(time_axis, rates, linewidth=1.2, label=method)
    axis.axvline(
        time_axis[validation_slice.start],
        color="black",
        linestyle="--",
        linewidth=0.9,
        label="validation start",
    )
    axis.axvline(
        time_axis[test_slice.start],
        color="black",
        linestyle=":",
        linewidth=0.9,
        label="test start",
    )
    axis.set_title("Percentage of finite series flagged as anomalous")
    axis.set_xlabel("UTC timestamp")
    axis.set_ylabel("Anomalous series (%)")
    axis.set_ylim(bottom=0)
    axis.legend()
    axis.grid(alpha=0.25)
    figure.autofmt_xdate()
    _save_figure(figure, report_dir / "anomaly_rate_by_time.png")
    plt.close(figure)

    (report_dir / "baseline_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Baseline report",
        "",
    ]
    if series_filter_report is not None:
        reasons = series_filter_report["removed_reasons"]
        lines += [
            "## Series quality filter",
            "",
            f"Retained `{series_filter_report['retained_series']}` of "
            f"`{series_filter_report['input_series']}` series using training "
            "data only.",
            "",
            "| Reason | Removed |",
            "| --- | ---: |",
            f"| Insufficient coverage | "
            f"{reasons['insufficient_train_observations']} |",
            f"| Always zero | {reasons['always_zero']} |",
            f"| Constant or numerically flat | "
            f"{reasons['constant_or_numerically_flat']} |",
            f"| Too few informative values | "
            f"{reasons['too_few_informative_values']} |",
            "",
        ]
    lines += [
        "## Full-week anomaly rate",
        "",
        "A series is flagged when its score is above the threshold calibrated "
        "on validation data. Percentages use only finite series at each timestamp.",
        "",
        "| Method | Mean % | Median % | P95 % | Max % | Peak timestamp |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for method, metrics in time_summary.items():
        lines.append(
            f"| {method} | {metrics['mean_percent']:.6g} | "
            f"{metrics['median_percent']:.6g} | {metrics['p95_percent']:.6g} | "
            f"{metrics['maximum_percent']:.6g} | {metrics['peak_timestamp']} |"
        )
    lines += [
        "",
        "| Method | Split | Median | P95 | P99 | Max |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for method, split_summaries in summaries.items():
        for split, metrics in split_summaries.items():
            lines.append(
                f"| {method} | {split} | {metrics.get('median', 0):.6g} | "
                f"{metrics.get('p95', 0):.6g} | {metrics.get('p99', 0):.6g} | "
                f"{metrics.get('maximum', 0):.6g} |"
            )
    lines += [
        "",
        "- `baseline_score_histograms.png` — validation/test score distributions.",
        "- `anomaly_rate_by_time.png` — full-week anomaly percentage by timestamp.",
        "- `anomaly_rate_by_time.json` — timestamp-level counts and percentages.",
    ]
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_dir


def _rate_summary(
    rates: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, float | int | None]:
    finite = np.isfinite(rates)
    if not finite.any():
        return {
            "evaluated_timestamps": 0,
            "mean_percent": None,
            "median_percent": None,
            "p95_percent": None,
            "maximum_percent": None,
            "peak_timestamp": None,
        }
    finite_rates = rates[finite]
    peak_index = int(np.nanargmax(rates))
    return {
        "evaluated_timestamps": int(finite.sum()),
        "mean_percent": float(finite_rates.mean()),
        "median_percent": float(np.median(finite_rates)),
        "p95_percent": float(np.quantile(finite_rates, 0.95)),
        "maximum_percent": float(finite_rates.max()),
        "peak_timestamp": int(timestamps[peak_index]),
    }


def save_model_anomaly_report(
    output_dir: str | Path,
    scores: Tensor,
    threshold: float,
    timestamps: Tensor,
    raw_values: Tensor,
    series_columns: list[str],
    validation_slice: slice,
    test_slice: slice,
    sample_count: int = 10,
    sample_seed: int | None = None,
    series_filter_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Report the learned model's anomaly percentage and example series."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_dir = Path(output_dir) / "model_anomaly_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    finite = torch.isfinite(scores)
    anomalous = finite & (scores > threshold)
    finite_count = finite.sum(dim=1)
    anomaly_count = anomalous.sum(dim=1)
    rates = torch.full_like(finite_count, torch.nan, dtype=torch.float32)
    valid_time = finite_count > 0
    rates[valid_time] = (
        100.0 * anomaly_count[valid_time] / finite_count[valid_time]
    )

    timestamp_values = timestamps.cpu().numpy().astype(np.int64)
    rate_values = rates.cpu().numpy()
    summary: dict[str, Any] = {
        "threshold": threshold,
        "full_week": _rate_summary(rate_values, timestamp_values),
        "train": _rate_summary(
            rate_values[: validation_slice.start],
            timestamp_values[: validation_slice.start],
        ),
        "validation": _rate_summary(
            rate_values[validation_slice],
            timestamp_values[validation_slice],
        ),
        "test": _rate_summary(
            rate_values[test_slice],
            timestamp_values[test_slice],
        ),
    }
    if series_filter_report is not None:
        summary["series_filter"] = series_filter_report

    time_payload = {
        "timestamps": timestamp_values.tolist(),
        "anomaly_percent": [
            float(value) if np.isfinite(value) else None for value in rate_values
        ],
        "anomalous_series_count": anomaly_count.cpu().tolist(),
        "finite_series_count": finite_count.cpu().tolist(),
        "threshold": threshold,
    }
    if series_filter_report is not None:
        time_payload["series_filter"] = series_filter_report
    (report_dir / "anomaly_rate_by_time.json").write_text(
        json.dumps(time_payload) + "\n", encoding="utf-8"
    )

    time_axis = timestamp_values.astype("datetime64[s]")
    figure, axis = plt.subplots(figsize=(12, 4.8))
    axis.plot(
        time_axis,
        rate_values,
        linewidth=1.25,
        marker=".",
        markersize=2.5,
        color="#c23b22",
    )
    axis.axvline(
        time_axis[validation_slice.start],
        color="black",
        linestyle="--",
        linewidth=0.9,
        label="validation start",
    )
    axis.axvline(
        time_axis[test_slice.start],
        color="black",
        linestyle=":",
        linewidth=0.9,
        label="test start",
    )
    axis.set_title("Learned-model anomaly percentage by timestamp")
    axis.set_xlabel("UTC timestamp")
    axis.set_ylabel("Anomalous finite series (%)")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.autofmt_xdate()
    _save_figure(figure, report_dir / "anomaly_rate_by_time.png")
    plt.close(figure)

    eligible = (
        torch.isfinite(scores).any(dim=0)
        & torch.isfinite(raw_values).any(dim=0)
    ).nonzero(as_tuple=False).squeeze(1)
    if sample_seed is None:
        sample_seed = secrets.randbits(63)
    generator = torch.Generator().manual_seed(sample_seed)
    if len(eligible) > sample_count:
        selection = eligible[
            torch.randperm(len(eligible), generator=generator)[:sample_count]
        ]
    else:
        selection = eligible
    summary["sample_selection_seed"] = sample_seed
    summary["sample_series"] = [
        {
            "index": int(index),
            "column": series_columns[int(index)],
        }
        for index in selection
    ]

    if len(selection):
        sample_dir = report_dir / "sample_series"
        sample_dir.mkdir(parents=True, exist_ok=True)
        raw_numpy = raw_values.cpu().numpy()
        score_numpy = scores.cpu().numpy()
        for sample_number, series_index_tensor in enumerate(selection, start=1):
            series_index = int(series_index_tensor)
            column = series_columns[series_index]
            figure, axes = plt.subplots(
                2, 1, figsize=(12, 7), sharex=True
            )
            axes[0].plot(
                time_axis,
                raw_numpy[:, series_index],
                linewidth=0.9,
                color="#4472c4",
            )
            axes[0].set_title(f"{column}: raw values")
            axes[0].set_ylabel("Raw value")
            axes[0].grid(alpha=0.2)
            axes[1].plot(
                time_axis,
                score_numpy[:, series_index],
                linewidth=0.9,
                marker=".",
                markersize=2.0,
                color="#c23b22",
            )
            axes[1].axhline(
                threshold,
                color="black",
                linestyle="--",
                linewidth=0.9,
                label="threshold",
            )
            axes[1].set_title(f"{column}: anomaly score")
            axes[1].set_xlabel("UTC timestamp")
            axes[1].set_ylabel("Score")
            axes[1].grid(alpha=0.2)
            axes[1].legend(loc="upper right")
            figure.autofmt_xdate()
            _save_figure(
                figure,
                sample_dir / f"{sample_number:02d}_{column}.png",
            )
            plt.close(figure)

    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Learned-model anomaly report",
        "",
        f"Validation-calibrated threshold: `{threshold:.6g}`",
        "",
    ]
    if series_filter_report is not None:
        lines += [
            f"Percentages use the `{series_filter_report['retained_series']}` "
            "series retained by the training-only quality filter "
            f"(from `{series_filter_report['input_series']}` input series).",
            "",
        ]
    lines += [
        "| Split | Evaluated timestamps | Mean % | Median % | P95 % | Max % |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("full_week", "train", "validation", "test"):
        metrics = summary[split]
        lines.append(
            f"| {split} | {metrics['evaluated_timestamps']} | "
            f"{metrics['mean_percent'] or 0:.6g} | "
            f"{metrics['median_percent'] or 0:.6g} | "
            f"{metrics['p95_percent'] or 0:.6g} | "
            f"{metrics['maximum_percent'] or 0:.6g} |"
        )
    lines += [
        "",
        "## Figures",
        "",
        "- `anomaly_rate_by_time.png` — percentage of series flagged per timestamp.",
        "- `sample_series/*.png` — ten newly randomized raw-series and score plots.",
        "- `anomaly_rate_by_time.json` — exact counts and percentages.",
    ]
    (report_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def save_panel_factor_report(
    output_dir: str | Path,
    history: list[dict[str, float | int]],
    split_metrics: dict[str, dict[str, float | int]],
    validation_inference_loss: Sequence[float],
    test_inference_loss: Sequence[float],
) -> Path:
    """Save compact optimization and reconstruction diagnostics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_dir = Path(output_dir) / "panel_factor_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(
        epochs,
        [float(row["reconstruction_loss"]) for row in history],
        marker="o",
        color="#4472c4",
    )
    axes[0].set_ylabel("Huber reconstruction loss")
    axes[0].set_title("Robust panel-factor training")
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        epochs,
        [float(row["regularization"]) for row in history],
        marker="o",
        color="#70ad47",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Temporal regularization")
    axes[1].grid(alpha=0.25)
    _save_figure(figure, report_dir / "training_objective.png")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        np.arange(1, len(validation_inference_loss) + 1),
        validation_inference_loss,
        label="validation",
    )
    axis.plot(
        np.arange(1, len(test_inference_loss) + 1),
        test_inference_loss,
        label="test",
    )
    axis.set_title("Frozen-loading latent-state inference")
    axis.set_xlabel("Inference step")
    axis.set_ylabel("Objective")
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(figure, report_dir / "inference_convergence.png")
    plt.close(figure)

    labels = list(split_metrics)
    x = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        x - width / 2,
        [float(split_metrics[label].get("mae", 0.0)) for label in labels],
        width,
        label="MAE",
    )
    axis.bar(
        x + width / 2,
        [float(split_metrics[label].get("rmse", 0.0)) for label in labels],
        width,
        label="RMSE",
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Combined anomaly score")
    axis.set_title("Full-grid score summary")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save_figure(figure, report_dir / "split_score_metrics.png")
    plt.close(figure)

    summary = {
        "history": history,
        "split_metrics": split_metrics,
        "validation_inference_loss": list(validation_inference_loss),
        "test_inference_loss": list(test_inference_loss),
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Robust panel-factor report",
        "",
        "The model learns relationships across all series jointly. Its final "
        "cell score combines robust reconstruction error with the effect of an "
        "unexpected shared latent-state change.",
        "",
        "| Split | Mean score | RMS score | Median | P95 | P99 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, metrics in split_metrics.items():
        lines.append(
            f"| {label} | {float(metrics.get('mae', 0)):.6g} | "
            f"{float(metrics.get('rmse', 0)):.6g} | "
            f"{float(metrics.get('median_absolute_error', 0)):.6g} | "
            f"{float(metrics.get('p95_absolute_error', 0)):.6g} | "
            f"{float(metrics.get('p99_absolute_error', 0)):.6g} |"
        )
    lines += [
        "",
        "## Figures",
        "",
        "- `training_objective.png` — reconstruction and temporal losses.",
        "- `inference_convergence.png` — validation/test latent-state fitting.",
        "- `split_score_metrics.png` — full-grid mean and RMS scores.",
    ]
    (report_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report_dir
