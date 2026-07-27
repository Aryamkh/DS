from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor

from .baselines import build_baseline_features
from .data import (
    chronological_splits,
    filter_useful_series,
    load_wide_csv,
)
from .reporting import save_baseline_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate seasonal MAD and Holt-Winters scores."
    )
    parser.add_argument("--csv", type=Path, default=Path("final.csv"))
    parser.add_argument(
        "--method",
        choices=("seasonal_mad", "holt_winters", "both"),
        default="both",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--period", type=int, default=288)
    parser.add_argument("--min-train-observations", type=int, default=288)
    parser.add_argument("--min-train-coverage", type=float, default=0.8)
    parser.add_argument(
        "--min-informative-observations", type=int, default=12
    )
    parser.add_argument("--hw-alpha", type=float, default=0.2)
    parser.add_argument("--hw-beta", type=float, default=0.02)
    parser.add_argument("--hw-gamma", type=float, default=0.05)
    parser.add_argument("--threshold-quantile", type=float, default=0.995)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("artifacts/baseline_report"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional .pt output; no CSV files are written.",
    )
    return parser.parse_args()


def summarize(scores: Tensor, name: str) -> dict[str, float | int | str]:
    finite = scores[torch.isfinite(scores)]
    if finite.numel() == 0:
        return {"split": name, "count": 0}
    quantiles = torch.quantile(
        finite, torch.tensor([0.5, 0.95, 0.99], dtype=finite.dtype)
    )
    return {
        "split": name,
        "count": finite.numel(),
        "median": float(quantiles[0]),
        "p95": float(quantiles[1]),
        "p99": float(quantiles[2]),
        "maximum": float(finite.max()),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold_quantile < 1.0:
        raise ValueError("--threshold-quantile must be between zero and one.")

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
    print({"series_filter": filter_result.report})
    values = loaded.values[:, keep]
    methods = (
        ("seasonal_mad", "holt_winters")
        if args.method == "both"
        else (args.method,)
    )

    saved: dict[str, object] = {
        "timestamps": loaded.timestamps,
        "series_columns": [
            column
            for column, selected in zip(
                loaded.columns, keep.tolist(), strict=True
            )
            if selected
        ],
        "splits": splits.as_dict(),
        "series_filter": filter_result.report,
    }
    summaries: dict[str, dict[str, dict[str, float | int]]] = {}
    scores_by_method: dict[str, Tensor] = {}
    thresholds: dict[str, float] = {}
    for method in methods:
        output = build_baseline_features(
            values,
            splits,
            method=method,
            period=args.period,
            alpha=args.hw_alpha,
            beta=args.hw_beta,
            gamma=args.hw_gamma,
        )
        scores = output.features.abs()
        validation_scores = scores[splits.validation]
        finite_validation = validation_scores[
            torch.isfinite(validation_scores)
        ]
        threshold = float(
            torch.quantile(finite_validation, args.threshold_quantile)
        )
        test_scores = scores[splits.test]
        finite_test = test_scores[torch.isfinite(test_scores)]
        print(f"\n{method}")
        summaries[method] = {}
        for split_name, split_slice in (
            ("train", splits.train),
            ("validation", splits.validation),
            ("test", splits.test),
        ):
            split_summary = summarize(scores[split_slice], split_name)
            print(split_summary)
            summaries[method][split_name] = split_summary
        print(
            {
                "threshold_quantile": args.threshold_quantile,
                "threshold": threshold,
                "test_exceedance_rate": float(
                    (finite_test > threshold).float().mean()
                ),
            }
        )
        saved[method] = {
            "scores": scores,
            "state": output.state,
            "threshold": threshold,
        }
        scores_by_method[method] = scores
        thresholds[method] = threshold

    save_baseline_report(
        args.report_dir,
        summaries=summaries,
        scores_by_method=scores_by_method,
        thresholds=thresholds,
        timestamps=loaded.timestamps,
        validation_slice=splits.validation,
        test_slice=splits.test,
        series_filter_report=filter_result.report,
    )

    if args.output is not None:
        if args.output.suffix != ".pt":
            raise ValueError("--output must use the .pt extension.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved, args.output)


if __name__ == "__main__":
    main()
