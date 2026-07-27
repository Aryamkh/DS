from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print compact EDA without writing CSV files."
    )
    parser.add_argument("--csv", type=Path, default=Path("final.csv"))
    parser.add_argument(
        "--metadata", type=Path, default=Path("metadata-hashed.csv")
    )
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--display-series", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--min-train-coverage", type=float, default=0.8)
    parser.add_argument(
        "--min-informative-observations", type=int, default=12
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.csv, dtype={"timestamp": np.int64})
    display_frame = frame.iloc[
        :, : min(args.display_series + 1, frame.shape[1])
    ]
    timestamps = frame.pop("timestamp").to_numpy(dtype=np.int64)
    values = frame.to_numpy(dtype=np.float32)
    finite = np.isfinite(values)
    valid_series = finite.any(axis=0)
    missing_per_series = (~finite).sum(axis=0)
    zero_count = ((values == 0.0) & finite).sum()

    minimum = np.full(values.shape[1], np.nan, dtype=np.float32)
    maximum = minimum.copy()
    minimum[valid_series] = np.nanmin(values[:, valid_series], axis=0)
    maximum[valid_series] = np.nanmax(values[:, valid_series], axis=0)
    finite_ranges = (maximum - minimum)[valid_series]

    train_end = int(len(frame) * args.train_ratio)
    validation_end = train_end + int(
        len(frame) * args.validation_ratio
    )
    train = values[:train_end]
    train_finite = np.isfinite(train)
    train_count = train_finite.sum(axis=0)
    required_count = max(
        288, int(np.ceil(train_end * args.min_train_coverage))
    )
    enough = train_count >= required_count
    train_minimum = np.min(
        np.where(train_finite, train, np.inf), axis=0
    )
    train_maximum = np.max(
        np.where(train_finite, train, -np.inf), axis=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        train_center = np.nanmedian(train, axis=0)
    tolerance = np.maximum(1e-8, np.abs(train_center) * 1e-6)
    variable = (
        np.isfinite(train_maximum - train_minimum)
        & ((train_maximum - train_minimum) > tolerance)
    )
    informative_count = (
        train_finite & (np.abs(train - train_center) > tolerance)
    ).sum(axis=0)
    informative = informative_count >= args.min_informative_observations
    train_maximum_absolute = np.max(
        np.where(train_finite, np.abs(train), 0.0), axis=0
    )
    always_zero = enough & (train_maximum_absolute <= 1e-8)
    constant = enough & ~always_zero & ~variable
    too_few_informative = enough & variable & ~informative
    useful = enough & variable & informative
    print(
        {
            "rows": len(frame),
            "series": values.shape[1],
            "first_timestamp": int(timestamps[0]),
            "last_timestamp": int(timestamps[-1]),
            "step_seconds": np.unique(np.diff(timestamps)).tolist(),
            "split_rows": {
                "train": train_end,
                "validation": validation_end - train_end,
                "test": len(frame) - validation_end,
            },
        }
    )
    print(
        {
            "train_only_series_filter": {
                "retained": int(useful.sum()),
                "removed": int((~useful).sum()),
                "required_train_observations": required_count,
                "insufficient_train_observations": int((~enough).sum()),
                "always_zero": int(always_zero.sum()),
                "constant_or_numerically_flat": int(constant.sum()),
                "too_few_informative_values": int(
                    too_few_informative.sum()
                ),
            }
        }
    )
    print(
        {
            "missing_rate": float((~finite).mean()),
            "all_missing_series": int((~valid_series).sum()),
            "series_with_missing": int((missing_per_series > 0).sum()),
            "missing_per_series_quantiles": np.quantile(
                missing_per_series, [0.0, 0.5, 0.9, 0.99, 1.0]
            ).tolist(),
            "zero_rate_among_finite": float(zero_count / finite.sum()),
            "constant_finite_series": int(
                np.isclose(
                    minimum[valid_series], maximum[valid_series]
                ).sum()
            ),
            "finite_range_quantiles": np.quantile(
                finite_ranges, [0.0, 0.5, 0.9, 0.99, 1.0]
            ).tolist(),
        }
    )

    print("\ndata head")
    print(display_frame.head(args.rows).to_string(index=False))
    print("\ndata tail")
    print(display_frame.tail(args.rows).to_string(index=False))

    if args.metadata.exists():
        metadata = pd.read_csv(args.metadata)
        print("\nmetadata summary")
        print(
            {
                "rows": len(metadata),
                "unique_group_tags": int(metadata["group_tag"].nunique()),
                "unique_query_tags": int(metadata["query_tag"].nunique()),
                "unique_label_tags": int(metadata["label_tag"].nunique()),
                "sample_count_quantiles": metadata["sample_count"]
                .quantile([0.0, 0.5, 0.9, 1.0])
                .to_dict(),
            }
        )
        print("\nmetadata head")
        print(metadata.head(args.rows).to_string(index=False))
        print("\nmetadata tail")
        print(metadata.tail(args.rows).to_string(index=False))


if __name__ == "__main__":
    main()
