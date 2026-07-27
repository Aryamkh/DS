# Final project run report

## Scope

The data contains 2,017 timestamps at five-minute resolution and 4,508
candidate series. Runs use chronological 80/10/10 train, validation, and test
splits. The pipeline was tested in Modal with CUDA on an NVIDIA L40S.

The experiments covered:

- Seasonal median/MAD residual features.
- Holt-Winters residual features.
- A global GRU/context-TCN residual model.
- A metadata-aware low-rank panel-factor model designed for short, very wide
  panels. It learns relationships between series jointly and adds a collective
  latent-state anomaly component.
- A training-only quality filter for empty, constant, always-zero, severely
  missing, and nearly uninformative series.

On the latest filtered baseline run, Holt-Winters produced a lower full-week
anomaly rate than seasonal MAD: mean 0.2343% versus 0.3417%, median 0.0963%
versus 0.1927%, and P95 0.5780% versus 1.0597%. The thresholds are calibrated
separately, so the raw threshold magnitudes are not directly comparable.

The final quality-filter configuration retained 2,076 of 4,508 series. It
removed 2,057 always-zero series, 264 constant/flat series, 72 series without
80% training coverage, and 39 with too few informative values. No test or
validation values were used to decide this filter.

## Selected runs

These are the three retained runs. Their retained Markdown reports and plots
are under
`results/non_smoke/completed/<run-id>/`.

| Run | Model | Series | Test MAE | Test RMSE | Full-week mean anomalous % | Full-week P95 % | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `20260727T225532Z_full_seed17_d4df6add` | Context-TCN + metadata | 2,076 | 1.3613 | 3.7962 | 0.4387 | 1.3006 | Latest filtered run; sampled test metrics and full-grid anomaly timeline |
| `20260727T225453Z_full_seed17_1883abd2` | Metadata panel-factor | 2,076 | 4.3846 | 7.1915 | 0.4770 | 2.6108 | Latest filtered full-grid panel model; includes collective movement scoring |
| `20260727T224639Z_full_seed17_ce67d937` | Metadata panel-factor | 4,436 | 2.1928 | 5.0705 | 0.3575 | 1.5104 | Strong pre-filter full-grid reference across more series |

The Context-TCN test metrics are evaluated on approximately 200,000 sampled
test cells. The panel-factor metrics are evaluated on the complete test grid;
therefore their raw MAE/RMSE values should not be compared as if they were the
same evaluation protocol. The timestamp-level anomaly percentages are the
more useful comparison for the final detection output.

## Main findings

- The panel-factor approach directly models cross-series relationships instead
  of treating each series as an isolated sequence.
- Metadata-conditioned loadings provide a compact way to use group, query,
  label, and sample-count information with thousands of columns.
- Filtering constant and unusable columns removes a large source of degenerate
  zero/MAD behavior while retaining the variable series that can contribute
  meaningful anomalies.
- The final output is a score for every observed cell plus an exact anomaly
  percentage for every timestamp. Collective latent-state changes are included
  so a system-wide shift is not automatically absorbed as normal behavior.
- Every selected run retains its Markdown loss/metric reports and PNG plots.
  JSON files, PyTorch checkpoints, and score grids were deliberately removed.
  This report intentionally embeds no images; select them from the run
  directories as needed.

## Recommended final reference

Use `20260727T225532Z_full_seed17_d4df6add` for the latest filtered temporal
model and `20260727T225453Z_full_seed17_1883abd2` for the latest filtered
cross-sectional panel model. Keep `20260727T224639Z_full_seed17_ce67d937` as
the broader pre-filter comparison.
