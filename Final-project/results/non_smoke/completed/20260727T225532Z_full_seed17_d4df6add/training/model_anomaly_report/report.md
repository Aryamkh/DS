# Learned-model anomaly report

Validation-calibrated threshold: `20.0026`

Percentages use the `2076` series retained by the training-only quality filter (from `4508` input series).

| Split | Evaluated timestamps | Mean % | Median % | P95 % | Max % |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_week | 1921 | 0.438678 | 0.289017 | 1.30058 | 4.09441 |
| train | 1517 | 0.420466 | 0.240848 | 1.30058 | 4.09441 |
| validation | 201 | 0.501087 | 0.385356 | 1.30058 | 3.09887 |
| test | 203 | 0.512979 | 0.433526 | 1.24765 | 2.6988 |

## Figures

- `anomaly_rate_by_time.png` — percentage of series flagged per timestamp.
- `sample_series/*.png` — ten newly randomized raw-series and score plots.
- `anomaly_rate_by_time.json` — exact counts and percentages.
