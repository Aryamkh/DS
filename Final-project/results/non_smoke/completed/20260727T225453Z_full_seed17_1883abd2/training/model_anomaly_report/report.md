# Learned-model anomaly report

Validation-calibrated threshold: `21.2275`

Percentages use the `2076` series retained by the training-only quality filter (from `4508` input series).

| Split | Evaluated timestamps | Mean % | Median % | P95 % | Max % |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_week | 2017 | 0.476998 | 0.0481696 | 2.61079 | 7.36994 |
| train | 1613 | 0.437753 | 0 | 2.90944 | 7.36994 |
| validation | 201 | 0.50018 | 0.433526 | 0.963391 | 1.34875 |
| test | 203 | 0.765874 | 0.722543 | 1.35936 | 2.07129 |

## Figures

- `anomaly_rate_by_time.png` — percentage of series flagged per timestamp.
- `sample_series/*.png` — ten newly randomized raw-series and score plots.
- `anomaly_rate_by_time.json` — exact counts and percentages.
