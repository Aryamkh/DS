# Baseline report

## Series quality filter

Retained `2076` of `4508` series using training data only.

| Reason | Removed |
| --- | ---: |
| Insufficient coverage | 72 |
| Always zero | 2057 |
| Constant or numerically flat | 264 |
| Too few informative values | 39 |

## Full-week anomaly rate

A series is flagged when its score is above the threshold calibrated on validation data. Percentages use only finite series at each timestamp.

| Method | Mean % | Median % | P95 % | Max % | Peak timestamp |
| --- | ---: | ---: | ---: | ---: | --- |
| seasonal_mad | 0.341739 | 0.192678 | 1.05973 | 1.39692 | 1784658300 |
| holt_winters | 0.23434 | 0.0963391 | 0.578035 | 0.770713 | 1784991300 |

| Method | Split | Median | P95 | P99 | Max |
| --- | --- | ---: | ---: | ---: | ---: |
| seasonal_mad | train | 0.674491 | 21276.6 | 1.94074e+06 | 6.61783e+16 |
| seasonal_mad | validation | 1.6511 | 212388 | 1.16444e+07 | 6.11417e+15 |
| seasonal_mad | test | 2.02076 | 295203 | 1.38846e+07 | 6.12539e+15 |
| holt_winters | train | 0.665385 | 18.9435 | 221.555 | 3.10485e+14 |
| holt_winters | validation | 0.890569 | 22.7384 | 6135.71 | 5.08742e+13 |
| holt_winters | test | 0.628657 | 18.3103 | 2998.58 | 2.58055e+14 |

- `baseline_score_histograms.png` — validation/test score distributions.
- `anomaly_rate_by_time.png` — full-week anomaly percentage by timestamp.
- `anomaly_rate_by_time.json` — timestamp-level counts and percentages.
