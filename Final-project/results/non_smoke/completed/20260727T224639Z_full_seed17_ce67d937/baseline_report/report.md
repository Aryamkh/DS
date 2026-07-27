# Baseline report

## Full-week anomaly rate

A series is flagged when its score is above the threshold calibrated on validation data. Percentages use only finite series at each timestamp.

| Method | Mean % | Median % | P95 % | Max % | Peak timestamp |
| --- | ---: | ---: | ---: | ---: | --- |
| seasonal_mad | 0.372509 | 0.180343 | 1.31129 | 1.71326 | 1784706000 |
| holt_winters | 0.243284 | 0.0676285 | 0.721371 | 0.924256 | 1785159000 |

| Method | Split | Median | P95 | P99 | Max |
| --- | --- | ---: | ---: | ---: | ---: |
| seasonal_mad | train | 0 | 10.6648 | 962617 | 6.61783e+16 |
| seasonal_mad | validation | 0 | 2011.6 | 1e+06 | 6.11417e+15 |
| seasonal_mad | test | 0 | 3703.62 | 1.05128e+06 | 6.12539e+15 |
| holt_winters | train | 0 | 6.7769 | 88.8507 | 3.10485e+14 |
| holt_winters | validation | 0 | 8.1389 | 538.984 | 5.08742e+13 |
| holt_winters | test | 0 | 7.36457 | 527.832 | 2.88438e+14 |

- `baseline_score_histograms.png` — validation/test score distributions.
- `anomaly_rate_by_time.png` — full-week anomaly percentage by timestamp.
- `anomaly_rate_by_time.json` — timestamp-level counts and percentages.
