# Learned-model anomaly report

Validation-calibrated threshold: `20.3078`

| Split | Evaluated timestamps | Mean % | Median % | P95 % | Max % |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_week | 2017 | 0.357453 | 0.112714 | 1.51037 | 4.75654 |
| train | 1613 | 0.298639 | 0.0676285 | 1.578 | 4.75654 |
| validation | 201 | 0.500245 | 0.450857 | 0.901713 | 1.38889 |
| test | 203 | 0.683395 | 0.631199 | 1.19477 | 1.66817 |

## Figures

- `anomaly_rate_by_time.png` — percentage of series flagged per timestamp.
- `sample_series_raw_and_anomaly.png` — five seeded random raw series and scores.
- `anomaly_rate_by_time.json` — exact counts and percentages.
