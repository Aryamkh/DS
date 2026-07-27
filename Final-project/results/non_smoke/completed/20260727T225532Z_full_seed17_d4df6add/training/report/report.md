# Training report

Anomaly threshold (validation quantile): `20.0026`

| Split | MAE | RMSE | Count |
| --- | ---: | ---: | ---: |
| train | 0.904303 | 3.18943 | 199881 |
| validation | 1.28567 | 3.71498 | 199787 |
| test | 1.36125 | 3.79621 | 199901 |

## Error metrics

| Split | Median AE | P95 AE | P99 AE | Max AE |
| --- | ---: | ---: | ---: | ---: |
| validation | 0.197388 | 6.67465 | 19.8654 | 40.625 |
| test | 0.25 | 7.21048 | 19.8799 | 40.5625 |

## Figures

- `loss_curve.png` — train loss and validation MAE/RMSE.
- `split_metrics.png` — MAE/RMSE by chronological split.
- `error_histograms.png` — log-scaled validation/test error distributions.
