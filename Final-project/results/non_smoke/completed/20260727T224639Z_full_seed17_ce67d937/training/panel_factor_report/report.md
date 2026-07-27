# Robust panel-factor report

The model learns relationships across all series jointly. Its final cell score combines robust reconstruction error with the effect of an unexpected shared latent-state change.

| Split | Mean score | RMS score | Median | P95 | P99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 1.22698 | 3.82566 | 0.0880017 | 7.09011 | 19.8208 |
| validation | 2.20915 | 5.20058 | 0.265082 | 16.5119 | 19.8707 |
| test | 2.19283 | 5.07049 | 0.27944 | 14.6358 | 20.0493 |

## Figures

- `training_objective.png` — reconstruction and temporal losses.
- `inference_convergence.png` — validation/test latent-state fitting.
- `split_score_metrics.png` — full-grid mean and RMS scores.
