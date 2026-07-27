# Robust panel-factor report

The model learns relationships across all series jointly. Its final cell score combines robust reconstruction error with the effect of an unexpected shared latent-state change.

| Split | Mean score | RMS score | Median | P95 | P99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 2.5657 | 5.55263 | 0.636291 | 18.0777 | 20.0194 |
| validation | 4.52413 | 7.50486 | 1.66469 | 18.3597 | 20.4918 |
| test | 4.38464 | 7.19155 | 1.65862 | 18.4819 | 20.8131 |

## Figures

- `training_objective.png` — reconstruction and temporal losses.
- `inference_convergence.png` — validation/test latent-state fitting.
- `split_score_metrics.png` — full-grid mean and RMS scores.
