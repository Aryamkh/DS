# Time-series anomaly detection

This project studies a wide time-series dataset from Thanos metrics. The
dataset contains 2,017 timestamps with five-minute spacing, covering one week.
There are 4,508 candidate series in `final.csv`, and the matching metadata is
in `metadata-hashed.csv`.

The goal is to produce an anomaly score for every observed series at every
timestamp and to summarize how many series are anomalous at each time.

## Data preparation

The data is split chronologically into 80% training, 10% validation, and 10%
test data. Missing values are kept as missing during scoring.

Before training the final models, I added a simple quality filter based only on
the training section. It removes series that are always zero, constant, almost
flat, too short, or have too few useful changes. With the final settings,
2,076 of the 4,508 series were retained:

- 2,057 always-zero series
- 264 constant or numerically flat series
- 72 series with insufficient training coverage
- 39 series with too few informative values

This prevents zero-heavy columns from dominating the seasonal statistics while
keeping series that have enough variation to learn.

## Methods tested

### Seasonal median and MAD

For each five-minute position in the day, the median and MAD are calculated on
the training data. The scaled residual is used as the baseline anomaly score.
This is robust to large spikes and is a useful seasonal reference.

### Holt-Winters residuals

An additive Holt-Winters model tracks level, trend, and daily seasonality. Its
one-step residual is used as another baseline. On the filtered data, it gave a
lower full-week anomaly percentage than seasonal MAD (mean 0.2343% compared
with 0.3417%).

### GRU and Context-TCN models

The recurrent model was tested first as a basic learned residual predictor. A
causal TCN was then added to handle longer temporal patterns more efficiently.
The contextual version also uses global activity statistics, time-of-day
features, and peer means from metadata groups.

### Panel-factor model

The final wide-panel idea learns a small number of shared latent factors across
all useful series. Metadata is used to build the series loadings. A robust
Huber reconstruction loss handles spikes, while smoothness and daily
regularization keep the latent state stable.

The final anomaly score combines the individual reconstruction error with the
effect of an unexpected shared-state change. This is useful when many related
series move together at the same timestamp.

## Selected results

The three most useful recent full runs are in
`results/non_smoke/completed/`.

| Run | Model | Series | Test MAE | Test RMSE | Mean anomalous series | P95 anomalous series |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `20260727T225532Z_full_seed17_d4df6add` | Context-TCN + metadata | 2,076 | 1.3613 | 3.7962 | 0.4387% | 1.3006% |
| `20260727T225453Z_full_seed17_1883abd2` | Metadata panel-factor | 2,076 | 4.3846 | 7.1915 | 0.4770% | 2.6108% |
| `20260727T224639Z_full_seed17_ce67d937` | Metadata panel-factor, before filtering | 4,436 | 2.1928 | 5.0705 | 0.3575% | 1.5104% |

The Context-TCN errors use a large sampled test evaluation, while the
panel-factor errors use the complete test grid, so the MAE/RMSE values are not
directly comparable. For the final use case, the timestamp-level anomaly
percentage is the main output.

The selected run folders contain the Markdown reports and PNG plots. The
per-series plots show raw values beside anomaly scores, and the other plots
show training loss, score distributions, split metrics, and anomaly percentage
over time. JSON files, PyTorch checkpoints, and score-grid files were removed
from the copied results package after selecting the runs.

## Running the final model

Use the ml environment configured for this project:

```bash
source /home/aria/sharif/DS/DS_HW4/ml-env/bin/activate
python modal_app.py \
  --full-only \
  --architecture panel_factor \
  --use-metadata \
  --epochs 30 \
  --learning-rate 0.003
```

The original source files are under `src/`. The main wide-panel entry point is
`src/train_panel.py`.
