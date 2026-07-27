# Time-series anomaly detection project report

## 1. Problem and data

The input is a wide table of Thanos metric series. It has 2,017 timestamps,
one row every five minutes for approximately one week, and 4,508 candidate
series. The timestamp column is kept separately and every other column is
treated as one series. Empty cells are missing observations, not zeros.

The main output is not only one score for the whole dataset. The required
output is a score for each observed series at each timestamp, together with a
percentage showing how many available series are anomalous at that timestamp.
This is why the relationship between columns is important: a change shared by
many related series should be handled differently from one unusual value in a
single series.

All experiments use a chronological split:

- 80% training
- 10% validation
- 10% test

The split is made before fitting baselines, selecting useful series, fitting
models, or choosing an anomaly threshold. The validation set is used to choose
the 99.5th-percentile threshold. The test set is only used for the final
comparison.

## 2. Data checks and preprocessing

The exploratory checks showed that the raw table contains many unhelpful
columns. A large number of series are always zero or constant, and a smaller
group has too many missing training values to estimate a reliable seasonal
distribution.

The final preprocessing step uses the training portion only. A series is kept
when it has at least 80% finite training coverage and at least 12 values that
are meaningfully different from its training median. It removes:

- 2,057 always-zero series
- 264 constant or numerically flat series
- 72 series with insufficient training coverage
- 39 series with too few informative changes

This leaves 2,076 useful series. The filter is applied before baseline fitting,
metadata selection, and model training. No validation or test values are used
to decide which columns survive.

I considered linearly interpolating missing values. I did not use it in the
final pipeline because interpolation would create values that were never
observed and could hide the exact missingness or an anomaly. Instead, missing
values are represented by an observation mask. The models can use the value
and the mask, while the final score is written only for finite observations.

## 3. Baseline methods

### Seasonal median and MAD

The first baseline estimates a separate median and MAD for each five-minute
position in the daily cycle. There are 288 positions in a day. The baseline is
fitted on training rows only. A residual is scaled by the robust MAD estimate;
the median and MAD are safer than a mean and standard deviation when the data
contains very large spikes.

This method is simple and useful as a reference, but it becomes unstable for
columns with almost no variation. The quality filter was added partly because
of this issue.

### Holt-Winters residual

The second baseline is an additive Holt-Winters model with level, trend, and a
288-step seasonal component. It is initialized on the training data and then
updated causally for validation and test rows. Its residual is robust-scaled
using training residuals.

On the latest filtered baseline run, Holt-Winters gave a lower full-week
anomaly percentage than seasonal MAD:

| Baseline | Mean anomalous series | Median | P95 |
| --- | ---: | ---: | ---: |
| Seasonal median/MAD | 0.3417% | 0.1927% | 1.0597% |
| Holt-Winters | 0.2343% | 0.0963% | 0.5780% |

The baseline thresholds are calibrated independently, so their absolute
threshold values should not be compared directly.

## 4. Learned temporal models

### GRU model

The first learned model is a small global GRU. It is shared by all series,
rather than training one separate network per column. Each input window has
the normalized value and an observed-value flag. The last GRU state is passed
to a small prediction head.

This model is a useful sanity check and captures short temporal dependencies,
but it mainly sees one series at a time. It does not naturally represent the
large cross-series structure of this dataset, so it was not selected as the
main final model.

### Context-TCN model

The Context-TCN replaces recurrent processing with causal one-dimensional
convolutions. It uses residual blocks with kernel size three and increasing
dilations, so it can see a longer history without processing every time step
through a recurrent loop. An attention pooling layer combines information from
the window.

The contextual version also receives:

- global mean and standard deviation at each historical timestamp
- global absolute activity and observed-series fraction
- sine and cosine time-of-day features
- leave-one-out peer means for metadata groups

Metadata categories are embedded for group, query, and label tags. The
metadata state is combined with the temporal state through a learned gate. The
latest filtered Context-TCN run had sampled test MAE 1.3613 and RMSE 3.7962,
and its full-grid anomaly timeline had mean anomalous percentage 0.4387% with
P95 1.3006%.

This was the best recent temporal forecaster by sampled error and is a good
choice when the deployment process is naturally causal and window-based.

## 5. Wide-panel factor model

The panel-factor model was designed specifically for the short-and-wide shape
of this data. Instead of creating millions of independent windows, it treats
the useful data as a time-by-series matrix and learns a low-rank system:

\[
  \hat{x}_{t,j} = f_t^T l_j + b_t + c_j + b_0
\]

Here, `f_t` is a small latent system state and `l_j` is the loading for series
`j`. The model uses a rank of 32 for full runs. The series loading is made from
a learned residual component plus embeddings for group, query, and label tags,
as well as the normalized metadata sample count.

The training objective uses a Huber reconstruction loss. This is less affected
by large spikes than mean squared error. Temporal smoothness and daily
regularization keep the latent state from changing arbitrarily at every row.
The model also learns series and timestamp bias terms.

For validation and test rows, the series loadings remain frozen. The local
latent state is inferred from the current panel, using the previous-day latent
state as a reference. The final score is the reconstruction error plus a
collective component caused by an unexpected latent or shared-bias change.
This makes the model useful for detecting both one-series deviations and
events affecting a group of related series.

The panel-factor approach is the most direct model of cross-series
relationships. Its full-grid output is also efficient because matrix
multiplication scores thousands of series at once.

## 6. What was tested and what worked

The experiments progressed from simple references to models that use more of
the data structure:

1. Seasonal median/MAD established a robust daily baseline.
2. Holt-Winters added level and trend adaptation and produced a lower anomaly
   percentage than seasonal MAD on the filtered data.
3. The GRU checked whether a compact recurrent window model was sufficient.
4. The Context-TCN improved long-window temporal processing and added global,
   peer, calendar, and metadata context.
5. The panel-factor model was added for the main short-wide problem, where the
   relationships among thousands of columns are more important than treating
   each series independently.
6. The quality filter removed degenerate columns that created unreliable
   zero/MAD behavior.
7. Full-grid scoring and timestamp plots were added so that anomaly behavior
   can be inspected over the complete week rather than only through a random
   batch metric.

The most useful final runs are:

| Run | Model | Series | Test MAE | Test RMSE | Mean anomalous % | P95 anomalous % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `20260727T225532Z_full_seed17_d4df6add` | Filtered Context-TCN + metadata | 2,076 | 1.3613 | 3.7962 | 0.4387 | 1.3006 |
| `20260727T225453Z_full_seed17_1883abd2` | Filtered metadata panel-factor | 2,076 | 4.3846 | 7.1915 | 0.4770 | 2.6108 |
| `20260727T224639Z_full_seed17_ce67d937` | Panel-factor before filtering | 4,436 | 2.1928 | 5.0705 | 0.3575 | 1.5104 |

The Context-TCN error is calculated on a large sampled test evaluation, while
the panel-factor error is calculated on the full test grid. Those MAE/RMSE
values are therefore not a fair direct ranking. For this task, the timestamp
anomaly percentage and the raw-series/score plots are more important than one
aggregate error number.

## 7. What did not work as well

- A plain recurrent model was useful for comparison but did not explicitly
  learn the relationships between series.
- Treating all columns as independent windows makes full-grid scoring more
  expensive and misses shared system events.
- Keeping constant and always-zero columns caused degenerate robust scales and
  very large meaningless residual values. Filtering solved this problem.
- Linear interpolation was not included because it would turn missing data
  into invented observations and could reduce anomaly visibility.
- A larger temporal model alone is not a complete solution for this topology;
  the panel-factor model is better aligned with the matrix structure, even
  though its raw full-grid MAE is not directly comparable to sampled TCN MAE.

## 8. Leakage and evaluation checks

The following rules were kept throughout the experiments:

- The chronological split is performed before fitting statistics.
- Series filtering uses only training observations.
- Seasonal medians, MAD scales, robust scalers, and Holt-Winters
  initialization use training data only.
- The anomaly threshold is selected from validation scores, not test scores.
- Validation and test observations are never used to update the learned series
  loadings.
- Missing values remain masked and are not counted as anomalies.
- Metadata is static descriptive information; it is aligned by series index and
  does not contain target values.

The panel-factor model is intended for offline or batch analysis: its held-out
latent state is inferred from the panel at those timestamps. It is not claimed
to be a strict one-step online forecaster. The Context-TCN is the better choice
when a strictly causal window-by-window deployment is required.

## 9. Reports and plots

The selected run folders contain Markdown reports and PNG files. The plots
include training objectives, split metrics, baseline score distributions,
anomaly percentage over time, and raw values beside anomaly scores for ten
randomly selected series in the latest runs. No images are embedded in this
report so they can be selected separately for the final submission.

The copied results package intentionally contains no JSON files, PyTorch
checkpoints, score-grid tensors, or execution logs. The numerical comparison
needed for the write-up is recorded above.

## 10. Final recommendation

For a causal temporal model, use
`20260727T225532Z_full_seed17_d4df6add` as the main reference. For the main
wide-panel anomaly output, use
`20260727T225453Z_full_seed17_1883abd2`. The older
`20260727T224639Z_full_seed17_ce67d937` run is useful as a comparison showing
the effect of applying the quality filter.
