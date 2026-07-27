# Repaired Thanos dataset

This directory contains a cleaned wide CSV built from the completed v3 export in `../`.

- Only `queries/` was used; `skipped_queries/` was ignored.
- Only complete queries with fewer than 50 returned series were kept.
- Series with fewer than 2,000 samples were removed.
- The data covers 7 days at 5-minute resolution: 2,017 timestamps.
- `final.csv` has one `timestamp` column followed by `series_######` columns. Timestamps are Unix seconds in UTC; blank cells mean that series had no sample at that time.
- Samples are aligned to the observed regular 5-minute timestamp grid, preserving the actual observed range.

`metadata.csv` maps each series column to its query, labels, PromQL, source file, and sample count. `metadata.json` records the filters, range, alignment, and dataset totals.

For sharing without secret strings, use `metadata-hashed.csv` and `metadata-hashed.json`. They contain only numeric `group_tag`, `query_tag`, and `label_tag` categories plus series indexes and sample counts; no original labels, query names, PromQL, IDs, or paths are included.

Current size: 4,508 series from 1,135 queries. The original dataset/output directories were not modified.
