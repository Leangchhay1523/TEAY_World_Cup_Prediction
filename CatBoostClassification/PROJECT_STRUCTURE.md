# Project Structure

This document describes the simplified FIFA World Cup 2026 prediction project layout.

## Target Layout

```text
worldcup_prediction_project/
  data/
    raw/
    processed/
    live_updates/

  notebooks/
  shared/
  tests/

  version_1_baseline/
    scripts/
    outputs/
    README.md

  core_prediction/
    scripts/
    notebooks/
    models/
    processed_data/
    outputs/
    reports/
    README.md

  _archive/
    old_logs/
    old_presentation/

  README.md
  PROJECT_STRUCTURE.md
  MIGRATION_REPORT.md
  requirements.txt
```

## Data

- `data/raw/`: Shared source datasets, including external ranking files used by Version 2.
- `data/processed/`: Shared processed datasets, including cleaned World Cup 2026 fixtures.
- `data/live_updates/`: Latest Elo and FIFA ranking snapshots for future prediction runs.

## Machine Learning

`core_prediction/` contains the ML workspace:

- `scripts/`: ML scripts.
- `notebooks/`: Version-specific ML notebooks.
- `models/`: Trained model artifacts.
- `processed_data/`: ML-ready datasets.
- `outputs/`: ML predictions and evaluation outputs.
- `reports/`: ML feature notes and evaluation reports.

## Shared And Tests

- `shared/`: Reusable code that can be shared by future project versions.
- `tests/`: Automated tests and validation checks.
- `notebooks/`: General exploratory notebooks and research notes.

## Archive

`_archive/` preserves files from folders that were removed or merged:

- `_archive/old_logs/`: Old log files.
- `_archive/old_presentation/`: Old presentation files.

No important files should be deleted during structure cleanup. Files should be moved into the archive or promoted into root documentation instead.
