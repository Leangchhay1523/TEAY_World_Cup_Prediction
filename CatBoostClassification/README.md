# FIFA World Cup 2026 Prediction Project

This repository contains the FIFA World Cup 2026 prediction project, organized into a compact structure that keeps the baseline system and machine learning system separate.

## Main Folders

- `data/`: Shared raw, processed, and live-update data.
- `notebooks/`: Exploratory analysis and research notes.
- `shared/`: Reusable project code for future versions.
- `tests/`: Future automated tests.
- `version_1_baseline/`: Baseline prediction system.
- `core_prediction/`: Machine learning workspace.
- `_archive/`: Preserved old logs and presentation material.


## Machine Learning

`core_prediction/` is reserved for ML experiments, feature documentation, notebooks, trained models, processed ML datasets, and generated ML outputs.

Current Version research inputs:

- `data/raw/results.csv`
- `data/live_updates/elo_ratings.csv`
- `data/live_updates/fifa_rankings.csv`
- `data/processed/worldcup_2026_fixtures_cleaned.csv`

Current Version report:

- `core_prediction/reports/DATA_FEATURE.md`
- `core_prediction/reports/training_report_v2.md`

## Documentation

- `PROJECT_STRUCTURE.md`: Current folder architecture.
- `MIGRATION_REPORT.md`: File movement and preservation history.
