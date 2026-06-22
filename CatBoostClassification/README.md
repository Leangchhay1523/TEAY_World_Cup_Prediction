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

## Version 1: Baseline

`version_1_baseline/` contains the original baseline system. It uses local data, cleaned data, processed Elo features, and an Elo plus Poisson prediction engine.

The Version 1 logic was not changed. Its local data folders remain in place so the existing scripts continue to behave the same way.

## Version 2: Machine Learning

`core_prediction/` is reserved for ML experiments, feature documentation, notebooks, trained models, processed ML datasets, and generated ML outputs.

Current Version 2 research inputs:

- `data/raw/results.csv`
- `data/live_updates/elo_ratings.csv`
- `data/live_updates/fifa_rankings.csv`
- `data/processed/worldcup_2026_fixtures_cleaned.csv`

Current Version 2 report:

- `core_prediction/reports/DATA_FEATURE.md`
- `core_prediction/reports/training_report_v2.md`

## Documentation

- `PROJECT_STRUCTURE.md`: Current folder architecture.
- `MIGRATION_REPORT.md`: File movement and preservation history.
