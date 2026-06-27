# Version 2 Machine Learning System

Version 2 is the machine learning system for the FIFA World Cup 2026 prediction project.

It is separate from `version_1_baseline/` and does not modify Version 1.

## Current Architecture

The active architecture is:

1. `CatBoostClassifier` predicts match outcome:
   - `team_1_win`
   - `draw`
   - `team_2_win`
2. `CatBoostRegressor` model 1 predicts `expected_goals_team_1`.
3. `CatBoostRegressor` model 2 predicts `expected_goals_team_2`.
4. A statistical xG baseline provides a stable expected-goals estimate.
5. The saved xG ensemble blends CatBoost xG and statistical xG.
6. A saved goal-scale calibration adjusts ensemble xG.
7. Poisson converts calibrated xG into a 0-0 to 6-6 score probability matrix.
8. The default selector chooses the highest expected competition points, combining outcome, goal-difference, and exact-score probabilities. Pure score-probability mode remains optional.

## Main Commands

Train the outcome and goal models:

```bash
python core_prediction/scripts/train_version_2_models.py
```

Predict one match:

```bash
python core_prediction/scripts/predict_single_match_v2.py --team_1 Brazil --team_2 Scotland --stage "Group Stage" --match_date 2026-06-24
```

## Folder Contents

- `scripts/train_version_2_models.py`: Builds the training dataset, trains the CatBoost outcome classifier plus two CatBoost goal regressors, tunes the xG ensemble, and tunes score-selection goal scale.
- `scripts/predict_single_match_v2.py`: Predicts one match using the classifier, goal regressors, statistical xG baseline, saved ensemble weights, calibrated Poisson score matrix, and expected-points selector.
- `notebooks/train_version_2_models.ipynb`: Step-by-step training notebook.
- `notebooks/predict_single_match_v2.ipynb`: Step-by-step prediction notebook.
- `models/catboost_outcome_model.cbm`: Active trained outcome classifier.
- `models/catboost_goals_team_1.cbm`: Active trained Team 1 goals regressor.
- `models/catboost_goals_team_2.cbm`: Active trained Team 2 goals regressor.
- `models/goal_ensemble_config.json`: Tuned CatBoost/statistical xG ensemble weights.
- `models/score_selection_config.json`: Tuned goal scale for score-probability selection.
- `processed_data/training_dataset_v2.csv`: Processed ML training dataset.
- `outputs/training_metrics_v2.json`: Metrics and saved feature contract.
- `outputs/goal_ensemble_tuning_results.csv`: Full validation results for the xG ensemble weight grid.
- `outputs/score_selection_tuning_results.csv`: Full validation results for the goal-scale grid.
- `outputs/version_2_predictions.csv`: Saved match predictions.
- `reports/version_2_architecture.md`: Current architecture explanation.
- `reports/training_report_v2.md`: Training metrics and feature importance.
- `reports/prediction_system_v2.md`: Prediction workflow explanation.
- `reports/DATA_FEATURE.md`: Feature and source-column notes.

## Required Inputs

- `data/raw/results.csv`
- `data/raw/elo_ratings.csv`
- `data/raw/fifa_rankings.csv`
- `data/worldcup_2026_fixtures/worldcup_2026_fixtures_cleaned.csv`
- `data/worldcup_2026_fixtures/future_match.csv`
