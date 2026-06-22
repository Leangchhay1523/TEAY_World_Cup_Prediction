# Version 2 Machine Learning System

Version 2 is the machine learning system for the FIFA World Cup 2026 prediction project.

It is separate from `version_1_baseline/` and does not modify Version 1.

## Current Architecture

Version 2 was simplified because the CatBoostRegressor goal models produced unstable expected-goals predictions.

The active architecture is now:

1. `CatBoostClassifier` predicts match outcome:
   - `team_1_win`
   - `draw`
   - `team_2_win`
2. A rating-based Poisson goal model estimates expected goals.
3. A 0-0 to 6-6 score matrix estimates exact-score and goal-difference probabilities.
4. A candidate optimizer selects the prediction with the highest expected competition score.

Old goal-regressor files may still exist in `models/`, but they are preserved only for archive/comparison and are not used by the active prediction script.

## Main Commands

Train the outcome model:

```bash
python core_prediction/scripts/train_version_2_models.py
```

Predict one match:

```bash
python core_prediction/scripts/predict_single_match_v2.py --team_1 Brazil --team_2 Scotland --stage "Group Stage" --match_date 2026-06-24
```

## Folder Contents

- `scripts/train_version_2_models.py`: Builds the training dataset and trains only the CatBoost outcome classifier.
- `scripts/predict_single_match_v2.py`: Predicts one match using the classifier, Poisson goal model, and candidate optimizer.
- `notebooks/train_version_2_models.ipynb`: Step-by-step training notebook.
- `notebooks/predict_single_match_v2.ipynb`: Step-by-step prediction notebook.
- `models/catboost_outcome_model.cbm`: Active trained outcome classifier.
- `models/catboost_goals_team_1.cbm`: Deprecated preserved goal model, not used.
- `models/catboost_goals_team_2.cbm`: Deprecated preserved goal model, not used.
- `processed_data/training_dataset_v2.csv`: Processed ML training dataset.
- `outputs/version_2_predictions.csv`: Saved match predictions.
- `reports/version_2_architecture.md`: Current architecture explanation.
- `reports/training_report_v2.md`: Training metrics and feature importance.
- `reports/prediction_system_v2.md`: Prediction workflow explanation.
- `reports/DATA_FEATURE.md`: Feature and source-column notes.

## Required Inputs

- `data/raw/results.csv`
- `data/live_updates/elo_ratings.csv`
- `data/live_updates/fifa_rankings.csv`
- `data/processed/worldcup_2026_fixtures_cleaned.csv`

Update the two files in `data/live_updates/` before predicting matches.
