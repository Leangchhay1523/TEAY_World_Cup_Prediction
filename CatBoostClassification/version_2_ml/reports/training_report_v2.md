# Version 2 Training Report

## Summary

- Active goal model: rating/statistical Poisson logic in prediction script
- Active trained model: CatBoostClassifier for match outcome only
- Deprecated: CatBoostRegressor goal models are preserved if present but not used
- Split strategy: time-aware split, older matches for training and newer matches for validation
- Latest rating snapshot features used in historical training: `False`

## Data

- Training rows: 39546
- Validation rows: 9887
- Training period: 1872-11-30 to 2016-03-25
- Validation period: 2016-03-25 to 2026-06-18
- Processed dataset: `version_2_ml\processed_data\training_dataset_v2.csv`

## Evaluation

- Outcome accuracy: 0.5976
- Outcome log loss: 0.8801

## Saved Model

- `version_2_ml\models\catboost_outcome_model.cbm`

## Most Important Features

```text
feature | outcome_importance
elo_diff | 26.324621216868113
tournament | 9.384079327253492
concede_rate_diff | 7.701089498047257
year | 7.021603814689946
team_2_elo | 6.347280651394956
team_2_goal_rate | 5.509598734812973
neutral | 5.435163155261128
team_2_concede_rate | 5.317753466412384
team_1_concede_rate | 5.2532862674704095
team_1_goal_rate | 4.677929189917114
team_1_elo | 4.270503695304335
goal_rate_diff | 3.8602367228615924
team_2_recent_form | 3.2844106708720613
recent_form_diff | 2.853217467013838
team_1_recent_form | 2.759226121820416
```

## Leakage Controls

- Rolling Elo, form, goal rate, and concede rate are computed before each match is added to team history.
- The default training mode does not use latest FIFA snapshots as historical match features.
- Use `--allow-latest-rating-features` only for experiments where you accept that leakage risk.