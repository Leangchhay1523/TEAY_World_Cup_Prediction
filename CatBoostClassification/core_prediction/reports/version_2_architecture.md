# Version 2 Architecture

Version 2 is a hybrid winner-and-score system:

```text
CatBoostClassifier
-> winner probabilities

CatBoostRegressor Team 1 Goals
-> expected_goals_team_1

CatBoostRegressor Team 2 Goals
-> expected_goals_team_2

Statistical xG baseline
-> stable expected-goals estimates

xG ensemble
-> final ensemble expected goals

Goal-scale calibration
-> calibrated Poisson lambdas

Poisson
-> score probability matrix from 0-0 to 6-6 using calibrated xG

Score-probability selector
-> final predicted winner, exact score, goal difference, confidence
```

## Active Training

`scripts/train_version_2_models.py` trains and saves exactly these active models:

```text
core_prediction/models/catboost_outcome_model.cbm
core_prediction/models/catboost_goals_team_1.cbm
core_prediction/models/catboost_goals_team_2.cbm
core_prediction/models/goal_ensemble_config.json
core_prediction/models/score_selection_config.json
```

Targets:

- Outcome classifier: `target_outcome` with classes `team_1_win`, `draw`, and `team_2_win`.
- Team 1 goal regressor: `goals_team_1`.
- Team 2 goal regressor: `goals_team_2`.

All three CatBoost models use the same Version 2 feature names and feature order. The feature contract is saved in `outputs/training_metrics_v2.json`.

Training also tunes the expected-goals ensemble on the time-based validation split. The grid tests CatBoost weights from `0.0` through `1.0` in `0.1` steps and selects the best average competition points per match, with exact score accuracy, goal difference accuracy, and lower goal MAE as tie-breakers.

Current saved ensemble:

```text
catboost_weight = 0.7
statistical_weight = 0.3
validation_avg_competition_points = 2.9985
```

Training also tunes final score selection on the same time-based validation split. Goal scales from `0.80` through `1.50` are evaluated with highest score-probability selection. The primary target is exact score accuracy, followed by goal difference accuracy, winner accuracy, and total-goal distribution closeness.

Current saved score-selection config:

```text
decision_rule = highest_adjusted_score_probability
goal_scale = 0.90
exact_score_accuracy = 0.1385
goal_difference_accuracy = 0.2533
winner_accuracy_from_score = 0.5742
draw_prediction_rate = 0.1756
```

## Leakage Policy

- Historical matches are sorted by date before feature generation.
- Rolling Elo, form, goal rates, concede rates, and opponent-strength features are computed before the current match is added to either team's history.
- Training uses a time-based split: older matches train the models and newer matches validate them.
- Random train/test splits are not used.
- Latest FIFA snapshot features are disabled by default for historical training. They can only be enabled with `--allow-latest-rating-features` for experiments that accept that leakage risk.

## Active Prediction

`scripts/predict_single_match_v2.py` predicts one match at a time with the unchanged CLI:

```text
python CatBoostClassification/core_prediction/scripts/predict_single_match_v2.py --team_1 Brazil --team_2 Scotland --stage "Group Stage" --match_date 2026-06-24
```

The script validates the fixture, builds the saved training feature contract, loads the classifier, both regressors, and the goal ensemble config, then saves the selected candidate to `outputs/version_2_predictions.csv`.

Default decision mode:

```text
python CatBoostClassification/core_prediction/scripts/predict_single_match_v2.py --team_1 Brazil --team_2 Scotland --stage "Group Stage" --match_date 2026-06-24
```

Optional pure score-probability mode:

```text
python CatBoostClassification/core_prediction/scripts/predict_single_match_v2.py --team_1 Brazil --team_2 Scotland --stage "Group Stage" --match_date 2026-06-24 --decision_mode score_probability
```

## Expected Goals

Expected goals come from an ensemble:

```text
final_xg_team_1 =
  weight_catboost * catboost_xg_team_1
  + (1 - weight_catboost) * statistical_xg_team_1

final_xg_team_2 =
  weight_catboost * catboost_xg_team_2
  + (1 - weight_catboost) * statistical_xg_team_2
```

The statistical xG baseline uses team goal rate, opponent concede rate, recent goals scored and conceded, Elo difference, FIFA points difference when available, and tournament importance weight.

Before score selection, ensemble xG is scaled:

```text
calibrated_xg_team_1 = ensemble_xg_team_1 * goal_scale
calibrated_xg_team_2 = ensemble_xg_team_2 * goal_scale
```

After scaling, calibrated xG is clamped to:

```text
minimum = 0.05
maximum = 6.00
```

## Score Selection

Every scoreline from `0-0` to `6-6` is a candidate, for 49 total scorelines.

Each candidate stores:

- `poisson_score_probability`
- `winner_from_score`
- `goal_difference`
- `outcome_probability`
- `goal_difference_probability`
- `expected_competition_points`

Default final decision:

```text
final_score = argmax(expected_competition_points)
```

The selected score directly determines `predicted_winner` and `predicted_goal_difference`.

Pure exact-score probability selection is retained for optional `score_probability` mode. The default expected-points selector is:

```text
expected_points =
  3 * outcome_probability
  + 2 * goal_difference_probability
  + 5 * poisson_score_probability
```

There is no high-confidence bonus.
