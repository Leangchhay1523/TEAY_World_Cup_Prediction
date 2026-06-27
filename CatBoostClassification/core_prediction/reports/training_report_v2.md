# Version 2 Training Report

## Summary

- Active outcome model: CatBoostClassifier for `team_1_win`, `draw`, and `team_2_win`.
- Active goal models: CatBoostRegressor for `goals_team_1` and `goals_team_2`.
- The outcome model predicts winner probabilities only.
- The goal models predict expected goals only.
- Statistical xG provides a stable expected-goals baseline from rates, recent goals, Elo, FIFA points when available, and tournament importance.
- The goal ensemble blends CatBoost xG and statistical xG before Poisson scoring.
- Goal-scale calibration adjusts ensemble xG before building the Poisson score matrix.
- Default prediction selects the highest expected competition points.
- Pure score-probability selection is retained only as optional prediction/debug mode.
- Version 2 upgrade: leakage-safe recent form, goals, opponent-strength, Package A, Package B, Package C, and tournament-type features
- Split strategy: time-aware split, older matches for training and newer matches for validation
- Latest rating snapshot features used in historical training: `False`

## Data

- Training rows: 39572
- Validation rows: 9893
- Training period: 1872-11-30 to 2016-03-26
- Validation period: 2016-03-27 to 2026-06-25
- Processed dataset: `core_prediction\processed_data\training_dataset_v2.csv`

## Evaluation

- Outcome accuracy: 0.6023
- Outcome log loss: 0.8713
- Team 1 goals MAE: 1.0196
- Team 2 goals MAE: 0.8357
- Team 1 goals RMSE: 1.3542
- Team 2 goals RMSE: 1.1189

## Goal Ensemble Tuning

- Best CatBoost xG weight: 0.6
- Best statistical xG weight: 0.4
- Validation exact score accuracy: 0.1422
- Validation goal difference accuracy: 0.2584
- Validation winner accuracy from selected score: 0.6007
- Validation average competition points: 3.0300
- Tuning results: `core_prediction\outputs\goal_ensemble_tuning_results.csv`
- Saved ensemble config: `core_prediction\models\goal_ensemble_config.json`

## Score Selection Calibration

- Default final score selection is highest expected competition points.
- Pure score-probability selection is retained only as optional prediction/debug mode.
- Best goal scale: 1.00
- Exact score accuracy: 0.1382
- Goal difference accuracy: 0.2577
- Winner accuracy from selected score: 0.5503
- Average absolute goal error: 1.8210
- Predicted average total goals: 1.7560
- Actual average total goals: 2.7331
- Draw prediction rate: 0.3096
- Tuning results: `core_prediction\outputs\score_selection_tuning_results.csv`
- Saved score-selection config: `core_prediction\models\score_selection_config.json`

## Saved Models

- `core_prediction\models\catboost_outcome_model.cbm`
- `core_prediction\models\catboost_goals_team_1.cbm`
- `core_prediction\models\catboost_goals_team_2.cbm`

## Most Important Features

```text
feature | outcome_importance | goals_team_1_importance | goals_team_2_importance
elo_diff | 9.82500206172249 | 5.296307116291896 | 4.362048752051176
elo_ratio | 9.072402891255694 | 6.237505388487436 | 6.448632189851887
avg_opponent_elo_diff_last_10 | 4.344688425448458 | 4.591438229615004 | 4.304215282197153
neutral | 3.668871937743648 | 1.6441217144461142 | 4.354725562125882
tournament | 2.251132141469299 | 1.0111933155168127 | 0.42728877876857146
h2h_avg_goal_difference_team_1 | 1.9589439185519337 | 2.191388809916197 | 1.5943177010791105
avg_goals_conceded_diff_last_10 | 1.7940509004472158 | 3.4700841111260377 | 3.6426994638306303
concede_rate_diff | 1.716868844106671 | 6.085474563492234 | 5.863664112527124
team_2_elo | 1.7164013649223169 | 2.2111896681943577 | 0.5387014062199937
year | 1.6124163413726929 | 3.9369541392958594 | 4.9072303647147955
avg_goal_difference_diff_last_10 | 1.5826382428278443 | 2.4870478539805534 | 3.342358076060816
team_2_avg_opponent_elo_last_10 | 1.4155615617471562 | 2.159248404895312 | 0.6992593492169527
team_1_avg_opponent_elo_last_10 | 1.3653434644248108 | 0.6472329090376521 | 1.516124578291091
team_2_away_goal_rate_last_10 | 1.353095553639057 | 0.8662476383100585 | 2.7236437789468626
h2h_team_1_win_rate | 1.346968952812722 | 0.4077176111068695 | 0.42547113357881605
```

## Version 2 Feature Families

- Recent result form: wins, draws, and losses in each team's last 5 and last 10 matches.
- Recent points form: football points over last 5 and last 10, plus team-difference columns.
- Recent attacking/defensive form: average goals scored and conceded over last 5 and last 10.
- Opponent strength: average pre-match rolling Elo of previous opponents over last 5 and last 10.
- Package A: Elo ratio, draw rate, clean-sheet rate, failed-to-score rate, and average goal-difference features.
- Package B: total-goals threshold rates and goal-scoring/conceding volatility features.
- Package C: head-to-head, neutral-ground, home/away style, and attack-vs-defense interaction features.
- Tournament context: `tournament_type_group` and `tournament_importance_weight`.

## Leakage Controls

- Rolling Elo, form, goal rate, concede rate, and opponent-strength features are computed before each match is added to team history.
- The default training mode does not use latest FIFA snapshots as historical match features.
- Use `--allow-latest-rating-features` only for experiments where you accept that leakage risk.