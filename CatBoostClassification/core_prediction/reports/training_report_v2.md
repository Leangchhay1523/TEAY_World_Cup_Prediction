# Version 2 Training Report

## Summary

- Active goal model: rating/statistical Poisson logic in prediction script
- Active trained model: CatBoostClassifier for match outcome only
- Version 2 upgrade: leakage-safe recent form, goals, opponent-strength, and tournament-type features
- Deprecated: CatBoostRegressor goal models are preserved if present but not used
- Split strategy: time-aware split, older matches for training and newer matches for validation
- Latest rating snapshot features used in historical training: `False`

## Data

- Training rows: 39556
- Validation rows: 9889
- Training period: 1872-11-30 to 2016-03-26
- Validation period: 2016-03-26 to 2026-06-21
- Processed dataset: `core_prediction\processed_data\training_dataset_v2.csv`

## Evaluation

- Outcome accuracy: 0.6015
- Outcome log loss: 0.8738

## Saved Model

- `core_prediction\models\catboost_outcome_model.cbm`

## Most Important Features

```text
feature | outcome_importance
elo_diff | 24.05968339375419
avg_opponent_elo_diff_last_10 | 4.834279411156228
neutral | 4.609280601462259
concede_rate_diff | 3.9225103263299976
avg_goals_conceded_diff_last_10 | 3.6255899672991445
year | 3.12136630715972
tournament | 3.0626285332903773
avg_goals_conceded_diff_last_5 | 2.8838665707607216
team_2_goal_rate | 2.836424078975415
team_2_concede_rate | 2.5232975972071023
team_2_elo | 2.4063249842907553
tournament_type_group | 2.2096927390370795
team_1_avg_opponent_elo_last_10 | 2.1665642809925925
team_1_goal_rate | 2.107528748426042
team_2_avg_goals_conceded_last_10 | 2.073176468187467
```

## Version 2 Feature Families

- Recent result form: wins, draws, and losses in each team's last 5 and last 10 matches.
- Recent points form: football points over last 5 and last 10, plus team-difference columns.
- Recent attacking/defensive form: average goals scored and conceded over last 5 and last 10.
- Opponent strength: average pre-match rolling Elo of previous opponents over last 5 and last 10.
- Tournament context: `tournament_type_group` and `tournament_importance_weight`.

## Leakage Controls

- Rolling Elo, form, goal rate, concede rate, and opponent-strength features are computed before each match is added to team history.
- The default training mode does not use latest FIFA snapshots as historical match features.
- Use `--allow-latest-rating-features` only for experiments where you accept that leakage risk.