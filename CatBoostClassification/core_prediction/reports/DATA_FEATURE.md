# Dataset And Feature Guide

This guide documents the data used by the existing `core_prediction` system after
the Version 2.5 feature upgrade. Folder names and output paths are unchanged.

## Raw Data Inputs

### `data/raw/results.csv`

Historical international match results. Version 2.5 uses this file to generate
leakage-safe rolling features.

Required columns:

| Column | Use |
| --- | --- |
| `date` | Chronological ordering and prediction cutoff date. |
| `home_team`, `away_team` | Team identity after common-name normalization. |
| `home_score`, `away_score` | Match outcome, points form, and goal form. |
| `tournament` | Tournament grouping and importance weight. |
| `neutral` | Neutral-site feature. |

### `data/raw/elo_ratings.csv`

Latest team-strength snapshot used during prediction.

| Column | Use |
| --- | --- |
| `team_name` | Team identity after normalization. |
| `elo` | Current team-strength feature. |
| `matches`, `goals_for`, `goals_against` | Fallback rate features if a team has no historical result state. |
| `recent_form` | Kept in the file, but Version 2.5 model form features come from `results.csv`. |

### `data/raw/fifa_rankings.csv`

Latest FIFA ranking snapshot used during prediction.

| Column | Use |
| --- | --- |
| `team` | Team identity after normalization. |
| `rank` | Current FIFA rank feature. Lower is stronger. |
| `points` | Current FIFA points feature. Higher is stronger. |
| `confederation` | Categorical team-context feature. |

## Base Model Features

These features are still used by the CatBoost outcome model:

| Feature | Meaning |
| --- | --- |
| `team_1_elo`, `team_2_elo`, `elo_diff` | Rolling Elo in training; latest Elo snapshot in prediction. |
| `team_1_fifa_rank`, `team_2_fifa_rank`, `fifa_rank_diff` | FIFA rank features. Disabled by default in historical training to avoid latest-snapshot leakage. |
| `team_1_fifa_points`, `team_2_fifa_points`, `fifa_points_diff` | FIFA points features. Disabled by default in historical training to avoid latest-snapshot leakage. |
| `team_1_recent_form`, `team_2_recent_form`, `recent_form_diff` | Backward-compatible aliases for last-5 football points form. |
| `team_1_goal_rate`, `team_2_goal_rate`, `goal_rate_diff` | Goals scored per prior match. |
| `team_1_concede_rate`, `team_2_concede_rate`, `concede_rate_diff` | Goals conceded per prior match. |
| `tournament` | Detailed tournament name in training; `FIFA World Cup` for 2026 fixture prediction. |
| `neutral` | Whether the match is at a neutral venue. |
| `year` | Match year. |
| `team_1_confederation`, `team_2_confederation` | FIFA confederation categories. |

## Version 2.5 Features

All rolling Version 2.5 features are computed before the current match is added
to team history. The current match result is never used to create its own row.

### Recent Result Form

Counts of wins, draws, and losses in each team's previous 5 and previous 10
matches:

| Features |
| --- |
| `team_1_wins_last_5`, `team_1_draws_last_5`, `team_1_losses_last_5` |
| `team_1_wins_last_10`, `team_1_draws_last_10`, `team_1_losses_last_10` |
| `team_2_wins_last_5`, `team_2_draws_last_5`, `team_2_losses_last_5` |
| `team_2_wins_last_10`, `team_2_draws_last_10`, `team_2_losses_last_10` |

### Recent Points Form

Football points are assigned as win = 3, draw = 1, loss = 0.

| Feature | Meaning |
| --- | --- |
| `team_1_form_points_last_5`, `team_1_form_points_last_10` | Team 1 points over recent windows. |
| `team_2_form_points_last_5`, `team_2_form_points_last_10` | Team 2 points over recent windows. |
| `form_points_diff_last_5`, `form_points_diff_last_10` | Team 1 minus Team 2 points form. |

### Recent Attacking And Defensive Form

Average goals scored and conceded in previous 5 and 10 matches:

| Feature family | Meaning |
| --- | --- |
| `team_1_avg_goals_scored_last_5`, `team_1_avg_goals_scored_last_10` | Team 1 recent attack. |
| `team_1_avg_goals_conceded_last_5`, `team_1_avg_goals_conceded_last_10` | Team 1 recent defense. |
| `team_2_avg_goals_scored_last_5`, `team_2_avg_goals_scored_last_10` | Team 2 recent attack. |
| `team_2_avg_goals_conceded_last_5`, `team_2_avg_goals_conceded_last_10` | Team 2 recent defense. |
| `avg_goals_scored_diff_last_5`, `avg_goals_scored_diff_last_10` | Team 1 minus Team 2 recent scoring. |
| `avg_goals_conceded_diff_last_5`, `avg_goals_conceded_diff_last_10` | Team 1 minus Team 2 recent goals conceded. Lower is better defensively. |

### Opponent Strength

For each historical match, the rolling Elo ratings are read before updating the
current result. Each team stores the opponent's pre-match rolling Elo, then
averages that history over 5 and 10 matches.

| Feature | Meaning |
| --- | --- |
| `team_1_avg_opponent_elo_last_5`, `team_1_avg_opponent_elo_last_10` | Average strength of Team 1's recent opponents. |
| `team_2_avg_opponent_elo_last_5`, `team_2_avg_opponent_elo_last_10` | Average strength of Team 2's recent opponents. |
| `avg_opponent_elo_diff_last_5`, `avg_opponent_elo_diff_last_10` | Team 1 minus Team 2 recent opponent strength. |

### Tournament Type

Detailed tournaments are grouped into a cleaner categorical feature:

| `tournament_type_group` | `tournament_importance_weight` |
| --- | ---: |
| `FIFA World Cup` | 1.00 |
| `Continental Championship` | 0.90 |
| `World Cup Qualification` | 0.80 |
| `Continental Qualification` | 0.70 |
| `Nations League` | 0.60 |
| `Friendly` | 0.30 |
| `Other` | 0.50 |

The importance value is used as a normal model feature, not as a sample weight.

## Saved Artifacts

| Artifact | Path |
| --- | --- |
| Training dataset | `CatBoostClassification/core_prediction/processed_data/training_dataset_v2.csv` |
| CatBoost outcome model | `CatBoostClassification/core_prediction/models/catboost_outcome_model.cbm` |
| Feature importance | `CatBoostClassification/core_prediction/outputs/feature_importance_v2.csv` |
| Metrics | `CatBoostClassification/core_prediction/outputs/training_metrics_v2.json` |
| Single-match predictions | `CatBoostClassification/core_prediction/outputs/version_2_predictions.csv` |
