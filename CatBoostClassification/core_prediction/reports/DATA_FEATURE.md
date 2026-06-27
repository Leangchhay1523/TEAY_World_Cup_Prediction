# Dataset And Feature Guide

This guide documents the data used by the existing `core_prediction` system after
the Version 2 feature upgrade. Folder names and output paths are unchanged.

## Raw Data Inputs

### `data/raw/results.csv`

Historical international match results. Version 2 uses this file to generate
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
| `recent_form` | Kept in the file, but Version 2 model form features come from `results.csv`. |

### `data/raw/fifa_rankings.csv`

Latest FIFA ranking snapshot used for team coverage validation and optional
snapshot features.

| Column | Use |
| --- | --- |
| `team` | Team identity after normalization. |
| `rank` | Current FIFA rank feature when latest snapshot features are enabled. Lower is stronger. |
| `previous_rank` | Previous FIFA rank used for ranking momentum. |
| `ranking_move` | Rank movement value used for ranking momentum. |
| `points` | Current FIFA points feature when latest snapshot features are enabled. Higher is stronger. |
| `previous_points` | Previous FIFA points value used to calculate points momentum. |
| `rated_matches` | Number of FIFA-rated matches used as an experience feature. |
| `confederation` | Categorical team-context feature when latest snapshot features are enabled. |

## Base Model Features

These features are used by the CatBoost outcome classifier and both CatBoost goal regressors:

| Feature | Meaning |
| --- | --- |
| `team_1_elo`, `team_2_elo`, `elo_diff` | Rolling Elo in training; latest Elo snapshot in prediction. |
| `elo_ratio` | Team 1 Elo divided by Team 2 Elo, using the same Elo source as `team_1_elo` and `team_2_elo`. |
| `team_1_fifa_rank`, `team_2_fifa_rank`, `fifa_rank_diff` | FIFA rank features. Disabled by default in historical training to avoid latest-snapshot leakage. |
| `team_1_fifa_points`, `team_2_fifa_points`, `fifa_points_diff` | FIFA points features. Disabled by default in historical training to avoid latest-snapshot leakage. |
| `team_1_previous_fifa_rank`, `team_2_previous_fifa_rank` | Previous FIFA rank for each team. |
| `team_1_ranking_move`, `team_2_ranking_move`, `ranking_move_diff` | FIFA ranking movement for each team and Team 1 minus Team 2 movement. |
| `team_1_previous_fifa_points`, `team_2_previous_fifa_points` | Previous FIFA points for each team. |
| `team_1_points_change`, `team_2_points_change`, `points_change_diff` | Current points minus previous points for each team, plus Team 1 minus Team 2 change. |
| `team_1_rated_matches`, `team_2_rated_matches`, `rated_matches_diff` | FIFA-rated match experience for each team and Team 1 minus Team 2 experience. |
| `fifa_rank_ratio` | `team_1_fifa_rank / max(team_2_fifa_rank, 1)`. Lower values favor Team 1 because FIFA ranks are ordinal. |
| `fifa_points_ratio` | `team_1_fifa_points / max(team_2_fifa_points, 1)`. Higher values favor Team 1. |
| `team_1_top10`, `team_2_top10` | Binary flags: `1` when rank is `<= 10`, otherwise `0` when rank is available. |
| `team_1_top20`, `team_2_top20` | Binary flags: `1` when rank is `<= 20`, otherwise `0` when rank is available. |
| `team_1_top30`, `team_2_top30` | Binary flags: `1` when rank is `<= 30`, otherwise `0` when rank is available. |
| `team_1_top50`, `team_2_top50` | Binary flags: `1` when rank is `<= 50`, otherwise `0` when rank is available. |
| `team_1_recent_form`, `team_2_recent_form`, `recent_form_diff` | Backward-compatible aliases for last-5 football points form. |
| `team_1_goal_rate`, `team_2_goal_rate`, `goal_rate_diff` | Goals scored per prior match. |
| `team_1_concede_rate`, `team_2_concede_rate`, `concede_rate_diff` | Goals conceded per prior match. |
| `tournament` | Detailed tournament name in training; `FIFA World Cup` for 2026 fixture prediction. |
| `neutral` | Whether the match is at a neutral venue. |
| `year` | Match year. |
| `team_1_confederation`, `team_2_confederation` | FIFA confederation categories. |

## Version 2 Features

All rolling Version 2 features are computed before the current match is added
to team history. The current match result is never used to create its own row.

## Model Use

The same ordered feature list is used by all three active CatBoost models:

- `catboost_outcome_model.cbm` is a CatBoostClassifier that predicts only `team_1_win`, `draw`, and `team_2_win`.
- `catboost_goals_team_1.cbm` is a CatBoostRegressor that predicts `expected_goals_team_1` from the `goals_team_1` target.
- `catboost_goals_team_2.cbm` is a CatBoostRegressor that predicts `expected_goals_team_2` from the `goals_team_2` target.

The goal regressors do not choose the winner directly. Their CatBoost xG is
blended with a statistical xG baseline from the same feature row. The final
ensemble xG is calibrated by the saved `goal_scale`, clamped to `0.05` through
`6.00`, and used as the Poisson lambda. Poisson converts those lambdas into a
`0-0` through `6-6` score matrix. The default selector chooses the highest
expected competition points, combining outcome, goal-difference, and exact-score
probabilities; pure score-probability mode is available only when requested.

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

### Match Rate And Goal Difference Features

These features use the same pre-match rolling state. Rates are computed over
the available prior matches in the requested window, so a team with three prior
matches uses those three matches for its last-5 rate. Teams with no prior
matches receive missing values for rate and average features.

| Feature | Meaning |
| --- | --- |
| `elo_ratio` | Team 1 Elo divided by Team 2 Elo. |
| `team_1_draw_rate_last_5`, `team_2_draw_rate_last_5` | Share of each team's previous 5 matches that were draws. |
| `team_1_draw_rate_last_10`, `team_2_draw_rate_last_10` | Share of each team's previous 10 matches that were draws. |
| `draw_rate_diff_last_5`, `draw_rate_diff_last_10` | Team 1 minus Team 2 draw rate for each window. |
| `team_1_clean_sheet_rate_last_5`, `team_2_clean_sheet_rate_last_5` | Share of each team's previous 5 matches with zero goals conceded. |
| `team_1_clean_sheet_rate_last_10`, `team_2_clean_sheet_rate_last_10` | Share of each team's previous 10 matches with zero goals conceded. |
| `clean_sheet_rate_diff_last_5`, `clean_sheet_rate_diff_last_10` | Team 1 minus Team 2 clean-sheet rate for each window. |
| `team_1_failed_to_score_rate_last_5`, `team_2_failed_to_score_rate_last_5` | Share of each team's previous 5 matches with zero goals scored. |
| `team_1_failed_to_score_rate_last_10`, `team_2_failed_to_score_rate_last_10` | Share of each team's previous 10 matches with zero goals scored. |
| `failed_to_score_rate_diff_last_5`, `failed_to_score_rate_diff_last_10` | Team 1 minus Team 2 failed-to-score rate for each window. Lower is better in attack. |
| `team_1_avg_goal_difference_last_5`, `team_2_avg_goal_difference_last_5` | Average goals-for minus goals-against across each team's previous 5 matches. |
| `team_1_avg_goal_difference_last_10`, `team_2_avg_goal_difference_last_10` | Average goals-for minus goals-against across each team's previous 10 matches. |
| `avg_goal_difference_diff_last_5`, `avg_goal_difference_diff_last_10` | Team 1 minus Team 2 average goal difference for each window. |

### Total Goals And Volatility Features

These features use only `data/raw/results.csv` and the same pre-match rolling
state as the other form features. Over/under rates are based on total goals in
previous matches. Goal standard deviation features use previous goals scored or
conceded and are set to `0` when a team has fewer than two previous matches in
the requested window.

| Feature | Meaning |
| --- | --- |
| `team_1_over_2_5_rate_last_5`, `team_2_over_2_5_rate_last_5` | Share of each team's previous 5 matches with total goals greater than 2.5. |
| `team_1_over_2_5_rate_last_10`, `team_2_over_2_5_rate_last_10` | Share of each team's previous 10 matches with total goals greater than 2.5. |
| `over_2_5_rate_diff_last_5`, `over_2_5_rate_diff_last_10` | Team 1 minus Team 2 over-2.5 rate for each window. |
| `team_1_over_3_5_rate_last_5`, `team_2_over_3_5_rate_last_5` | Share of each team's previous 5 matches with total goals greater than 3.5. |
| `team_1_over_3_5_rate_last_10`, `team_2_over_3_5_rate_last_10` | Share of each team's previous 10 matches with total goals greater than 3.5. |
| `over_3_5_rate_diff_last_5`, `over_3_5_rate_diff_last_10` | Team 1 minus Team 2 over-3.5 rate for each window. |
| `team_1_under_2_5_rate_last_5`, `team_2_under_2_5_rate_last_5` | Share of each team's previous 5 matches with total goals less than 2.5. |
| `team_1_under_2_5_rate_last_10`, `team_2_under_2_5_rate_last_10` | Share of each team's previous 10 matches with total goals less than 2.5. |
| `under_2_5_rate_diff_last_5`, `under_2_5_rate_diff_last_10` | Team 1 minus Team 2 under-2.5 rate for each window. |
| `team_1_goals_scored_std_last_5`, `team_2_goals_scored_std_last_5` | Standard deviation of goals scored across each team's previous 5 matches. |
| `team_1_goals_scored_std_last_10`, `team_2_goals_scored_std_last_10` | Standard deviation of goals scored across each team's previous 10 matches. |
| `goals_scored_std_diff_last_5`, `goals_scored_std_diff_last_10` | Team 1 minus Team 2 goals-scored standard deviation for each window. |
| `team_1_goals_conceded_std_last_5`, `team_2_goals_conceded_std_last_5` | Standard deviation of goals conceded across each team's previous 5 matches. |
| `team_1_goals_conceded_std_last_10`, `team_2_goals_conceded_std_last_10` | Standard deviation of goals conceded across each team's previous 10 matches. |
| `goals_conceded_std_diff_last_5`, `goals_conceded_std_diff_last_10` | Team 1 minus Team 2 goals-conceded standard deviation for each window. |

### Matchup And Context Features

These features use only matches before the current row. Head-to-head values
come from previous matches between the two teams only. Neutral-ground features
use the `neutral` column from `data/raw/results.csv`. Home and away style
features use the `home_team` and `away_team` roles in the same historical file.

If no previous head-to-head match exists, all head-to-head rates and averages are
set to `0`, with `h2h_matches_count = 0`.

| Feature | Meaning |
| --- | --- |
| `h2h_matches_count` | Number of previous matches between Team 1 and Team 2. |
| `h2h_team_1_win_rate` | Share of previous head-to-head matches won by Team 1 from the current row perspective. |
| `h2h_team_2_win_rate` | Share of previous head-to-head matches won by Team 2 from the current row perspective. |
| `h2h_draw_rate` | Share of previous head-to-head matches that ended in a draw. |
| `h2h_avg_total_goals` | Average total goals in previous head-to-head matches. |
| `h2h_avg_goal_difference_team_1` | Average Team 1 goals minus Team 2 goals in previous head-to-head matches. |
| `h2h_last_match_goal_difference_team_1` | Team 1 goal difference in the most recent previous head-to-head match. |
| `team_1_neutral_win_rate_last_10`, `team_2_neutral_win_rate_last_10` | Win rate in each team's previous 10 neutral-ground matches. |
| `team_1_neutral_goal_rate_last_10`, `team_2_neutral_goal_rate_last_10` | Average goals scored in each team's previous 10 neutral-ground matches. |
| `team_1_neutral_concede_rate_last_10`, `team_2_neutral_concede_rate_last_10` | Average goals conceded in each team's previous 10 neutral-ground matches. |
| `neutral_win_rate_diff_last_10` | Team 1 minus Team 2 neutral win rate. |
| `neutral_goal_rate_diff_last_10` | Team 1 minus Team 2 neutral scoring rate. |
| `neutral_concede_rate_diff_last_10` | Team 1 minus Team 2 neutral concede rate. Lower is better defensively. |
| `team_1_home_goal_rate_last_10`, `team_2_home_goal_rate_last_10` | Average goals scored in each team's previous 10 matches as the listed home team. |
| `team_1_away_goal_rate_last_10`, `team_2_away_goal_rate_last_10` | Average goals scored in each team's previous 10 matches as the listed away team. |
| `team_1_home_win_rate_last_10`, `team_2_home_win_rate_last_10` | Win rate in each team's previous 10 matches as the listed home team. |
| `team_1_away_win_rate_last_10`, `team_2_away_win_rate_last_10` | Win rate in each team's previous 10 matches as the listed away team. |
| `team_1_attack_vs_team_2_defense_ratio_last_10` | `team_1_avg_goals_scored_last_10 / max(team_2_avg_goals_conceded_last_10, 0.1)`. |
| `team_2_attack_vs_team_1_defense_ratio_last_10` | `team_2_avg_goals_scored_last_10 / max(team_1_avg_goals_conceded_last_10, 0.1)`. |
| `attack_defense_ratio_diff_last_10` | Team 1 attack-defense ratio minus Team 2 attack-defense ratio. |

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
| CatBoost team 1 goal model | `CatBoostClassification/core_prediction/models/catboost_goals_team_1.cbm` |
| CatBoost team 2 goal model | `CatBoostClassification/core_prediction/models/catboost_goals_team_2.cbm` |
| Goal ensemble config | `CatBoostClassification/core_prediction/models/goal_ensemble_config.json` |
| Score selection config | `CatBoostClassification/core_prediction/models/score_selection_config.json` |
| Feature importance | `CatBoostClassification/core_prediction/outputs/feature_importance_v2.csv` |
| Goal ensemble tuning results | `CatBoostClassification/core_prediction/outputs/goal_ensemble_tuning_results.csv` |
| Score selection tuning results | `CatBoostClassification/core_prediction/outputs/score_selection_tuning_results.csv` |
| Metrics | `CatBoostClassification/core_prediction/outputs/training_metrics_v2.json` |
| Single-match predictions | `CatBoostClassification/core_prediction/outputs/version_2_predictions.csv` |
