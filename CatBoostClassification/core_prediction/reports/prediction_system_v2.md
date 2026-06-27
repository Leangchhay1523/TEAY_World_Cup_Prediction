# Version 2 Prediction System

This document explains `core_prediction/scripts/predict_single_match_v2.py`.

## Active Method

```text
CatBoostClassifier + CatBoostRegressor/Statistical xG Ensemble + Expected-Points Score Selection
```

## Required Inputs

- `core_prediction/models/catboost_outcome_model.cbm`
- `core_prediction/models/catboost_goals_team_1.cbm`
- `core_prediction/models/catboost_goals_team_2.cbm`
- `core_prediction/models/goal_ensemble_config.json`
- `core_prediction/models/score_selection_config.json`
- `core_prediction/outputs/training_metrics_v2.json`
- `data/raw/elo_ratings.csv`
- `data/raw/fifa_rankings.csv`
- `data/raw/results.csv`
- `data/worldcup_2026_fixtures/worldcup_2026_fixtures_cleaned.csv`
- `data/worldcup_2026_fixtures/future_match.csv`

## Prediction Steps

1. Validate the fixture date, stage, teams, and rating coverage.
2. Reject unresolved placeholders such as `TBD`, `1A`, `W101`, `Winner Group A`, and playoff placeholders.
3. Build the same ordered Version 2 feature row used during training.
4. Load the CatBoostClassifier and predict:
   - `team_1_win_probability`
   - `draw_probability`
   - `team_2_win_probability`
5. Load both CatBoostRegressor goal models and predict CatBoost xG.
6. Estimate statistical xG from goal rates, recent goal form, opponent concede rates, Elo difference, FIFA points when available, and tournament importance.
7. Load `goal_ensemble_config.json` and blend CatBoost xG with statistical xG.
8. Load `score_selection_config.json` and apply `goal_scale`.
9. Clamp calibrated xG to the range `0.05` through `6.00`.
10. Use calibrated xG as Poisson lambdas.
11. Generate a normalized `0-0` through `6-6` score matrix with 49 candidates.
12. By default, select the candidate with the highest expected competition points:

```text
expected_points =
  3 * outcome_probability
  + 2 * goal_difference_probability
  + 5 * poisson_score_probability
```

13. Pure exact-score probability selection remains available with `--decision_mode score_probability`.
14. The selected scoreline directly determines winner and goal difference.

## Model Responsibilities

- The outcome CatBoostClassifier predicts only winner/draw probabilities.
- The two CatBoostRegressor goal models predict expected goals only.
- Statistical xG provides a stable baseline.
- The xG ensemble combines CatBoost xG and statistical xG with the saved tuned weight.
- Goal-scale calibration adjusts ensemble xG before Poisson scoring.
- Poisson converts calibrated xG into exact-score probabilities.
- Default score selection is `argmax(expected_competition_points)`.
- Pure exact-score probability selection remains available only with `--decision_mode score_probability`.

## Decision Modes

Default:

```powershell
python CatBoostClassification/core_prediction/scripts/predict_single_match_v2.py --team_1 Brazil --team_2 Scotland --stage "Group Stage" --match_date 2026-06-24
```

Optional pure score-probability mode:

```powershell
python CatBoostClassification/core_prediction/scripts/predict_single_match_v2.py --team_1 Brazil --team_2 Scotland --stage "Group Stage" --match_date 2026-06-24 --decision_mode score_probability
```

## Debug Output

The CLI prints normalized team names, fixture validation, the active training feature contract, outcome probabilities, CatBoost xG, statistical xG, ensemble xG, goal scale, calibrated xG, top 5 scorelines by score probability, top 5 scorelines by expected points, final decision mode, and the final selected prediction.

## Output

Predictions are saved to:

```text
core_prediction/outputs/version_2_predictions.csv
```

Columns:

- `match_date`
- `team_1`
- `team_2`
- `stage`
- `predicted_winner`
- `predicted_score`
- `predicted_goal_difference`
- `confidence`
- `team_1_win_probability`
- `draw_probability`
- `team_2_win_probability`
- `expected_goals_team_1`
- `expected_goals_team_2`
- `catboost_xg_team_1`
- `catboost_xg_team_2`
- `statistical_xg_team_1`
- `statistical_xg_team_2`
- `ensemble_xg_team_1`
- `ensemble_xg_team_2`
- `goal_ensemble_weight_catboost`
- `goal_ensemble_weight_statistical`
- `decision_mode`
- `goal_scale`
- `calibrated_xg_team_1`
- `calibrated_xg_team_2`
- `selected_score_probability`
- `predicted_total_goals`
- `expected_competition_points`
- `prediction_method`
- `explanation`

Confidence is the selected winner/outcome probability reported from 0 to 100. `selected_score_probability` remains the exact-score probability.
