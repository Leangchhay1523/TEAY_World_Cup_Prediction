# Version 2 Prediction System

This document explains `version_2_ml/scripts/predict_single_match_v2.py`.

## Active Method

```text
CatBoost Outcome Model + Rating-Based Poisson Goal Model + Candidate Expected-Points Optimizer
```

The active system no longer uses CatBoostRegressor goal models.

## Required Inputs

- `version_2_ml/models/catboost_outcome_model.cbm`
- `data/live_updates/elo_ratings.csv`
- `data/live_updates/fifa_rankings.csv`
- `data/processed/worldcup_2026_fixtures_cleaned.csv`

## Prediction Steps

1. Validate the fixture.
2. Reject unresolved placeholders such as `TBD`, `1A`, `W101`, `Winner Group A`, and playoff placeholders.
3. Confirm both teams exist in the latest Elo and FIFA ranking CSVs.
4. Build the model feature row.
5. Use CatBoostClassifier to estimate:
   - `P(team_1_win)`
   - `P(draw)`
   - `P(team_2_win)`
6. Estimate expected goals with rating/statistical logic using:
   - Elo difference
   - FIFA points difference
   - goal scoring rates
   - concede rates
   - recent form
7. Generate a normalized Poisson score matrix from `0-0` to `6-6`.
8. Score every candidate with:

```text
expected_points =
  3 * outcome_probability
  + 2 * goal_difference_probability
  + 5 * exact_score_probability
```

9. Select the candidate with the highest expected competition points.

## Output

Predictions are saved to:

```text
version_2_ml/outputs/version_2_predictions.csv
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
- `expected_competition_points`
- `prediction_method`
- `explanation`

Confidence is based mainly on the selected winner probability from CatBoost and is reported from 0 to 100.
