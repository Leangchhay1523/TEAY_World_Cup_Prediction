# Version 2 Architecture

Version 2 is now a hybrid system:

```text
CatBoostClassifier outcome probabilities
        +
rating/statistical expected goals
        +
Poisson score matrix
        +
candidate expected-points optimizer
```

## Why It Changed

The earlier Version 2 used CatBoostRegressor models for expected goals:

- `catboost_goals_team_1.cbm`
- `catboost_goals_team_2.cbm`

Those goal predictions were unstable, so they are no longer active. The files may remain in `models/` for archive/comparison, but prediction now ignores them.

## Active Training

`scripts/train_version_2_models.py` trains only:

```text
version_2_ml/models/catboost_outcome_model.cbm
```

Target:

- `team_1_win`
- `draw`
- `team_2_win`

The training split is time-aware: older matches train the model and newer matches validate it.

## Active Prediction

`scripts/predict_single_match_v2.py` predicts one match at a time.

Inputs:

- user `team_1`
- user `team_2`
- `stage`
- `match_date`

The script validates the fixture, loads live Elo/FIFA data, builds features, runs the outcome classifier, estimates goals with statistical logic, generates candidate scorelines, and saves the final answer.

## Expected Goals Logic

Expected goals are estimated using:

- team scoring rate
- opponent concede rate
- Elo difference
- FIFA points difference
- recent form difference
- neutral venue status

Expected goals are clamped to a realistic football range:

```text
minimum = 0.2
maximum = 4.5
```

## Candidate Optimizer

Every scoreline from `0-0` to `6-6` is a candidate.

Each candidate receives:

```text
expected_points =
  3 * outcome_probability
  + 2 * goal_difference_probability
  + 5 * exact_score_probability
```

The highest expected-points candidate becomes the final submitted prediction.

## Leakage Policy

Historical training features are built before each match is added to team history.

Latest FIFA snapshot features are disabled by default during historical training because using today's rankings for old matches would leak future information.
