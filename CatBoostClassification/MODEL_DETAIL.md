# CatBoostClassification — Full Technical Reference

## 1. Overview

CatBoostClassification is a **hybrid prediction system** for FIFA World Cup 2026 matches. It combines:

- **CatBoostClassifier** (gradient boosting) — predicts match outcome probabilities (win/draw/loss)
- **Rating-based expected goals** — estimates scoring rates using Elo, FIFA, and form statistics
- **Poisson score matrix** — generates exact-score probabilities from expected goals
- **Candidate optimizer** — selects the final prediction by maximizing expected competition points

The system is trained on ~39,500 historical international matches (1872–2016) and validated on ~9,900 newer matches (2016–2026), achieving **~59.8% outcome accuracy** and **0.88 log loss**.

---

## 2. The Model: CatBoostClassifier

### 2.1 What is CatBoost?

CatBoost (Categorical Boosting) is a gradient boosting algorithm developed by Yandex. It builds an ensemble of decision trees sequentially, where each new tree corrects the errors of the previous ones. CatBoost's key differentiators are:

- **Native categorical feature support** — handles string/categorical columns without manual one-hot encoding
- **Ordered boosting** — prevents target leakage during training by using permutations
- **Symmetric trees** — builds oblivious (level-wise symmetric) decision trees by default
- **GPU training** — native CUDA support

### 2.2 Model Configuration

The project uses `CatBoostClassifier` with:

| Parameter       | Value        | Rationale                                                     |
| --------------- | ------------ | ------------------------------------------------------------- |
| `loss_function` | `MultiClass` | 3 mutually exclusive outcomes: team_1_win, draw, team_2_win   |
| `eval_metric`   | `Accuracy`   | 直观的分类评估指标                                            |
| `iterations`    | `500`        | Number of boosting rounds (configurable via `--iterations`)   |
| `learning_rate` | `0.05`       | Step size shrinkage — lower = more robust, slower convergence |
| `depth`         | `6`          | Tree depth — controls interaction depth between features      |
| `random_seed`   | `42`         | Reproducibility                                               |

### 2.3 Training Pipeline

**File:** `version_2_ml/scripts/train_version_2_models.py`

```
Historical results → build_training_dataset() → time_aware_split(80/20) → CatBoostClassifier.fit()
```

1. **Load results** — reads a CSV with columns: `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`, `neutral`
2. **Build features** — for each historical match, compute pre-match rolling stats (see §4 Features)
3. **Time-aware split** — sort by date; oldest 80% for training, newest 20% for validation
4. **Train** — create `Pool` objects with categorical feature indices, call `fit()`
5. **Save** — model to `.cbm` file, metrics JSON, feature importance CSV, training report

---

## 3. Algorithm: Gradient Boosting with CatBoost

### 3.1 Core Algorithm

For a dataset `D = {(x_i, y_i)}` with `n` samples:

1. **Initialize** model with a constant value: `F_0(x) = argmin_γ Σ L(y_i, γ)` where `L` is the loss function.

2. **For** `m = 1 to M` (500 iterations):
   - Compute the gradient (pseudo-residuals): `r_i = -[∂L(y_i, F(x_i)) / ∂F(x_i)]` evaluated at `F = F_{m-1}`
   - Fit a decision tree `h_m(x)` to predict the residuals `r_i`
   - Find optimal step size: `γ_m = argmin_γ Σ L(y_i, F_{m-1}(x_i) + γ * h_m(x_i))`
   - Update: `F_m(x) = F_{m-1}(x) + learning_rate * γ_m * h_m(x)`

3. **Return** `F_M(x)`

### 3.2 CatBoost-Specific Innovations

**Ordered Boosting** — Traditional gradient boosting suffers from target leakage because the same data points are used to compute residuals and build the tree. CatBoost uses random permutations of the training data so that each tree is built on a different subset, reducing overfitting.

**Oblivious Decision Trees** — CatBoost grows symmetric trees where the same split criterion is applied at every node of a given level. This reduces overfitting and speeds up inference, at the cost of some expressiveness.

**MultiClass loss** — For 3 classes, the loss is categorical cross-entropy:

```
L(y, p) = -Σ_{k=1}^{3} y_k * log(p_k)
```

where `y_k` is 1 for the true class and 0 otherwise, and `p_k` is the predicted probability (softmax of tree outputs).

### 3.3 Categorical Feature Handling

CatBoost uses **ordered target statistics** to encode categorical features:

1. Randomly permute the data rows
2. For each category, compute the target statistic (e.g., mean outcome) only from previous rows in the permutation
3. Apply a prior smoothing term to reduce noise for rare categories

This avoids the target leakage that plagues simple mean-target encoding.

---

## 4. Features

### 4.1 Full Feature List (23 features)

| Feature                | Type        | Description                                                 | Source                |
| ---------------------- | ----------- | ----------------------------------------------------------- | --------------------- |
| `team_1_elo`           | Numeric     | Pre-match rolling Elo rating of Team 1                      | Computed from results |
| `team_2_elo`           | Numeric     | Pre-match rolling Elo rating of Team 2                      | Computed from results |
| `elo_diff`             | Numeric     | `team_1_elo - team_2_elo` — the most important feature      | Derived               |
| `team_1_fifa_rank`     | Numeric     | Current FIFA rank of Team 1 (disabled by default)           | FIFA rankings CSV     |
| `team_2_fifa_rank`     | Numeric     | Current FIFA rank of Team 2 (disabled by default)           | FIFA rankings CSV     |
| `fifa_rank_diff`       | Numeric     | `team_1_fifa_rank - team_2_fifa_rank` (disabled by default) | Derived               |
| `team_1_fifa_points`   | Numeric     | Current FIFA points of Team 1                               | FIFA rankings CSV     |
| `team_2_fifa_points`   | Numeric     | Current FIFA points of Team 2                               | FIFA rankings CSV     |
| `fifa_points_diff`     | Numeric     | `team_1_fifa_points - team_2_fifa_points`                   | Derived               |
| `team_1_recent_form`   | Numeric     | Sum of competition points from Team 1's last 5 matches      | Computed from results |
| `team_2_recent_form`   | Numeric     | Sum of competition points from Team 2's last 5 matches      | Computed from results |
| `recent_form_diff`     | Numeric     | `team_1_recent_form - team_2_recent_form`                   | Derived               |
| `team_1_goal_rate`     | Numeric     | Team 1 goals scored per match (pre-match)                   | Computed from results |
| `team_2_goal_rate`     | Numeric     | Team 2 goals scored per match (pre-match)                   | Computed from results |
| `goal_rate_diff`       | Numeric     | `team_1_goal_rate - team_2_goal_rate`                       | Derived               |
| `team_1_concede_rate`  | Numeric     | Team 1 goals conceded per match (pre-match)                 | Computed from results |
| `team_2_concede_rate`  | Numeric     | Team 2 goals conceded per match (pre-match)                 | Computed from results |
| `concede_rate_diff`    | Numeric     | `team_1_concede_rate - team_2_concede_rate`                 | Derived               |
| `tournament`           | Categorical | Tournament name (e.g., "FIFA World Cup", "Friendly")        | Results CSV           |
| `neutral`              | Categorical | Whether the match was at a neutral venue ("True"/"False")   | Results CSV           |
| `year`                 | Numeric     | Year of the match                                           | Derived from date     |
| `team_1_confederation` | Categorical | Confederation of Team 1 (e.g., "UEFA", "CONMEBOL")          | FIFA rankings CSV     |
| `team_2_confederation` | Categorical | Confederation of Team 2                                     | FIFA rankings CSV     |

### 4.2 Feature Importance (Top 10)

| Feature               | Importance |
| --------------------- | ---------- |
| `elo_diff`            | 26.32      |
| `tournament`          | 9.38       |
| `concede_rate_diff`   | 7.70       |
| `year`                | 7.02       |
| `team_2_elo`          | 6.35       |
| `team_2_goal_rate`    | 5.51       |
| `neutral`             | 5.44       |
| `team_2_concede_rate` | 5.32       |
| `team_1_concede_rate` | 5.25       |
| `team_1_goal_rate`    | 4.68       |

### 4.3 Leakage Prevention

All rolling features (Elo, form, goal rate, concede rate) are computed **before** the current match's result is added to the team's history. This ensures:

- `team_1_elo` for a match in 2000 only uses data from matches before that date
- No future information leaks into training features

FIFA ranking snapshot features are **disabled by default** during historical training because using today's rankings for 1930 matches would severely leak future information. The `--allow-latest-rating-features` flag is available for experimental use only.

---

## 5. How Prediction Works

**File:** `version_2_ml/scripts/predict_single_match_v2.py`

The prediction workflow has **9 steps**:

### Step 1: Validate Fixture

```
team_1, team_2, stage, match_date → find_fixture()
```

- Normalize team names (e.g., "USA" → "United States", "Côte d'Ivoire" → "Cote d'Ivoire")
- Reject unresolved placeholders: `TBD`, `1A`, `W101`, `Winner Group A`, playoff winners
- Confirm fixture exists in `worldcup_2026_fixtures_cleaned.csv`
- Confirm both teams have live Elo and FIFA ranking data

### Step 2: Build Feature Row

```
build_feature_row(fixture, elo_df, fifa_df) → single-row DataFrame
```

Reads live Elo data and FIFA rankings to populate all 23 features for the specific match.

### Step 3: CatBoost Outcome Prediction

```
load_outcome_model() → CatBoostClassifier
predict_outcome_probabilities(model, feature_row) → {team_1_win: p1, draw: p2, team_2_win: p3}
```

The trained `catboost_outcome_model.cbm` is loaded and `predict_proba()` produces a 3-element probability vector summing to 1.0.

### Step 4: Estimate Expected Goals

```
estimate_expected_goals(feature_row) → (xG_team_1, xG_team_2)
```

A statistical formula (not CatBoost) blends:

- **Attacking base**: average of (team_1_goal_rate, team_2_concede_rate) for Team 1, and vice versa
- **Elo adjustment**: `0.22 * (elo_diff / 400)`
- **FIFA adjustment**: `0.12 * (fifa_points_diff / 350)`
- **Form adjustment**: `0.08 * (recent_form_diff / 300)`
- **Neutral modifier**: `+0.08` for the home team if not neutral
- **Strong mismatch bonus**: `+0.15` for the stronger team and `-0.08` for the weaker when `|elo_diff| > 200`

Results are clamped to `[0.2, 4.5]` to keep football-realistic.

### Step 5: Poisson Score Matrix

```
generate_score_matrix(xG_1, xG_2, max_goals=6) → 49-row DataFrame (0-0 through 6-6)
```

Uses the Poisson probability mass function:

```
P(k; λ) = (e^(-λ) * λ^k) / k!
```

For each scoreline (g1, g2):

```
p(g1, g2) = P(g1; xG_1) * P(g2; xG_2)
```

- **Normalized** so all 49 probabilities sum to 1.0
- Each row has: `goals_team_1`, `goals_team_2`, `predicted_score`, `poisson_score_probability`, `goal_difference`, `winner_from_score`

### Step 6: Candidate Scoring

```
generate_candidates(score_matrix, outcome_probabilities)
```

Each candidate scoreline receives an **expected competition points** score:

```
expected_points = 3 * outcome_probability
                + 2 * goal_difference_probability
                + 5 * exact_score_probability
```

| Component                          | Weight | Rationale                                                                   |
| ---------------------------------- | ------ | --------------------------------------------------------------------------- |
| `outcome_probability` (3×)         | 3.0    | The winner must be plausible — from CatBoost                                |
| `goal_difference_probability` (2×) | 2.0    | The margin must be plausible — sum of Poisson probabilities for the same GD |
| `exact_score_probability` (5×)     | 5.0    | The exact score must be the most likely within its GD group                 |

### Step 7: Select Winner

The candidate with the highest `expected_competition_points` is selected. This chooses the scoreline that is:

- Consonant with the CatBoost outcome prediction
- Statistically likely under the Poisson model
- The most probable exact score among those that satisfy the above

### Step 8: Derive Final Output

From the selected candidate:

- **Predicted winner**: maps `winner_from_score` to team name or "Draw"
- **Predicted score**: e.g. "2-1"
- **Predicted goal difference**: e.g. +1
- **Confidence**: winner probability from CatBoost × 100 (0–100)
- **Expected competition points**: the winning candidate's score

### Step 9: Save

Appended to `version_2_ml/outputs/version_2_predictions.csv` with full explanation.

---

## 6. Data Sources

### 6.1 Historical Results

**File:** `data/raw/results.csv`

| Column       | Type    | Example        |
| ------------ | ------- | -------------- |
| `date`       | date    | 2022-12-18     |
| `home_team`  | string  | Argentina      |
| `away_team`  | string  | France         |
| `home_score` | integer | 3              |
| `away_score` | integer | 3              |
| `tournament` | string  | FIFA World Cup |
| `neutral`    | boolean | True           |

Contains ~200 years of international football results (1872–present). Team names are standardized through a normalization pipeline that handles aliases (USA → United States, Korea Republic → South Korea, etc.) and unicode normalization.

### 6.2 Elo Ratings (Live)

**File:** `data/live_updates/elo_ratings.csv`

| Column          | Type    | Description                                  |
| --------------- | ------- | -------------------------------------------- |
| `team_name`     | string  | Country name                                 |
| `country_code`  | string  | Short code (e.g., "ES")                      |
| `elo`           | float   | Current Elo rating (1200–2200 range)         |
| `matches`       | integer | Total international matches played           |
| `goals_for`     | integer | Total goals scored                           |
| `goals_against` | integer | Total goals conceded                         |
| `recent_form`   | float   | Sum of Elo point changes from last 6 matches |
| `goal_rate`     | float   | `goals_for / matches` (computed on load)     |
| `concede_rate`  | float   | `goals_against / matches` (computed on load) |

### 6.3 FIFA Rankings (Live)

**File:** `data/live_updates/fifa_rankings.csv`

| Column          | Type    | Description                                   |
| --------------- | ------- | --------------------------------------------- |
| `team`          | string  | Country name                                  |
| `rank`          | integer | Current FIFA world rank                       |
| `points`        | float   | Current FIFA ranking points                   |
| `confederation` | string  | Continental federation (UEFA, CONMEBOL, etc.) |
| `previous_rank` | integer | Rank from last update                         |
| `ranking_move`  | integer | Change in rank                                |

### 6.4 Fixtures

**File:** `data/processed/worldcup_2026_fixtures_cleaned.csv`

| Column         | Type   | Example     |
| -------------- | ------ | ----------- |
| `date`         | date   | 2026-06-11  |
| `team_1`       | string | Mexico      |
| `team_2`       | string | Canada      |
| `stage`        | string | Group Stage |
| `host_country` | string | Canada      |

---

## 7. Expected Goals Formula (Detailed)

The expected goals formula at `predict_single_match_v2.py:456-490`:

```
base_1 = mean(team_1_goal_rate, team_2_concede_rate)
base_2 = mean(team_2_goal_rate, team_1_concede_rate)

elo_adj      = 0.22 * (elo_diff / 400)
fifa_adj     = 0.12 * (fifa_points_diff / 350)
form_adj     = 0.08 * (recent_form_diff / 300)
strength_adj = elo_adj + fifa_adj + form_adj

neutral_mod  = 0.0 if neutral else 0.08

xG_1 = base_1 + strength_adj + neutral_mod
xG_2 = base_2 - strength_adj

if elo_diff > 200:
    xG_1 += 0.15; xG_2 -= 0.08
elif elo_diff < -200:
    xG_1 -= 0.08; xG_2 += 0.15

clamp each to [0.2, 4.5]
```

The weights (0.22, 0.12, 0.08, 0.08, 0.15) were tuned to calibrate expected goals to realistic match values for World Cup-level competition.

---

## 8. Candidate Optimizer (Detailed)

At `predict_single_match_v2.py:535-552`:

For every (g1, g2) ∈ {0..6}²:

```
goal_diff_probability[gd] = sum of Poisson probabilities for all scorelines with that goal difference

outcome_probability(g1, g2) = CatBoost probability for the matching outcome class
goal_difference_probability(g1, g2) = goal_diff_probability[g1 - g2]
exact_score_probability(g1, g2) = P(g1; xG_1) * P(g2; xG_2)

expected_points =
    3.0 * outcome_probability(g1, g2)
  + 2.0 * goal_difference_probability(g1, g2)
  + 5.0 * exact_score_probability(g1, g2)
```

The candidate with the highest `expected_points` is the final prediction. This optimizer ensures that:

1. The predicted winner aligns with CatBoost's strongest outcome
2. The goal difference is plausible given the rating gap
3. The exact score is the most likely within that goal-difference group
4. All three signals are combined into a single, transparent objective

---

## 9. Training Metrics

| Metric            | Value           |
| ----------------- | --------------- |
| Training rows     | 39,546          |
| Validation rows   | 9,887           |
| Training period   | 1872–2016       |
| Validation period | 2016–2026       |
| Outcome accuracy  | 0.5976 (59.76%) |
| Outcome log loss  | 0.8801          |

Random baseline accuracy for 3-class balanced prediction is ~33.3%. The model's 59.8% demonstrates significant skill, driven primarily by Elo difference and tournament type.

---

## 10. Dependencies

**File:** `requirements.txt`

```
pandas
numpy
scikit-learn
catboost
```

---

## 11. Project Structure (CatBoostClassification)

```
CatBoostClassification/
  README.md
  PROJECT_STRUCTURE.md
  requirements.txt
  CATBOOST_DETAILED.md                              ← this document
  data/
    raw/
      results.csv                                   ← historical matches
      fifa_rankings.csv                             ← latest FIFA snapshot
      elo_ratings.csv                               ← latest Elo snapshot
    processed/
      worldcup_2026_fixtures_cleaned.csv             ← World Cup 2026 schedule
    live_updates/
      fifa_rankings.csv                             ← refresh before predictions
      elo_ratings.csv                               ← refresh before predictions
  version_2_ml/
    scripts/
      train_version_2_models.py                     ← CatBoost training pipeline
      predict_single_match_v2.py                    ← Hybrid prediction pipeline
    notebooks/
      train_version_2_models.ipynb                  ← Training notebook
      predict_single_match_v2.ipynb                 ← Prediction notebook
    models/
      catboost_outcome_model.cbm                    ← Trained CatBoostClassifier
    processed_data/
      training_dataset_v2.csv                       ← ML-ready training data
    outputs/
      version_2_predictions.csv                     ← Prediction results
      training_metrics_v2.json                      ← Evaluation metrics
      feature_importance_v2.csv                     ← Feature importance scores
      catboost_info_classifier/                     ← CatBoost training logs
    reports/
      version_2_architecture.md
      prediction_system_v2.md
      training_report_v2.md
      DATA_FEATURE.md
```

---

## 12. Running Instructions

### Train

```bash
python version_2_ml/scripts/train_version_2_models.py \
    --iterations 500 \
    --train-fraction 0.8
```

Optional:

- `--allow-latest-rating-features` — use FIFA snapshot in historical rows (⚠ leaks future data)
- `--results-path` — custom results CSV
- `--fifa-path` — custom FIFA ranking CSV

### Predict

```bash
python version_2_ml/scripts/predict_single_match_v2.py \
    --team_1 "Brazil" \
    --team_2 "Argentina" \
    --stage "Group Stage" \
    --match_date "2026-06-15"
```

### Install

```bash
pip install -r requirements.txt
```
