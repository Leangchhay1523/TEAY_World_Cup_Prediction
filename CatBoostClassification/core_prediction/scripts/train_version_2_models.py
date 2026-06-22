"""Train the simplified Version 2 outcome model.

Active Version 2 architecture:
- CatBoostClassifier predicts match outcome only.
- Expected goals are not learned by CatBoost anymore.
- The prediction script estimates expected goals with rating/statistical logic
  and uses a Poisson score matrix.

The dataset builder remains leakage-aware. It computes pre-match rolling team
features from historical results before each match is added to team history.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from v25_feature_engineering import (
    CATEGORICAL_MODEL_FEATURES,
    MODEL_FEATURE_COLUMNS,
    RollingTeamState,
    build_form_feature_pair,
    get_tournament_importance_weight,
    get_tournament_type_group,
    load_results,
    outcome_label,
    safe_rate,
    standardize_team_name,
    update_states_after_match,
)

try:
    from catboost import CatBoostClassifier, Pool
except ImportError as exc:  # pragma: no cover - handled when training starts.
    CatBoostClassifier = None
    Pool = None
    CATBOOST_IMPORT_ERROR = exc
else:
    CATBOOST_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION_DIR = PROJECT_ROOT / "core_prediction"
DATA_ROOT = PROJECT_ROOT.parent / "data"

DEFAULT_RESULTS_PATHS = [
    DATA_ROOT / "raw" / "results.csv",
    PROJECT_ROOT / "version_1_baseline" / "data" / "results.csv",
]
DEFAULT_FIFA_PATHS = [
    DATA_ROOT / "raw" / "fifa_rankings.csv",
]

TRAINING_DATA_PATH = VERSION_DIR / "processed_data" / "training_dataset_v2.csv"
REPORT_PATH = VERSION_DIR / "reports" / "training_report_v2.md"
FEATURE_IMPORTANCE_PATH = VERSION_DIR / "outputs" / "feature_importance_v2.csv"
METRICS_PATH = VERSION_DIR / "outputs" / "training_metrics_v2.json"
OUTCOME_MODEL_PATH = VERSION_DIR / "models" / "catboost_outcome_model.cbm"

def find_existing_path(candidates: list[Path], label: str, required: bool = True) -> Path | None:
    """Return the first existing path from accepted locations."""

    for path in candidates:
        if path.exists():
            return path
    if required:
        checked = "\n".join(f"- {path}" for path in candidates)
        raise FileNotFoundError(f"Could not find {label}. Checked:\n{checked}")
    return None


def load_latest_fifa(path: Path | None) -> pd.DataFrame:
    """Load latest FIFA rankings for optional experimental features."""

    if path is None:
        return pd.DataFrame()
    fifa = pd.read_csv(path)
    if "team" not in fifa.columns:
        return pd.DataFrame()
    fifa = fifa.copy()
    fifa["team_key"] = fifa["team"].map(standardize_team_name)
    return fifa


def get_fifa_feature(fifa: pd.DataFrame, team: str, column: str) -> Any:
    """Read a latest FIFA feature. Disabled by default to avoid leakage."""

    if fifa.empty or column not in fifa.columns:
        return np.nan
    match = fifa[fifa["team_key"].eq(team)]
    if match.empty:
        return np.nan
    return match.iloc[0][column]


def build_training_dataset(
    results: pd.DataFrame,
    fifa_table: pd.DataFrame | None = None,
    allow_latest_rating_features: bool = False,
) -> pd.DataFrame:
    """Build one row per historical match with pre-match features."""

    fifa_table = fifa_table if fifa_table is not None else pd.DataFrame()
    states: dict[str, RollingTeamState] = {}
    rows: list[dict[str, Any]] = []

    for match in results.itertuples(index=False):
        team_1 = match.home_team
        team_2 = match.away_team
        goals_1 = int(match.home_score)
        goals_2 = int(match.away_score)
        state_1 = states.setdefault(team_1, RollingTeamState())
        state_2 = states.setdefault(team_2, RollingTeamState())

        team_1_goal_rate = safe_rate(state_1.goals_for, state_1.matches)
        team_2_goal_rate = safe_rate(state_2.goals_for, state_2.matches)
        team_1_concede_rate = safe_rate(state_1.goals_against, state_1.matches)
        team_2_concede_rate = safe_rate(state_2.goals_against, state_2.matches)
        form_features = build_form_feature_pair(state_1, state_2)
        tournament_type_group = get_tournament_type_group(match.tournament)

        row = {
            "date": match.date.date().isoformat(),
            "year": int(match.date.year),
            "team_1": team_1,
            "team_2": team_2,
            "stage": match.tournament,
            "tournament": match.tournament,
            "neutral": bool(match.neutral),
            "team_1_elo": state_1.elo,
            "team_2_elo": state_2.elo,
            "elo_diff": state_1.elo - state_2.elo,
            "team_1_recent_form": form_features["team_1_form_points_last_5"],
            "team_2_recent_form": form_features["team_2_form_points_last_5"],
            "recent_form_diff": form_features["form_points_diff_last_5"],
            "team_1_goal_rate": team_1_goal_rate,
            "team_2_goal_rate": team_2_goal_rate,
            "goal_rate_diff": team_1_goal_rate - team_2_goal_rate,
            "team_1_concede_rate": team_1_concede_rate,
            "team_2_concede_rate": team_2_concede_rate,
            "concede_rate_diff": team_1_concede_rate - team_2_concede_rate,
            "tournament_type_group": tournament_type_group,
            "tournament_importance_weight": get_tournament_importance_weight(tournament_type_group),
            "goals_team_1": goals_1,
            "goals_team_2": goals_2,
            "target_outcome": outcome_label(goals_1, goals_2),
        }
        row.update(form_features)

        if allow_latest_rating_features:
            rank_1 = get_fifa_feature(fifa_table, team_1, "rank")
            rank_2 = get_fifa_feature(fifa_table, team_2, "rank")
            points_1 = get_fifa_feature(fifa_table, team_1, "points")
            points_2 = get_fifa_feature(fifa_table, team_2, "points")
            row.update(
                {
                    "team_1_fifa_rank": rank_1,
                    "team_2_fifa_rank": rank_2,
                    "fifa_rank_diff": rank_1 - rank_2,
                    "team_1_fifa_points": points_1,
                    "team_2_fifa_points": points_2,
                    "fifa_points_diff": points_1 - points_2,
                    "team_1_confederation": get_fifa_feature(fifa_table, team_1, "confederation"),
                    "team_2_confederation": get_fifa_feature(fifa_table, team_2, "confederation"),
                }
            )
        else:
            row.update(
                {
                    "team_1_fifa_rank": np.nan,
                    "team_2_fifa_rank": np.nan,
                    "fifa_rank_diff": np.nan,
                    "team_1_fifa_points": np.nan,
                    "team_2_fifa_points": np.nan,
                    "fifa_points_diff": np.nan,
                    "team_1_confederation": "unknown",
                    "team_2_confederation": "unknown",
                }
            )

        rows.append(row)
        update_states_after_match(state_1, state_2, goals_1, goals_2)

    dataset = pd.DataFrame(rows)
    dataset["neutral"] = dataset["neutral"].astype(str)
    return dataset


def model_features(dataset: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return active classifier features and categorical feature names."""

    features = [name for name in MODEL_FEATURE_COLUMNS if name in dataset.columns]
    categorical = [name for name in CATEGORICAL_MODEL_FEATURES if name in features]
    return features, categorical


def time_aware_split(dataset: pd.DataFrame, train_fraction: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on older matches and validate on newer matches."""

    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0.5 and 1.0")
    dataset = dataset.sort_values(["date", "team_1", "team_2"]).reset_index(drop=True)
    split_index = max(1, min(len(dataset) - 1, int(len(dataset) * train_fraction)))
    return dataset.iloc[:split_index].copy(), dataset.iloc[split_index:].copy()


def train_models(
    dataset: pd.DataFrame,
    train_fraction: float = 0.8,
    random_seed: int = 42,
    iterations: int = 500,
) -> dict[str, Any]:
    """Train only the CatBoost outcome classifier."""

    if CATBOOST_IMPORT_ERROR is not None:
        raise RuntimeError(
            "CatBoost is required for Version 2 training. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from CATBOOST_IMPORT_ERROR

    features, categorical = model_features(dataset)
    train_df, valid_df = time_aware_split(dataset, train_fraction=train_fraction)
    x_train = train_df[features]
    x_valid = valid_df[features]
    y_train = train_df["target_outcome"]
    y_valid = valid_df["target_outcome"]
    categorical_indices = [features.index(name) for name in categorical]

    train_pool = Pool(x_train, y_train, cat_features=categorical_indices)
    valid_pool = Pool(x_valid, y_valid, cat_features=categorical_indices)
    outcome_model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="Accuracy",
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        random_seed=random_seed,
        train_dir=str(VERSION_DIR / "outputs" / "catboost_info_classifier"),
        verbose=100,
    )
    outcome_model.fit(train_pool, eval_set=valid_pool)

    predictions = outcome_model.predict(x_valid).reshape(-1)
    probabilities = outcome_model.predict_proba(x_valid)
    labels_seen = list(outcome_model.classes_)
    metrics = {
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "train_start": str(train_df["date"].min()),
        "train_end": str(train_df["date"].max()),
        "validation_start": str(valid_df["date"].min()),
        "validation_end": str(valid_df["date"].max()),
        "outcome_accuracy": float(accuracy_score(y_valid, predictions)),
        "outcome_log_loss": float(log_loss(y_valid, probabilities, labels=labels_seen)),
    }

    importance = pd.DataFrame(
        {
            "feature": features,
            "outcome_importance": outcome_model.get_feature_importance(),
        }
    ).sort_values("outcome_importance", ascending=False)

    return {
        "features": features,
        "categorical_features": categorical,
        "metrics": metrics,
        "feature_importance": importance,
        "outcome_model": outcome_model,
    }


def save_training_outputs(dataset: pd.DataFrame, training_result: dict[str, Any], allow_latest: bool) -> None:
    """Save dataset, classifier, feature importance, and report."""

    VERSION_DIR.joinpath("models").mkdir(parents=True, exist_ok=True)
    VERSION_DIR.joinpath("processed_data").mkdir(parents=True, exist_ok=True)
    VERSION_DIR.joinpath("outputs").mkdir(parents=True, exist_ok=True)
    VERSION_DIR.joinpath("reports").mkdir(parents=True, exist_ok=True)

    dataset.to_csv(TRAINING_DATA_PATH, index=False)
    training_result["outcome_model"].save_model(OUTCOME_MODEL_PATH)
    training_result["feature_importance"].to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    metrics = training_result["metrics"]
    top_features = training_result["feature_importance"].head(15)
    report = [
        "# Version 2 Training Report",
        "",
        "## Summary",
        "",
        "- Active goal model: rating/statistical Poisson logic in prediction script",
        "- Active trained model: CatBoostClassifier for match outcome only",
        "- Version 2 upgrade: leakage-safe recent form, goals, opponent-strength, and tournament-type features",
        "- Deprecated: CatBoostRegressor goal models are preserved if present but not used",
        "- Split strategy: time-aware split, older matches for training and newer matches for validation",
        f"- Latest rating snapshot features used in historical training: `{allow_latest}`",
        "",
        "## Data",
        "",
        f"- Training rows: {metrics['train_rows']}",
        f"- Validation rows: {metrics['validation_rows']}",
        f"- Training period: {metrics['train_start']} to {metrics['train_end']}",
        f"- Validation period: {metrics['validation_start']} to {metrics['validation_end']}",
        f"- Processed dataset: `{TRAINING_DATA_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "## Evaluation",
        "",
        f"- Outcome accuracy: {metrics['outcome_accuracy']:.4f}",
        f"- Outcome log loss: {metrics['outcome_log_loss']:.4f}",
        "",
        "## Saved Model",
        "",
        f"- `{OUTCOME_MODEL_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "## Most Important Features",
        "",
        "```text",
        top_features.to_csv(index=False, lineterminator="\n").replace(",", " | ").strip(),
        "```",
        "",
        "## Version 2 Feature Families",
        "",
        "- Recent result form: wins, draws, and losses in each team's last 5 and last 10 matches.",
        "- Recent points form: football points over last 5 and last 10, plus team-difference columns.",
        "- Recent attacking/defensive form: average goals scored and conceded over last 5 and last 10.",
        "- Opponent strength: average pre-match rolling Elo of previous opponents over last 5 and last 10.",
        "- Tournament context: `tournament_type_group` and `tournament_importance_weight`.",
        "",
        "## Leakage Controls",
        "",
        "- Rolling Elo, form, goal rate, concede rate, and opponent-strength features are computed before each match is added to team history.",
        "- The default training mode does not use latest FIFA snapshots as historical match features.",
        "- Use `--allow-latest-rating-features` only for experiments where you accept that leakage risk.",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    metadata = {
        "metrics": metrics,
        "features": training_result["features"],
        "categorical_features": training_result["categorical_features"],
        "version_2_5_feature_note": (
            "Rolling result counts, points, goals, opponent Elo, and tournament type are computed "
            "from prior matches only."
        ),
        "allow_latest_rating_features": allow_latest,
        "active_model_paths": {
            "outcome": str(OUTCOME_MODEL_PATH.relative_to(PROJECT_ROOT)),
        },
        "deprecated_model_note": "Old CatBoost goal model files may remain in models/ but are not used.",
    }
    METRICS_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Train Version 2 CatBoost outcome model.")
    parser.add_argument("--results-path", type=Path, default=None, help="Historical results CSV.")
    parser.add_argument("--fifa-path", type=Path, default=None, help="Latest FIFA ranking CSV.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Chronological train fraction.")
    parser.add_argument("--iterations", type=int, default=500, help="CatBoost iterations.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--allow-latest-rating-features",
        action="store_true",
        help="Use current FIFA snapshot features in historical rows. This may leak future information.",
    )
    return parser.parse_args()


def main() -> None:
    """Run training end to end."""

    args = parse_args()
    results_path = args.results_path or find_existing_path(DEFAULT_RESULTS_PATHS, "historical results")
    fifa_path = args.fifa_path or find_existing_path(DEFAULT_FIFA_PATHS, "latest FIFA rankings", required=False)
    results = load_results(results_path)
    latest_fifa = load_latest_fifa(fifa_path)

    print(f"Loaded historical results: {results_path}")
    print(f"Rows: {len(results):,}")
    if args.allow_latest_rating_features:
        print("WARNING: latest FIFA snapshot features are being used in historical rows.")
    else:
        print("Leakage-safe mode: latest FIFA snapshot features are loaded but not used for historical rows.")

    dataset = build_training_dataset(
        results,
        fifa_table=latest_fifa,
        allow_latest_rating_features=args.allow_latest_rating_features,
    )
    print(f"Built training dataset: {dataset.shape[0]:,} rows x {dataset.shape[1]:,} columns")

    training_result = train_models(
        dataset,
        train_fraction=args.train_fraction,
        random_seed=args.random_seed,
        iterations=args.iterations,
    )
    save_training_outputs(dataset, training_result, allow_latest=args.allow_latest_rating_features)

    print(f"Saved training dataset: {TRAINING_DATA_PATH}")
    print(f"Saved outcome model: {OUTCOME_MODEL_PATH}")
    print("Old goal-regressor .cbm files are not deleted, but Version 2 no longer uses them.")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
