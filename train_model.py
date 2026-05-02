from __future__ import annotations

import json
from pathlib import Path

from data_pipeline import build_training_dataset
from model import RidgeRegressor, regression_metrics, train_test_split


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2]

SPOTIFY_CSV = PROJECT_ROOT / "SpotifyAudioFeaturesApril2019.csv"
EMOTION_CSV = PROJECT_ROOT / "emotion_palette.csv"
DEAM_ROOT = ROOT / "data"

MODEL_PATH = ROOT / "sensory_model.json"
TRAINING_DATA_PATH = ROOT / "training_dataset.csv"
TOP_TRACKS_PATH = ROOT / "top_spotify_recommendations.csv"
REPORT_PATH = ROOT / "training_report.json"



def main() -> None:
    bundle = build_training_dataset(
        spotify_csv=SPOTIFY_CSV,
        emotion_palette_csv=EMOTION_CSV,
        deam_root=DEAM_ROOT,
    )
    dataset = bundle.data
    x = dataset[bundle.feature_columns].to_numpy(dtype=float)
    y = dataset[bundle.target_column].to_numpy(dtype=float)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_ratio=0.2, random_state=42)
    model = RidgeRegressor(feature_names=bundle.feature_columns, alpha=1.5).fit(x_train, y_train)

    train_metrics = regression_metrics(y_train, model.predict(x_train))
    test_metrics = regression_metrics(y_test, model.predict(x_test))

    model.save(MODEL_PATH, metrics={"train": train_metrics, "test": test_metrics})
    dataset.to_csv(TRAINING_DATA_PATH, index=False)

    top_tracks = (
        dataset[
            [
                "artist_name",
                "track_name",
                "track_id",
                "subconscious_buying_influence_score",
                "energy",
                "valence",
                "danceability",
                "tempo",
                "popularity",
            ]
        ]
        .sort_values("subconscious_buying_influence_score", ascending=False)
        .head(50)
    )
    top_tracks.to_csv(TOP_TRACKS_PATH, index=False)

    report = {
        "rows_used_for_training": int(len(dataset)),
        "feature_count": len(bundle.feature_columns),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "deam_reference": bundle.metadata,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Training complete.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
