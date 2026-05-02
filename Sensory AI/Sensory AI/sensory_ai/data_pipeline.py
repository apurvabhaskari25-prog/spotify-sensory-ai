from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from color_analysis import COLOR_TAGS, ColorEmotionMapper

AUDIO_FEATURES = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
    "popularity",
]


@dataclass
class DatasetBundle:
    data: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    metadata: dict[str, float]


def minmax(series: pd.Series) -> pd.Series:
    span = series.max() - series.min()
    if span == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / span


def load_deam_reference(deam_root: str | Path) -> dict[str, float]:
    deam_root = Path(deam_root)
    frames = []
    for file_name in [
        "static_annotations_averaged_songs_1_2000.csv",
        "static_annotations_averaged_songs_2000_2058.csv",
    ]:
        path = (
            deam_root
            / "annotations"
            / "annotations averaged per song"
            / "song_level"
            / file_name
        )
        frame = pd.read_csv(path)
        frame.columns = [c.strip().replace(" ", "_") for c in frame.columns]
        frames.append(frame)
    deam = pd.concat(frames, ignore_index=True)
    return {
        "valence_mean": float(deam["valence_mean"].mean()),
        "valence_std": float(deam["valence_mean"].std(ddof=0)),
        "arousal_mean": float(deam["arousal_mean"].mean()),
        "arousal_std": float(deam["arousal_mean"].std(ddof=0)),
        "engagement_anchor": float(
            (
                0.55 * minmax(deam["arousal_mean"])
                + 0.45 * minmax(deam["valence_mean"])
            ).quantile(0.75)
        ),
    }


def compute_subconscious_score(
    spotify_df: pd.DataFrame,
    color_mapper: ColorEmotionMapper,
    deam_reference: dict[str, float],
) -> pd.DataFrame:
    df = spotify_df.copy()
    df["tempo_norm"] = minmax(df["tempo"])
    df["loudness_norm"] = np.clip((df["loudness"] + 60.0) / 60.0, 0.0, 1.0)
    df["popularity_norm"] = df["popularity"] / 100.0

    audio_valence_proxy = (
        0.55 * df["valence"]
        + 0.25 * df["danceability"]
        + 0.20 * (1.0 - df["speechiness"].clip(0, 1))
    )
    audio_arousal_proxy = (
        0.40 * df["energy"]
        + 0.20 * df["tempo_norm"]
        + 0.20 * df["loudness_norm"]
        + 0.20 * df["liveness"].clip(0, 1)
    )

    df["audio_valence_deam_scale"] = (
        deam_reference["valence_mean"]
        + (audio_valence_proxy - audio_valence_proxy.mean())
        / (audio_valence_proxy.std(ddof=0) + 1e-9)
        * (deam_reference["valence_std"] + 1e-9)
    )
    df["audio_arousal_deam_scale"] = (
        deam_reference["arousal_mean"]
        + (audio_arousal_proxy - audio_arousal_proxy.mean())
        / (audio_arousal_proxy.std(ddof=0) + 1e-9)
        * (deam_reference["arousal_std"] + 1e-9)
    )

    synthetic_profiles = df.apply(color_mapper.build_synthetic_color_profile, axis=1, result_type="expand")
    synthetic_profiles = synthetic_profiles.rename(columns={tag: f"color_{tag}" for tag in synthetic_profiles.columns})
    df = pd.concat([df, synthetic_profiles], axis=1)

    color_resonance = (
        0.22 * df["color_energetic"]
        + 0.18 * df["color_vibrant"]
        + 0.18 * df["color_playful"]
        + 0.14 * df["color_warm"]
        + 0.14 * df["color_modern"]
        + 0.14 * df["color_luxurious"]
    )
    listening_intent = (
        0.35 * df["popularity_norm"]
        + 0.25 * df["danceability"]
        + 0.20 * df["energy"]
        + 0.20 * df["valence"]
    )
    emotion_alignment = 1.0 - np.abs(
        (0.55 * minmax(df["audio_arousal_deam_scale"]) + 0.45 * minmax(df["audio_valence_deam_scale"]))
        - deam_reference["engagement_anchor"]
    )

    df["subconscious_buying_influence_score"] = 100.0 * (
        0.45 * listening_intent + 0.35 * emotion_alignment + 0.20 * color_resonance
    )
    return df


def build_training_dataset(
    spotify_csv: str | Path,
    emotion_palette_csv: str | Path,
    deam_root: str | Path,
    sample_size: int = 12000,
    random_state: int = 42,
) -> DatasetBundle:
    spotify = pd.read_csv(spotify_csv)
    if sample_size and len(spotify) > sample_size:
        spotify = spotify.sample(sample_size, random_state=random_state).reset_index(drop=True)

    mapper = ColorEmotionMapper(emotion_palette_csv)
    deam_reference = load_deam_reference(deam_root)
    prepared = compute_subconscious_score(spotify, mapper, deam_reference)

    feature_columns = AUDIO_FEATURES + [f"color_{tag}" for tag in COLOR_TAGS]
    return DatasetBundle(
        data=prepared,
        feature_columns=feature_columns,
        target_column="subconscious_buying_influence_score",
        metadata=deam_reference,
    )
