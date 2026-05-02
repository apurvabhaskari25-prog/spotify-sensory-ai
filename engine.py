from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from color_analysis import COLOR_TAGS, ColorEmotionMapper
from data_pipeline import AUDIO_FEATURES
from model import RidgeRegressor




@dataclass
class PredictionResult:
    score: float
    feature_row: pd.DataFrame
    color_scores: dict[str, float]
    dominant_colors: list[str]
    feature_contributions: pd.DataFrame
    strategic_message: str
    thumbnail_recommendations: list[dict[str, object]]


def strategic_message(score: float) -> str:
    if score >= 85:
        return "This sensory combination is highly persuasive for fast attention capture and playlist engagement."
    if score >= 70:
        return "This mix is strong for personalized discovery and commercial playlist placement."
    if score >= 55:
        return "This concept is promising, but the thumbnail and emotional contrast can be improved further."
    return "This concept needs stronger sensory alignment between sound mood and visual identity."


def build_feature_frame(
    mapper: ColorEmotionMapper,
    audio_inputs: dict[str, float],
    image_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    dominant_colors: list[str] = []
    if image_path:
        palette_match = mapper.analyze_image(image_path)
        color_scores = palette_match.color_tag_scores
        dominant_colors = palette_match.dominant_hex
    else:
        color_scores = mapper.build_synthetic_color_profile(pd.Series(audio_inputs))

    row = {feature: float(audio_inputs[feature]) for feature in AUDIO_FEATURES}
    for tag in COLOR_TAGS:
        row[f"color_{tag}"] = float(color_scores.get(tag, 0.0))
    return pd.DataFrame([row]), color_scores, dominant_colors


def compute_feature_contributions(
    model: RidgeRegressor,
    feature_row: pd.DataFrame,
) -> pd.DataFrame:
    x = feature_row[model.feature_names].to_numpy(dtype=float)[0]
    scaled = (x - model.means) / model.stds
    contributions = scaled * model.weights
    frame = pd.DataFrame(
        {
            "feature": model.feature_names,
            "value": x,
            "contribution": contributions,
            "abs_contribution": np.abs(contributions),
        }
    ).sort_values("abs_contribution", ascending=False)
    return frame


def recommend_thumbnail_direction(
    model: RidgeRegressor,
    mapper: ColorEmotionMapper,
    feature_row: pd.DataFrame,
    top_k: int = 3,
) -> list[dict[str, object]]:
    contribution_frame = compute_feature_contributions(model, feature_row)
    desired_tags: dict[str, float] = {}
    for _, row in contribution_frame.iterrows():
        feature = str(row["feature"])
        if not feature.startswith("color_"):
            continue
        tag = feature.replace("color_", "", 1)
        if float(row["contribution"]) > 0:
            desired_tags[tag] = abs(float(row["contribution"]))
    if not desired_tags:
        desired_tags = {"warm": 1.0, "vibrant": 0.9, "modern": 0.8}
    return mapper.recommend_palette_rows(desired_tags, top_k=top_k)


def predict_with_explanations(
    model: RidgeRegressor,
    mapper: ColorEmotionMapper,
    audio_inputs: dict[str, float],
    image_path: str | Path | None = None,
) -> PredictionResult:
    feature_row, color_scores, dominant_colors = build_feature_frame(mapper, audio_inputs, image_path=image_path)
    score = float(np.clip(model.predict(feature_row[model.feature_names].to_numpy(dtype=float))[0], 0.0, 100.0))
    contributions = compute_feature_contributions(model, feature_row)
    thumbnail_recommendations = recommend_thumbnail_direction(model, mapper, feature_row, top_k=3)
    return PredictionResult(
        score=score,
        feature_row=feature_row,
        color_scores=color_scores,
        dominant_colors=dominant_colors,
        feature_contributions=contributions,
        strategic_message=strategic_message(score),
        thumbnail_recommendations=thumbnail_recommendations,
    )


def recommend_tracks(
    library_df: pd.DataFrame,
    model: RidgeRegressor,
    feature_row: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    required_cols = [col for col in model.feature_names if col in library_df.columns]
    library = library_df.dropna(subset=required_cols).copy()
    if library.empty:
        return library

    x_library = library[model.feature_names].to_numpy(dtype=float)
    x_query = feature_row[model.feature_names].to_numpy(dtype=float)
    distances = np.linalg.norm(x_library - x_query, axis=1)
    library["match_distance"] = distances

    columns = [
        "artist_name",
        "track_name",
        "track_id",
        "subconscious_buying_influence_score",
        "match_distance",
        "energy",
        "valence",
        "danceability",
        "tempo",
        "popularity",
    ]
    available_cols = [col for col in columns if col in library.columns]
    return library.sort_values(
        ["match_distance", "subconscious_buying_influence_score"],
        ascending=[True, False],
    )[available_cols].head(top_k)


def score_batch(
    batch_df: pd.DataFrame,
    model: RidgeRegressor,
    mapper: ColorEmotionMapper,
) -> pd.DataFrame:
    scored_rows = []
    for _, row in batch_df.iterrows():
        audio_inputs = {feature: float(row.get(feature, 0.0)) for feature in AUDIO_FEATURES}
        feature_row, color_scores, _ = build_feature_frame(mapper, audio_inputs, image_path=None)
        score = float(np.clip(model.predict(feature_row[model.feature_names].to_numpy(dtype=float))[0], 0.0, 100.0))
        scored_row = row.to_dict()
        scored_row["predicted_subconscious_buying_influence_score"] = score
        top_tag = max(color_scores.items(), key=lambda item: item[1])[0]
        scored_row["dominant_inferred_visual_emotion"] = top_tag
        scored_rows.append(scored_row)
    return pd.DataFrame(scored_rows)