from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image


COLOR_TAGS = [
    "bold",
    "calm",
    "cool",
    "dramatic",
    "dreamy",
    "elegant",
    "energetic",
    "joy",
    "luxurious",
    "modern",
    "peaceful",
    "playful",
    "powerful",
    "soft",
    "sophisticated",
    "vibrant",
    "warm",
    "youthful",
]


def hex_to_rgb(hex_code: str) -> np.ndarray:
    hex_code = str(hex_code).strip().lstrip("#")
    return np.array(
        [int(hex_code[0:2], 16), int(hex_code[2:4], 16), int(hex_code[4:6], 16)],
        dtype=float,
    )


def rgb_to_hex(rgb: Iterable[float]) -> str:
    clipped = np.clip(np.round(np.asarray(rgb)), 0, 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*clipped.tolist())


@dataclass
class PaletteMatch:
    dominant_hex: list[str]
    color_tag_scores: dict[str, float]
    mean_rgb: np.ndarray


class ColorEmotionMapper:
    def __init__(self, palette_csv: str | Path):
        palette = pd.read_csv(palette_csv)
        color_cols = [c for c in palette.columns if c.lower().startswith("color ")]
        self.palette = palette.copy()
        self.color_cols = color_cols
        self.tag_cols = [c for c in COLOR_TAGS if c in palette.columns]

        row_rgbs = []
        for _, row in self.palette.iterrows():
            rgbs = np.vstack([hex_to_rgb(row[col]) for col in color_cols])
            row_rgbs.append(rgbs.mean(axis=0))
        self.row_rgb = np.vstack(row_rgbs)
        self.row_tags = self.palette[self.tag_cols].astype(float).to_numpy()

    def _match_rgb(self, rgb: np.ndarray, top_k: int = 3) -> np.ndarray:
        distances = np.linalg.norm(self.row_rgb - rgb, axis=1)
        nearest_idx = np.argsort(distances)[:top_k]
        nearest_distances = distances[nearest_idx]
        weights = 1.0 / (nearest_distances + 1e-6)
        weights = weights / weights.sum()
        return (self.row_tags[nearest_idx] * weights[:, None]).sum(axis=0)

    def analyze_image(self, image_path: str | Path, n_colors: int = 5) -> PaletteMatch:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((250, 250))
        quantized = image.quantize(colors=n_colors, method=Image.MEDIANCUT).convert("RGB")
        colors = quantized.getcolors(maxcolors=n_colors * 20) or []
        colors = sorted(colors, key=lambda item: item[0], reverse=True)[:n_colors]

        if not colors:
            raise ValueError("No colors could be extracted from the image.")

        total = float(sum(count for count, _ in colors))
        weighted_tags = np.zeros(len(self.tag_cols), dtype=float)
        weighted_rgb = np.zeros(3, dtype=float)
        dominant_hex = []

        for count, color in colors:
            rgb = np.array(color, dtype=float)
            dominant_hex.append(rgb_to_hex(rgb))
            weight = count / total
            weighted_rgb += weight * rgb
            weighted_tags += weight * self._match_rgb(rgb)

        color_tag_scores = {
            tag: float(score) for tag, score in zip(self.tag_cols, weighted_tags, strict=True)
        }
        return PaletteMatch(
            dominant_hex=dominant_hex,
            color_tag_scores=color_tag_scores,
            mean_rgb=weighted_rgb,
        )

    def build_synthetic_color_profile(self, audio_row: pd.Series) -> dict[str, float]:
        danceability = float(audio_row.get("danceability", 0.0))
        energy = float(audio_row.get("energy", 0.0))
        valence = float(audio_row.get("valence", 0.0))
        acousticness = float(audio_row.get("acousticness", 0.0))
        speechiness = float(audio_row.get("speechiness", 0.0))
        popularity = float(audio_row.get("popularity", 0.0)) / 100.0
        loudness = float(audio_row.get("loudness", -60.0))
        loudness_norm = np.clip((loudness + 60.0) / 60.0, 0.0, 1.0)

        return {
            "bold": 0.55 * energy + 0.45 * loudness_norm,
            "calm": 0.55 * acousticness + 0.45 * (1.0 - energy),
            "cool": 0.45 * acousticness + 0.35 * (1.0 - valence) + 0.20 * energy,
            "dramatic": 0.60 * energy + 0.40 * (1.0 - valence),
            "dreamy": 0.50 * acousticness + 0.30 * valence + 0.20 * (1.0 - speechiness),
            "elegant": 0.45 * valence + 0.30 * acousticness + 0.25 * popularity,
            "energetic": 0.65 * energy + 0.20 * danceability + 0.15 * valence,
            "joy": 0.65 * valence + 0.35 * danceability,
            "luxurious": 0.50 * popularity + 0.25 * valence + 0.25 * (1.0 - speechiness),
            "modern": 0.50 * energy + 0.30 * danceability + 0.20 * popularity,
            "peaceful": 0.60 * acousticness + 0.40 * (1.0 - energy),
            "playful": 0.55 * danceability + 0.45 * valence,
            "powerful": 0.65 * energy + 0.35 * loudness_norm,
            "soft": 0.60 * acousticness + 0.40 * (1.0 - loudness_norm),
            "sophisticated": 0.45 * popularity + 0.30 * acousticness + 0.25 * (1.0 - speechiness),
            "vibrant": 0.45 * energy + 0.35 * valence + 0.20 * danceability,
            "warm": 0.70 * valence + 0.30 * energy,
            "youthful": 0.55 * danceability + 0.25 * energy + 0.20 * popularity,
        }

    def recommend_palette_rows(
        self,
        target_tag_scores: dict[str, float],
        top_k: int = 3,
    ) -> list[dict[str, object]]:
        target = np.array([float(target_tag_scores.get(tag, 0.0)) for tag in self.tag_cols], dtype=float)
        if np.allclose(target, 0):
            return []

        tag_matrix = self.row_tags
        scores = (tag_matrix @ target) / (
            np.linalg.norm(tag_matrix, axis=1) * np.linalg.norm(target) + 1e-9
        )
        best_idx = np.argsort(scores)[::-1][:top_k]

        recommendations = []
        for idx in best_idx:
            row = self.palette.iloc[idx]
            colors = [str(row[col]) for col in self.color_cols]
            tag_values = {tag: float(row[tag]) for tag in self.tag_cols}
            recommendations.append(
                {
                    "colors": colors,
                    "score": float(scores[idx]),
                    "tags": tag_values,
                }
            )
        return recommendations
