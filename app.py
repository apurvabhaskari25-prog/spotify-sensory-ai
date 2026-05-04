from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from color_analysis import ColorEmotionMapper
from data_pipeline import AUDIO_FEATURES
from engine import predict_with_explanations, recommend_tracks, score_batch
from model import RidgeRegressor

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "sensory_model.json"
EMOTION_CSV = ROOT / "emotion_palette.csv"
LIBRARY_PATH = ROOT / "training_dataset.csv"
TOP_TRACKS_PATH = ROOT / "top_spotify_recommendations.csv"



def playlist_message(score: float) -> str:
    if score >= 80:
        return "This combination suggests strong subconscious buying behavior signals, high emotional arousal, and stronger listening engagement on Spotify. It is well suited to Discover Weekly, Release Radar, and high-save mood playlists."
    if score >= 65:
        return "This combination shows good subconscious engagement potential and can support longer listening sessions, mood playlist retention, and playlist saves."
    if score >= 50:
        return "This combination has moderate influence. Better color-sound alignment could improve emotional arousal and subconscious listening engagement."
    return "This combination is likely to have weaker subconscious pull. Stronger warmth, positivity, or tempo-energy alignment may improve engagement."


def render_palette(colors: list[str]) -> None:
    if not colors:
        return
    html = "".join(
        f"<div style='width:68px;height:68px;border-radius:14px;background:{color};display:inline-block;margin-right:10px;border:1px solid rgba(0,0,0,0.08);'></div>"
        for color in colors
    )
    st.markdown(html, unsafe_allow_html=True)


def download_button_from_df(df: pd.DataFrame, label: str, filename: str) -> None:
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def warmth_score(color_scores: dict[str, float]) -> float:
    return 100.0 * (
        0.45 * float(color_scores.get("warm", 0.0))
        + 0.25 * float(color_scores.get("vibrant", 0.0))
        + 0.15 * float(color_scores.get("joy", 0.0))
        + 0.15 * float(color_scores.get("bold", 0.0))
    )


def saturation_score(color_scores: dict[str, float]) -> float:
    return 100.0 * (
        0.40 * float(color_scores.get("vibrant", 0.0))
        + 0.25 * float(color_scores.get("bold", 0.0))
        + 0.20 * float(color_scores.get("energetic", 0.0))
        + 0.15 * float(color_scores.get("playful", 0.0))
    )


def dominant_color_name(color_scores: dict[str, float]) -> str:
    ranked = sorted(color_scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return "Balanced tone"
    top = ranked[0][0]
    mapping = {
        "warm": "Warm tone",
        "vibrant": "Vibrant tone",
        "bold": "Bold tone",
        "cool": "Cool tone",
        "soft": "Soft tone",
        "energetic": "Energetic tone",
        "peaceful": "Peaceful tone",
        "playful": "Playful tone",
        "luxurious": "Luxurious tone",
    }
    return mapping.get(top, top.replace("_", " ").title())


def sensory_congruence_label(
    audio_inputs: dict[str, float],
    color_scores: dict[str, float],
) -> tuple[str, str]:
    warm = float(color_scores.get("warm", 0.0))
    vibrant = float(color_scores.get("vibrant", 0.0))
    energetic = float(color_scores.get("energetic", 0.0))

    valence = float(audio_inputs["valence"])
    energy = float(audio_inputs["energy"])
    tempo = float(audio_inputs["tempo"])

    audio_liveliness = 0.45 * valence + 0.35 * energy + 0.20 * min(tempo / 180.0, 1.0)
    visual_liveliness = 0.45 * warm + 0.30 * vibrant + 0.25 * energetic
    gap = abs(audio_liveliness - visual_liveliness)

    if gap <= 0.18:
        return (
            "Strong",
            "The album-art color and sound profile support the same emotional message. This usually improves subconscious engagement, click appeal, and listening continuity.",
        )
    if gap <= 0.35:
        return (
            "Moderate",
            "The sound and color cues are partially aligned. This can still work well, but stronger mood matching may improve playlist saves and first-click attraction.",
        )
    return (
        "Weak",
        "The visual and audio signals are pulling in different emotional directions. That can weaken subconscious influence and reduce listening engagement.",
    )


def combined_insight(
    audio_inputs: dict[str, float],
    color_scores: dict[str, float],
    dominant_color: str,
) -> str:
    tempo = float(audio_inputs["tempo"])
    valence = float(audio_inputs["valence"])
    energy = float(audio_inputs["energy"])
    warmth = warmth_score(color_scores)

    tempo_band = "high" if tempo >= 125 else "medium" if tempo >= 105 else "low"
    valence_band = "high" if valence >= 0.60 else "medium" if valence >= 0.40 else "low"
    warm_band = "warm" if warmth >= 55 else "cooler"

    return (
        f"The {dominant_color.lower()} visual profile combined with {valence_band} valence, "
        f"{tempo_band} tempo, and energy level {energy:.2f} creates a sensory pattern that affects emotional arousal "
        f"and subconscious listening behavior. When Spotify users see a {warm_band} visual cue alongside a track with this mood profile, "
        f"they are more likely to feel curiosity, continue listening, and save playlists when the emotional cues are aligned."
    )


def contribution_summary(contribution_df: pd.DataFrame) -> pd.DataFrame:
    audio_total = float(
        contribution_df.loc[~contribution_df["feature"].str.startswith("color_"), "abs_contribution"].sum()
    )
    color_total = float(
        contribution_df.loc[contribution_df["feature"].str.startswith("color_"), "abs_contribution"].sum()
    )
    total = audio_total + color_total + 1e-9
    return pd.DataFrame(
        {
            "Contribution Type": ["Sound Features", "Color Features"],
            "Relative Influence": [audio_total / total, color_total / total],
        }
    ).set_index("Contribution Type")


def recommendation_message(score: float, audio_inputs: dict[str, float], warmth: float) -> str:
    tempo = float(audio_inputs["tempo"])
    valence = float(audio_inputs["valence"])

    if score >= 80 and tempo >= 118 and valence >= 0.60:
        return "Best for Discover Weekly or Release Radar, energetic morning playlists, and festive-season listening where Spotify users respond to high arousal and positive mood."
    if score >= 70 and warmth >= 55:
        return "Well suited for mood playlists, upbeat recommendation surfaces, and festive periods in India where warm color + lively sound can improve playlist saves."
    if score >= 60:
        return "Useful for personalized mood playlists where emotional fit matters more than instant intensity."
    return "Better suited for niche or reflective playlist contexts unless the visual warmth or musical positivity is increased."


def why_this_score_bullets(
    audio_inputs: dict[str, float],
    color_scores: dict[str, float],
    warmth: float,
) -> list[str]:
    tempo = float(audio_inputs["tempo"])
    valence = float(audio_inputs["valence"])
    energy = float(audio_inputs["energy"])

    color_direction = (
        "Warm colors are helping raise excitement and urgency"
        if warmth >= 55
        else "Cooler or softer colors are reducing urgency and excitement"
    )
    tempo_direction = "Higher tempo is increasing arousal" if tempo >= 118 else "Tempo is moderate, so arousal is more controlled"
    valence_direction = (
        "High valence is making the mood more positive and approachable"
        if valence >= 0.60
        else "Moderate or low valence is reducing positivity"
    )
    energy_direction = (
        "Strong energy increases movement, momentum, and engagement"
        if energy >= 0.70
        else "Lower energy reduces immediate impact"
    )

    return [
        f"{color_direction}, which matters because users often respond subconsciously to visual warmth before deciding to click or listen.",
        f"{tempo_direction}, while {valence_direction}. Together these shape emotional arousal and listening mood.",
        f"{energy_direction}, making the soundtrack either more playlist-friendly or more subdued.",
        "The combination of color and sound matters because subconscious influence is stronger when the emotional signal is coherent across both thumbnail and audio mood.",
    ]


def example_box() -> None:
    st.info(
        "Example: a warm red-orange thumbnail paired with a high-valence, upbeat track often creates strong sensory congruence, which can improve emotional arousal, listening engagement, and playlist-save behavior on Spotify."
    )


def main() -> None:
    st.set_page_config(page_title="Spotify Sensory AI", layout="wide")
    st.title("Spotify Sensory AI")

    st.caption("Predicting How Color & Sound Influence Subconscious Buying Behavior")
    st.write(
        "This AI tool predicts how album art color and audio features influence user engagement and subconscious buying behavior on Spotify."
    )

    if not MODEL_PATH.exists():
        st.error("Model artifact not found. Run `train_model.py` first.")
        st.stop()

    model = RidgeRegressor.load(MODEL_PATH)
    mapper = ColorEmotionMapper(EMOTION_CSV)
    library_df = pd.read_csv(LIBRARY_PATH) if LIBRARY_PATH.exists() else pd.DataFrame()
    top_tracks_df = pd.read_csv(TOP_TRACKS_PATH) if TOP_TRACKS_PATH.exists() else pd.DataFrame()

    with st.sidebar:
        st.header("Audio Inputs")
        audio_inputs = {
            "acousticness": st.slider("Acousticness", 0.0, 1.0, 0.35, 0.01),
            "danceability": st.slider("Danceability", 0.0, 1.0, 0.70, 0.01),
            "energy": st.slider("Energy", 0.0, 1.0, 0.75, 0.01),
            "instrumentalness": st.slider("Instrumentalness", 0.0, 1.0, 0.05, 0.01),
            "liveness": st.slider("Liveness", 0.0, 1.0, 0.18, 0.01),
            "loudness": st.slider("Loudness", -30.0, 0.0, -6.0, 0.5),
            "speechiness": st.slider("Speechiness", 0.0, 1.0, 0.08, 0.01),
            "tempo": st.slider("Tempo", 50.0, 220.0, 120.0, 1.0),
            "valence": st.slider("Valence", 0.0, 1.0, 0.62, 0.01),
            "popularity": st.slider("Popularity", 0.0, 100.0, 60.0, 1.0),
        }
        uploaded_file = st.file_uploader("Upload album art / thumbnail", type=["png", "jpg", "jpeg"])
        st.caption("Tip: upload an album-art concept to compare visual mood with the song mood.")

    image_path = None
    if uploaded_file is not None:
        image_path = ROOT / f"uploaded_{uploaded_file.name}"

        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(uploaded_file.getbuffer())

    result = predict_with_explanations(model, mapper, audio_inputs, image_path=image_path)
    score = result.score
    dominant_color = dominant_color_name(result.color_scores)
    warm = warmth_score(result.color_scores)
    saturation = saturation_score(result.color_scores)
    congruence_label, congruence_text = sensory_congruence_label(audio_inputs, result.color_scores)
    contribution_df = contribution_summary(result.feature_contributions)

    predictor_tab, strategy_tab, insights_tab, batch_tab = st.tabs(
        ["Live Predictor", "Thumbnail Strategy", "Spotify Insights", "Batch Scoring"]
    )

    with predictor_tab:
        left, right = st.columns([1.2, 1.0])

        with left:
            st.subheader("Subconscious Buying Influence Score")
            st.metric("Subconscious Buying Influence Score", f"{score:.2f} / 100")
            st.write(playlist_message(score))
            st.success(f"Sensory Congruence: {congruence_label}")
            st.caption(congruence_text)

            st.subheader("Color Analysis")
            color_df = pd.DataFrame(
                [
                    ("Dominant Color", dominant_color),
                    ("Warmth Score", f"{warm:.1f} / 100"),
                    ("Saturation Score", f"{saturation:.1f} / 100"),
                ],
                columns=["Factor", "Value"],
            )
            st.dataframe(color_df, hide_index=True, use_container_width=True)

            st.subheader("Sound Analysis")
            sound_df = pd.DataFrame(
                [
                    ("Tempo", f"{audio_inputs['tempo']:.0f} BPM"),
                    ("Valence", f"{audio_inputs['valence']:.2f}"),
                    ("Energy", f"{audio_inputs['energy']:.2f}"),
                ],
                columns=["Factor", "Value"],
            )
            st.dataframe(sound_df, hide_index=True, use_container_width=True)

            st.subheader("Combined Insight")
            st.write(combined_insight(audio_inputs, result.color_scores, dominant_color))

            st.subheader("Why This Score?")
            for bullet in why_this_score_bullets(audio_inputs, result.color_scores, warm):
                st.markdown(f"- {bullet}")

            ranked_tags = sorted(result.color_scores.items(), key=lambda item: item[1], reverse=True)[:6]
            st.subheader("Top Sensory Color Signals")
            st.dataframe(pd.DataFrame(ranked_tags, columns=["Color Emotion Tag", "Score"]), hide_index=True)

        with right:
            st.subheader("Color Readout")
            if image_path is not None:
                st.image(str(image_path), caption="Uploaded visual input", use_container_width=True)
            if result.dominant_colors:
                render_palette(result.dominant_colors)
            else:
                st.info("No image uploaded. A synthetic visual profile was inferred from the audio mood.")
            example_box()

        st.subheader("Color vs Sound Contribution")
        st.bar_chart(contribution_df)
        st.caption(
            "This shows whether the final score was influenced more by album-art color features or sound features like tempo, valence, and energy."
        )

        st.subheader("Feature Contribution Snapshot")
        st.dataframe(
            result.feature_contributions[["feature", "value", "contribution"]].head(10),
            hide_index=True,
            use_container_width=True,
        )

    with strategy_tab:
        st.subheader("Thumbnail Strategy")
        st.write(
            "These palette suggestions are designed to improve subconscious impact. Warm, saturated, and energetic thumbnails can increase impulse-like listening decisions by strengthening the emotional signal before playback begins."
        )

        for idx, recommendation in enumerate(result.thumbnail_recommendations, start=1):
            st.markdown(f"**Palette {idx}**")
            render_palette(recommendation["colors"])
            st.caption(
                "Why it matters: warm and vibrant visuals can increase emotional arousal, click attraction, and playlist-save intent in Spotify browsing."
            )
            top_tags = sorted(
                recommendation["tags"].items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
            st.dataframe(pd.DataFrame(top_tags, columns=["Emotion Tag", "Strength"]), hide_index=True)

    with insights_tab:
        st.subheader("Spotify Recommendations")
        if not library_df.empty:
            recommended_tracks = recommend_tracks(library_df, model, result.feature_row, top_k=10)
            st.info(recommendation_message(score, audio_inputs, warm))
            st.dataframe(recommended_tracks, hide_index=True, use_container_width=True)
            download_button_from_df(
                recommended_tracks,
                label="Download recommendation CSV",
                filename="spotify_sensory_recommendations.csv",
            )
        else:
            st.info("Training dataset artifact not found yet. Run the training step first.")

        if not top_tracks_df.empty:
            st.subheader("Top Tracks in the Current Spotify Case Study Dataset")
            st.dataframe(top_tracks_df.head(15), hide_index=True, use_container_width=True)

        st.subheader("Spotify Case Study Insight")
        st.write(
            "This prototype helps explain how album-art color and sound features work together in Spotify contexts such as Discover Weekly, Release Radar, and mood playlists. Strong sensory congruence can improve subconscious engagement, increase listening continuity, and support more playlist saves."
        )

    with batch_tab:
        st.subheader("Batch Scoring")
        st.write("Upload a CSV with the audio columns used by the model to score many tracks at once.")
        st.code(", ".join(AUDIO_FEATURES))
        batch_file = st.file_uploader("Upload batch audio CSV", type=["csv"], key="batch_csv")

        if batch_file is not None:
            batch_df = pd.read_csv(batch_file)
            scored_batch = score_batch(batch_df, model, mapper)
            st.dataframe(scored_batch.head(25), use_container_width=True)
            download_button_from_df(
                scored_batch,
                label="Download scored batch CSV",
                filename="scored_sensory_marketing_batch.csv",
            )


if __name__ == "__main__":
    main()
