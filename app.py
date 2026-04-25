from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from model import RidgeRegressor
from color_analysis import ColorEmotionMapper
from engine import predict_with_explanations, recommend_tracks, score_batch
from data_pipeline import AUDIO_FEATURES


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "sensory_model.json"
EMOTION_CSV = ROOT / "emotion_palette.csv"
LIBRARY_PATH = ROOT / "training_dataset.csv"
TOP_TRACKS_PATH = ROOT / "top_spotify_recommendations.csv"


def playlist_message(score: float) -> str:
    if score >= 80:
        return "This combination signals high emotional arousal and stronger subconscious engagement, making it suitable for upbeat Spotify moments like Discover Weekly, mood boosters, and high-save playlists."
    if score >= 65:
        return "This mix suggests good subconscious engagement, stronger listening continuity, and a higher chance of playlist saves in personalized Spotify journeys."
    if score >= 50:
        return "This combination has moderate influence, but stronger color-emotion alignment or a more energetic mood could improve longer listening and save intent."
    return "This combination is likely to create weaker emotional pull, so warmer colors, stronger tempo-energy cues, or more positive valence may improve engagement."


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


def sensory_congruence_label(audio_inputs: dict[str, float], color_scores: dict[str, float]) -> tuple[str, str]:
    audio_energy = (
        0.40 * float(audio_inputs["energy"])
        + 0.30 * float(audio_inputs["valence"])
        + 0.20 * min(float(audio_inputs["tempo"]) / 180.0, 1.0)
        + 0.10 * float(audio_inputs["danceability"])
    )
    color_energy = (
        0.35 * float(color_scores.get("warm", 0.0))
        + 0.25 * float(color_scores.get("energetic", 0.0))
        + 0.20 * float(color_scores.get("vibrant", 0.0))
        + 0.20 * float(color_scores.get("bold", 0.0))
    )
    gap = abs(audio_energy - color_energy)
    if gap <= 0.18:
        return (
            "Strong Sensory Congruence",
            "The thumbnail mood and audio mood reinforce each other, which can improve subconscious engagement, reduce skips, and support longer Spotify listening sessions.",
        )
    if gap <= 0.35:
        return (
            "Moderate Sensory Congruence",
            "The audio and color cues are partially aligned. This can still attract users, but stronger mood matching may improve playlist saves and sustained attention.",
        )
    return (
        "Weak Sensory Congruence",
        "The sound and visual signals are pulling in different emotional directions, which may weaken first-click impact and lower subconscious engagement.",
    )


def combined_statement(audio_inputs: dict[str, float], color_scores: dict[str, float], dominant_color: str) -> str:
    tempo = float(audio_inputs["tempo"])
    valence = float(audio_inputs["valence"])
    warm = float(color_scores.get("warm", 0.0))
    energetic = float(color_scores.get("energetic", 0.0))
    if tempo >= 120 and valence >= 0.6 and warm >= 0.55:
        return f"The dominant color {dominant_color} supports an upbeat, positive sound profile. Together, the warm visual cue and energetic audio signal are likely to increase emotional arousal and impulse-like listening on Spotify."
    if tempo < 105 and valence < 0.45 and warm < 0.4:
        return f"The dominant color {dominant_color} leans away from warmth while the audio profile is calmer or moodier. This may suit reflective playlists, but it is less likely to trigger fast subconscious engagement."
    if energetic >= 0.55:
        return f"The dominant color {dominant_color} adds energetic visual reinforcement to the track. This supports stronger mood coherence and can improve attention in competitive playlist environments."
    return f"The dominant color {dominant_color} creates a moderate visual-emotional cue. Its impact depends on how strongly the track's tempo, valence, and energy reinforce that same mood."


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
            "Contribution Type": ["Audio Features", "Color Features"],
            "Relative Influence": [audio_total / total, color_total / total],
        }
    ).set_index("Contribution Type")


def recommendation_message(score: float, audio_inputs: dict[str, float], warmth: float) -> str:
    tempo = float(audio_inputs["tempo"])
    valence = float(audio_inputs["valence"])
    if score >= 80 and tempo >= 118 and valence >= 0.6:
        return "Best suited for energetic morning playlists, Discover Weekly-style discovery moments, and festive high-save listening behavior."
    if score >= 70 and warmth >= 55:
        return "Well suited for Release Radar promotion, mood playlists, and attention-grabbing cover art strategies during culturally vibrant or festive listening seasons in India."
    if score >= 60:
        return "Useful for personalized mood playlists where emotional fit matters more than instant excitement."
    return "Better suited for niche or reflective playlists unless the thumbnail becomes warmer and the audio mood becomes more uplifting."


def example_box() -> None:
    st.info(
        "Example: a warm red-orange thumbnail combined with an upbeat, high-valence track can produce a high influence score because both the visual and sound cues increase emotional arousal and subconscious engagement."
    )


def main() -> None:
    st.set_page_config(page_title="AI-Predicted Sensory Marketing for Spotify", layout="wide")
    st.title("AI-Predicted Sensory Marketing for Spotify: How Color and Sound Influence Subconscious Buying Behavior")
    st.caption(
        "This app demonstrates how AI can predict the subconscious impact of color (album art) and sound (audio features) on Spotify user engagement and buying behavior (subscription / playlist saves)."
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
        image_path = ROOT / "artifacts" / f"uploaded_{uploaded_file.name}"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(uploaded_file.getbuffer())

    result = predict_with_explanations(model, mapper, audio_inputs, image_path=image_path)
    score = result.score
    dominant_color = result.dominant_colors[0] if result.dominant_colors else "synthetic mood color"
    warm_score = warmth_score(result.color_scores)
    congruence_label, congruence_text = sensory_congruence_label(audio_inputs, result.color_scores)
    contribution_df = contribution_summary(result.feature_contributions)

    predictor_tab, strategy_tab, insights_tab, batch_tab = st.tabs(
        ["Single Track Predictor", "Thumbnail Strategy", "Spotify Case Study / Insights", "Batch Scoring"]
    )

    with predictor_tab:
        left, right = st.columns([1.15, 1.0])
        with left:
            st.subheader("Predicted Influence Score")
            st.metric("Subconscious Buying Influence Score", f"{score:.2f} / 100")
            st.write(result.strategic_message)
            st.caption(playlist_message(score))
            st.success(f"{congruence_label}: {congruence_text}")

            st.subheader("How Color + Sound Influence the Score")
            influence_df = pd.DataFrame(
                [
                    ("Dominant Color", dominant_color),
                    ("Warmth Score", f"{warm_score:.1f} / 100"),
                    ("Tempo", f"{audio_inputs['tempo']:.0f} BPM"),
                    ("Valence", f"{audio_inputs['valence']:.2f}"),
                ],
                columns=["Signal", "Value"],
            )
            st.dataframe(influence_df, hide_index=True, use_container_width=True)
            st.write(combined_statement(audio_inputs, result.color_scores, dominant_color))

            st.subheader("Key Insight")
            st.markdown(
                "\n".join(
                    [
                        f"- Warm color strength is estimated at **{warm_score:.1f}/100**, which matters because warmer tones are more likely to trigger excitement, urgency, and impulse-like attention.",
                        f"- The current sound profile uses **tempo {audio_inputs['tempo']:.0f} BPM** and **valence {audio_inputs['valence']:.2f}**, which shape emotional arousal, positivity, and readiness for longer listening.",
                        "- When color warmth and audio mood move in the same direction, subconscious influence becomes stronger because users receive one coherent emotional signal instead of mixed cues.",
                        "- In Spotify terms, stronger sensory alignment can increase the likelihood of longer listening, lower skip risk, and higher playlist-save behavior.",
                    ]
                )
            )

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

        st.subheader("Why the model gave this score")
        st.dataframe(
            result.feature_contributions[["feature", "value", "contribution"]].head(10),
            hide_index=True,
            use_container_width=True,
        )
        st.subheader("Color Features vs Audio Features")
        st.bar_chart(contribution_df)
        st.subheader("Feature Vector Used by the Model")
        st.dataframe(result.feature_row[model.feature_names], use_container_width=True)

    with strategy_tab:
        st.subheader("Recommended Thumbnail Directions")
        st.write(
            "These palettes are suggested from the emotion tags that most positively influence the current prediction. Warm, vibrant, and bold thumbnail directions can increase impulse-like listening decisions by reinforcing subconscious excitement before the user even presses play."
        )
        for idx, recommendation in enumerate(result.thumbnail_recommendations, start=1):
            st.markdown(f"**Palette {idx}**")
            render_palette(recommendation["colors"])
            st.caption(
                "Why it matters: stronger warmth and vibrancy can improve first-click attraction, emotional arousal, and playlist-save intent in Spotify browsing contexts."
            )
            top_tags = sorted(
                recommendation["tags"].items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
            st.dataframe(pd.DataFrame(top_tags, columns=["Emotion Tag", "Strength"]), hide_index=True)

    with insights_tab:
        st.subheader("Nearest Spotify-style Recommendations")
        if not library_df.empty:
            recommended_tracks = recommend_tracks(library_df, model, result.feature_row, top_k=10)
            st.info(recommendation_message(score, audio_inputs, warm_score))
            st.dataframe(recommended_tracks, hide_index=True, use_container_width=True)
            download_button_from_df(
                recommended_tracks,
                label="Download recommendation CSV",
                filename="spotify_sensory_recommendations.csv",
            )
        else:
            st.info("Training dataset artifact not found yet. Run the training step first.")

        if not top_tracks_df.empty:
            st.subheader("Highest-scoring tracks in the current case study dataset")
            st.dataframe(top_tracks_df.head(15), hide_index=True, use_container_width=True)

        st.subheader("Spotify Case Study Insight")
        st.write(
            "For Spotify, this prototype shows how thumbnail color and track mood can work together to influence subconscious engagement. Strong sensory congruence can support Discover Weekly clicks, mood-playlist retention, and playlist saves by making the emotional promise of the visual match the emotional payoff of the audio."
        )

    with batch_tab:
        st.subheader("Score a whole CSV")
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
