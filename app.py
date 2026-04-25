def main() -> None:
    st.set_page_config(page_title="AI Predicted Sensory Marketing", layout="wide")
    st.title("AI-Predicted Sensory Marketing")
    st.caption("Spotify case study: color + sound -> Subconscious Buying Influence Score")

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

    overview_tab, strategy_tab, playlist_tab, batch_tab = st.tabs(
        ["Live Predictor", "Thumbnail Strategy", "Spotify Recommendations", "Batch Scoring"]
    )

    with overview_tab:
        left, right = st.columns([1.15, 1.0])
        with left:
            st.subheader("Predicted Influence Score")
            st.metric("Subconscious Buying Influence Score", f"{score:.2f} / 100")
            st.write(result.strategic_message)
            st.caption(playlist_message(score))

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

        st.subheader("Why the model gave this score")
        st.dataframe(
            result.feature_contributions[["feature", "value", "contribution"]].head(10),
            hide_index=True,
            use_container_width=True,
        )
        st.subheader("Feature Vector Used by the Model")
        st.dataframe(result.feature_row[model.feature_names], use_container_width=True)

    with strategy_tab:
        st.subheader("Recommended Thumbnail Directions")
        st.write(
            "These palettes are suggested from the emotion tags that most positively influence the current prediction."
        )
        for idx, recommendation in enumerate(result.thumbnail_recommendations, start=1):
            st.markdown(f"**Palette {idx}**")
            render_palette(recommendation["colors"])
            top_tags = sorted(
                recommendation["tags"].items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
            st.dataframe(pd.DataFrame(top_tags, columns=["Emotion Tag", "Strength"]), hide_index=True)

    with playlist_tab:
        st.subheader("Nearest Spotify-style Recommendations")
        if not library_df.empty:
            recommended_tracks = recommend_tracks(library_df, model, result.feature_row, top_k=10)
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

    with batch_tab:
        st.subheader("Score a whole CSV")
        st.write(
            "Upload a CSV with the audio columns used by the model to score many tracks at once."
        )
        st.code(", ".join(AUDIO_FEATURES))
        batch_file = st.file_uploader(
            "Upload batch audio CSV",
            type=["csv"],
            key="batch_csv",
        )
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