import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import streamlit as st
from collections import Counter, defaultdict

from components import (
    render_prediction,
    render_batch_chart,
    no_models_warning,
    ASPECT_EMOJI,
)


st.set_page_config(
    page_title="Aspectify — Review Intelligence",
    page_icon="\u26a1",
    layout="wide",
)



@st.cache_resource(show_spinner="Loading models\u2026")
def load_pipeline():
    from predict import AspectSentimentPipeline
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    return AspectSentimentPipeline.load(model_dir)


with st.sidebar:
    st.title("\u26a1 Aspectify")
    st.caption("Aspect-Based Sentiment Analysis for Hinglish & English E-Commerce Reviews")
    st.divider()

    st.markdown("**Tracked Aspects**")
    for aspect, emoji in ASPECT_EMOJI.items():
        st.markdown(f"{emoji} {aspect.capitalize()}")

    st.divider()
    st.caption("Classical ML baseline  \u00b7  TF-IDF + LinearSVC  \u00b7  Hinglish-aware")



st.title("\u26a1 Aspectify \u2014 Review Intelligence")
st.markdown(
    "*Turn \"Battery bahut accha hai but delivery late thi\" into actionable, "
    "aspect-level signals.*"
)
st.divider()


pipeline = None
models_loaded = False
try:
    pipeline = load_pipeline()
    models_loaded = True
except FileNotFoundError:
    no_models_warning()


tab1, tab2 = st.tabs(["\U0001f50d Single Review", "\U0001f4ca Batch Analysis (CSV)"])



with tab1:
    st.subheader("Analyse a Single Review")

    sample_reviews = [
        "Battery bahut accha hai but delivery 10 din late aayi aur packaging bekar thi.",
        "Camera ekdum mast hai! Charging thodi slow hai though.",
        "Paisa vasool product! Quality solid hai, service bhi accha tha.",
        "Worst product ever. Quality kharab hai, return bhi nahi le rahe.",
        "Delivery time pe aayi, packaging bhi theek thi, overall satisfied.",
    ]

    example = st.selectbox(
        "Try a sample review:",
        options=["(type your own below)"] + sample_reviews,
        index=0,
    )

    default_text = example if example != "(type your own below)" else ""
    review_text = st.text_area(
        "Review text",
        value=default_text,
        height=100,
        placeholder="Enter an English or Hinglish product review\u2026",
    )

    predict_btn = st.button(
        "Analyse Review", type="primary", disabled=not models_loaded
    )

    if predict_btn and review_text.strip():
        with st.spinner("Analysing\u2026"):
            result = pipeline.predict(review_text.strip())

        st.markdown("#### Detected Aspects & Sentiment")
        render_prediction(result)

        if result:
            st.divider()
            st.markdown("**Raw output:**")
            st.json(result)

    elif predict_btn:
        st.warning("Please enter a review text first.")



with tab2:
    st.subheader("Batch Analysis \u2014 Upload a CSV")
    st.markdown(
        "Upload a CSV file with a **`text`** column. "
        "Aspectify will run the full pipeline on each row and produce "
        "an aspect-wise sentiment summary chart."
    )

    uploaded = st.file_uploader("Upload reviews CSV", type=["csv"])

    if uploaded and models_loaded:
        df = pd.read_csv(uploaded)

        if "text" not in df.columns:
            st.error("CSV must contain a `text` column.")
        else:
            max_rows = st.slider(
                "Max rows to analyse", min_value=10, max_value=500,
                value=100, step=10,
            )
            sample_df = df.head(max_rows).copy()

            with st.spinner(f"Analysing {len(sample_df)} reviews\u2026"):
                sample_df["predictions"] = pipeline.predict_batch(
                    sample_df["text"].astype(str).tolist()
                )

            st.success(f"Done! Analysed **{len(sample_df)}** reviews.")
            st.divider()

            # Build aspect x sentiment count table
            agg: defaultdict = defaultdict(Counter)
            for pred in sample_df["predictions"]:
                for aspect, sentiment in pred.items():
                    agg[aspect][sentiment] += 1

            if agg:
                records = [
                    {"aspect": aspect, "sentiment": sentiment, "count": count}
                    for aspect, sentiments in agg.items()
                    for sentiment, count in sentiments.items()
                ]
                agg_df = pd.DataFrame(records)

                st.markdown("#### Aspect-Wise Sentiment Summary")
                render_batch_chart(agg_df)

                st.markdown("#### Aggregated Counts")
                pivot = agg_df.pivot_table(
                    index="aspect", columns="sentiment",
                    values="count", aggfunc="sum", fill_value=0,
                )
                st.dataframe(pivot, use_container_width=True)

            st.markdown("#### Full Predictions")
            st.dataframe(
                sample_df[["text", "predictions"]], use_container_width=True
            )

            csv_out = (
                sample_df[["text", "predictions"]]
                .to_csv(index=False)
                .encode("utf-8")
            )
            st.download_button(
                "\u2b07\ufe0f Download Predictions CSV",
                csv_out,
                file_name="aspectify_predictions.csv",
                mime="text/csv",
            )

    elif uploaded and not models_loaded:
        no_models_warning()
