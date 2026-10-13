import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



ASPECT_EMOJI = {
    "battery":   "\U0001f50b",
    "delivery":  "\U0001f69a",
    "price":     "\U0001f4b0",
    "quality":   "\u2b50",
    "packaging": "\U0001f4e6",
    "service":   "\U0001f3a7",
}

SENTIMENT_COLOR = {
    "positive": "#27ae60",
    "negative": "#e74c3c",
    "neutral":  "#f39c12",
}

SENTIMENT_BG = {
    "positive": "#d5f5e3",
    "negative": "#fadbd8",
    "neutral":  "#fef9e7",
}


def aspect_badge(aspect: str, sentiment: str) -> None:
    emoji  = ASPECT_EMOJI.get(aspect, "\U0001f539")
    color  = SENTIMENT_COLOR.get(sentiment, "#7f8c8d")
    bg     = SENTIMENT_BG.get(sentiment, "#ecf0f1")
    label  = sentiment.upper()

    st.markdown(
        f"""
        <div style="
            display: inline-block;
            border: 2px solid {color};
            background-color: {bg};
            border-radius: 12px;
            padding: 8px 18px;
            margin: 6px;
            font-family: monospace;
        ">
            <span style="font-size: 1.3em;">{emoji}</span>
            <strong style="margin-left: 6px;">{aspect.upper()}</strong>
            <span style="
                background: {color};
                color: white;
                border-radius: 8px;
                padding: 2px 10px;
                margin-left: 8px;
                font-size: 0.85em;
                font-weight: bold;
            ">{label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction(result: dict) -> None:
    if not result:
        st.info("No aspects detected in this review.")
        return
    n_cols = min(len(result), 3)
    cols = st.columns(n_cols)
    for i, (aspect, sentiment) in enumerate(result.items()):
        with cols[i % n_cols]:
            aspect_badge(aspect, sentiment)


def render_batch_chart(df: pd.DataFrame) -> None:

    if df.empty:
        st.warning("No aspect predictions found in the uploaded data.")
        return

    pivot = df.pivot_table(
        index="aspect", columns="sentiment",
        values="count", aggfunc="sum", fill_value=0,
    )

    color_map = {
        "positive": "#27ae60",
        "negative": "#e74c3c",
        "neutral":  "#f39c12",
    }
    available = [s for s in ["positive", "neutral", "negative"] if s in pivot.columns]
    bar_colors = [color_map[s] for s in available]

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot[available].plot(
        kind="bar", ax=ax,
        color=bar_colors, edgecolor="white", width=0.7,
    )
    ax.set_title("Aspect-Wise Sentiment Distribution", fontsize=14)
    ax.set_xlabel("Aspect")
    ax.set_ylabel("Number of Reviews")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Sentiment")
    plt.tight_layout()
    st.pyplot(fig)


def no_models_warning() -> None:
    st.error(
        "**Models not found.**\n\n"
        "Train the models first:\n\n"
        "1. Run `notebooks/03_aspect_detection.ipynb`\n"
        "2. Run `notebooks/04_sentiment_analysis.ipynb`\n\n"
        "Models will be saved to `models/` automatically."
    )
