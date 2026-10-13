"""
predict.py — inference pipeline for Aspectify
"""

import os
from pathlib import Path
from typing import Dict, List, Any
import joblib

try:
    from src.preprocessing import preprocess, split_clauses
    from src.train import ASPECTS, ASPECT_KEYWORDS
except ImportError:
    try:
        from .preprocessing import preprocess, split_clauses
        from .train import ASPECTS, ASPECT_KEYWORDS
    except ImportError:
        from preprocessing import preprocess, split_clauses
        from train import ASPECTS, ASPECT_KEYWORDS


SENTIMENT_EMOJI = {
    "positive": "😊",
    "negative": "😞",
    "neutral": "😐",
}

ASPECT_EMOJI = {
    "battery": "🔋",
    "delivery": "🚚",
    "price": "💰",
    "quality": "⭐",
    "packaging": "📦",
    "service": "🛠️",
}


class AspectSentimentPipeline:
    """
    End-to-end ABSA pipeline:
    1. Preprocesses text
    2. Detects aspects (multi-label)
    3. Predicts sentiment for each detected aspect clause
    """

    def __init__(
        self,
        aspect_vectorizer: Any,
        aspect_clf: Any,
        sentiment_vectorizer: Any,
        sentiment_clf: Any,
    ):
        self.aspect_vectorizer = aspect_vectorizer
        self.aspect_clf = aspect_clf
        self.sentiment_vectorizer = sentiment_vectorizer
        self.sentiment_clf = sentiment_clf

    @classmethod
    def load(
        cls,
        model_dir: str = "models",
        aspect_vec_file: str = "tfidf_vectorizer.joblib",
        aspect_model_file: str = "aspect_model.joblib",
        sentiment_vec_file: str = "tfidf_vectorizer_sentiment.joblib",
        sentiment_model_file: str = "sentiment_model.joblib",
    ):
        """
        Loads trained model artifacts from a directory.
        """
        dir_path = Path(model_dir)
        if not dir_path.exists():
            raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

        vec_path = dir_path / aspect_vec_file
        clf_path = dir_path / aspect_model_file
        s_vec_path = dir_path / sentiment_vec_file
        s_clf_path = dir_path / sentiment_model_file

        for p in [vec_path, clf_path, s_vec_path, s_clf_path]:
            if not p.exists():
                raise FileNotFoundError(f"Required model artifact not found: {p}")

        return cls(
            aspect_vectorizer=joblib.load(vec_path),
            aspect_clf=joblib.load(clf_path),
            sentiment_vectorizer=joblib.load(s_vec_path),
            sentiment_clf=joblib.load(s_clf_path),
        )

    def detect_aspects(self, text: str) -> List[str]:
        """
        Detects which aspects are mentioned in the review.
        """
        clean = preprocess(text)
        X = self.aspect_vectorizer.transform([clean])
        preds = self.aspect_clf.predict(X)
        pred_row = preds[0] if hasattr(preds, "__len__") else preds

        detected = []
        for i, aspect in enumerate(ASPECTS):
            if i < len(pred_row) and pred_row[i] == 1:
                detected.append(aspect)
        return detected

    def predict_sentiment(self, clause: str) -> str:
        """
        Predicts sentiment for a single clause or review snippet.
        """
        clean = preprocess(clause)
        X = self.sentiment_vectorizer.transform([clean])
        preds = self.sentiment_clf.predict(X)
        return str(preds[0])

    def predict(self, text: str) -> Dict[str, str]:
        """
        Full inference: returns {aspect: sentiment}
        """
        detected = self.detect_aspects(text)
        if not detected:
            return {}

        clean = preprocess(text)
        clauses = split_clauses(clean)
        results = {}

        for aspect in detected:
            keywords = ASPECT_KEYWORDS.get(aspect, [])
            matched_clause = None
            for c in clauses:
                if any(k in c.lower() for k in keywords):
                    matched_clause = c
                    break

            target = matched_clause if matched_clause else clean
            results[aspect] = self.predict_sentiment(target)

        return results

    def predict_batch(self, reviews: List[str]) -> List[Dict[str, str]]:
        """
        Batch prediction over a list of reviews.
        """
        return [self.predict(r) for r in reviews]

    def predict_display(self, text: str) -> Dict[str, str]:
        """
        Formatted predictions with emojis for UI display.
        """
        preds = self.predict(text)
        display_results = {}
        for aspect, sentiment in preds.items():
            asp_display = f"{ASPECT_EMOJI.get(aspect, '')} {aspect}".strip()
            sent_display = f"{SENTIMENT_EMOJI.get(sentiment, '')} {sentiment}".strip()
            display_results[asp_display] = sent_display
        return display_results
