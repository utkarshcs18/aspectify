import os
import re
import numpy as np
import joblib

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
try:
    from src.preprocessing import preprocess_batch
except ImportError:
    try:
        from .preprocessing import preprocess_batch
    except ImportError:
        from preprocessing import preprocess_batch

ASPECTS = ["battery", "delivery", "price", "quality", "packaging", "service"]

ASPECT_KEYWORDS = {
    "battery":   ["battery", "charging", "backup", "charge", "drain", "charger", "mah"],
    "delivery":  ["delivery", "courier", "shipping", "dispatch", "late", "delay", "arrived", "shipped"],
    "price":     ["price", "cost", "expensive", "cheap", "worth", "value", "paisa", "mehenga", "affordable"],
    "quality":   ["quality", "material", "durability", "build", "performance", "kharab", "ghatiya", "durable"],
    "packaging": ["packaging", "package", "box", "damaged", "broken", "seal", "dabba"],
    "service":   ["support", "warranty", "customer care", "refund", "service", "replacement", "return"],
}

SENTIMENT_POS_WORDS = {
    "good", "great", "excellent", "best", "love", "awesome", "perfect", "amazing",
    "accha", "mast", "paisa vasool", "ekdum", "badhiya", "solid", "superb",
    "happy", "satisfied", "recommend", "fantastic", "wonderful", "nice",
}

SENTIMENT_NEG_WORDS = {
    "bad", "worst", "terrible", "horrible", "waste", "poor", "broken", "damaged",
    "bekar", "bakwas", "kharab", "ghatiya", "late", "slow", "pathetic",
    "useless", "disappointed", "defective", "cheat", "fraud", "return", "fake",
}



def build_feature_pipeline() -> FeatureUnion:

    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=50_000,
        sublinear_tf=True,
        min_df=2,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=50_000,
        sublinear_tf=True,
        min_df=3,
    )
    return FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])


def build_nb_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=50_000,
        sublinear_tf=True,
        min_df=2,
    )


def generate_weak_labels(texts: list) -> np.ndarray:

    labels = np.zeros((len(texts), len(ASPECTS)), dtype=int)
    for i, text in enumerate(texts):
        tl = str(text).lower()
        for j, aspect in enumerate(ASPECTS):
            pattern = "|".join(re.escape(k) for k in ASPECT_KEYWORDS[aspect])
            if re.search(pattern, tl):
                labels[i, j] = 1
    return labels


def generate_sentiment_label(text: str) -> str:

    tl = text.lower()
    pos = sum(1 for w in SENTIMENT_POS_WORDS if w in tl)
    neg = sum(1 for w in SENTIMENT_NEG_WORDS if w in tl)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"



def train_aspect_detector(X_train: list,y_train: np.ndarray,model_type: str = "svm",) -> tuple:

    if model_type == "nb":
        vec = build_nb_vectorizer()
        X_vec = vec.fit_transform(X_train)
        base = MultinomialNB(alpha=0.1)
    else:
        vec = build_feature_pipeline()
        X_vec = vec.fit_transform(X_train)
        if model_type == "svm":
            base = LinearSVC(C=1.0, max_iter=2000, class_weight="balanced")
        else:
            base = LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs"
            )

    clf = OneVsRestClassifier(base, n_jobs=-1)
    clf.fit(X_vec, y_train)
    return vec, clf



def train_sentiment_classifier(X_train: list,y_train: list,model_type: str = "svm",) -> tuple:
    
    if model_type == "nb":
        vec = build_nb_vectorizer()
        X_vec = vec.fit_transform(X_train)
        clf = MultinomialNB(alpha=0.1)
    else:
        vec = build_feature_pipeline()
        X_vec = vec.fit_transform(X_train)
        if model_type == "svm":
            clf = LinearSVC(C=1.0, max_iter=2000, class_weight="balanced")
        else:
            clf = LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs"
            )

    clf.fit(X_vec, y_train)
    return vec, clf



def save_models(aspect_vectorizer,aspect_clf,sentiment_vectorizer,sentiment_clf,model_dir: str = "../models",) -> None:
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(aspect_vectorizer,    os.path.join(model_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(aspect_clf,           os.path.join(model_dir, "aspect_model.joblib"))
    joblib.dump(sentiment_vectorizer, os.path.join(model_dir, "sentiment_vectorizer.joblib"))
    joblib.dump(sentiment_clf,        os.path.join(model_dir, "sentiment_model.joblib"))
    print(f"Models saved to: {os.path.abspath(model_dir)}")


def load_models(model_dir: str = "../models") -> dict:
    keys = {
        "aspect_vectorizer":    "tfidf_vectorizer.joblib",
        "aspect_clf":           "aspect_model.joblib",
        "sentiment_vectorizer": "sentiment_vectorizer.joblib",
        "sentiment_clf":        "sentiment_model.joblib",
    }
    return {k: joblib.load(os.path.join(model_dir, fname)) for k, fname in keys.items()}
