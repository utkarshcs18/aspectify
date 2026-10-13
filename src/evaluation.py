import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

ASPECTS    = ["battery", "delivery", "price", "quality", "packaging", "service"]
SENTIMENTS = ["positive", "neutral", "negative"]


def aspect_detection_report(y_true: np.ndarray,y_pred: np.ndarray,aspects: list = ASPECTS,) -> pd.DataFrame:
    
    rows = []
    for i, aspect in enumerate(aspects):
        rows.append({
            "aspect":    aspect,
            "precision": precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "recall":    recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "f1":        f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "support":   int(y_true[:, i].sum()),
        })
    df = pd.DataFrame(rows).set_index("aspect")
    macro = df[["precision", "recall", "f1"]].mean()
    df.loc["MACRO AVG"] = [macro["precision"], macro["recall"], macro["f1"], int(df["support"].sum())]
    return df.round(4)


def plot_aspect_f1_comparison(model_results: dict,title: str = "Aspect Detection — Macro F1 Comparison",) -> None:

    scores = {name: df.loc["MACRO AVG", "f1"] for name, df in model_results.items()}
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = sns.color_palette("muted", len(scores))
    bars = ax.bar(scores.keys(), scores.values(), color=palette, edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1")
    ax.set_title(title, fontsize=13)
    for bar, val in zip(bars, scores.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", fontsize=10,
        )
    plt.tight_layout()
    plt.show()


def sentiment_report(y_true: list,y_pred: list,labels: list = SENTIMENTS,) -> str:
    return classification_report(y_true, y_pred, labels=labels, zero_division=0)


def plot_confusion_matrix(y_true: list,y_pred: list,labels: list = SENTIMENTS,title: str = "Sentiment Confusion Matrix",) -> None:

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def comparison_table(metrics: dict) -> pd.DataFrame:

    rows = []
    for name, m in metrics.items():
        rows.append({
            "Model":              name,
            "Aspect Precision":   round(m.get("aspect_precision", 0), 4),
            "Aspect Recall":      round(m.get("aspect_recall", 0), 4),
            "Aspect Macro F1":    round(m.get("aspect_f1", 0), 4),
            "Sentiment Macro F1": round(m.get("sentiment_f1", 0), 4),
            "Sentiment Accuracy": round(m.get("sentiment_accuracy", 0), 4),
        })
    return pd.DataFrame(rows).set_index("Model")


def collect_errors(texts: list,y_true: np.ndarray,y_pred: np.ndarray,aspects: list = ASPECTS,n: int = 50,) -> pd.DataFrame:

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    wrong = np.where((y_true != y_pred).any(axis=1))[0][:n]
    rows = []
    for idx in wrong:
        rows.append({
            "text":        texts[idx],
            "true_labels": {a: int(y_true[idx, i]) for i, a in enumerate(aspects)},
            "pred_labels": {a: int(y_pred[idx, i]) for i, a in enumerate(aspects)},
        })
    return pd.DataFrame(rows)
