import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np
from unittest.mock import MagicMock

from predict import AspectSentimentPipeline, SENTIMENT_EMOJI, ASPECT_EMOJI
from train import ASPECTS


def make_pipeline(
    detected: list = None,
    sentiment: str = "positive",
) -> AspectSentimentPipeline:
    if detected is None:
        detected = ["battery"]

    pred_arr = np.array(
        [[1 if a in detected else 0 for a in ASPECTS]]
    )

    av = MagicMock()
    ac = MagicMock()
    sv = MagicMock()
    sc = MagicMock()

    av.transform.return_value = MagicMock()
    ac.predict.return_value   = pred_arr
    sv.transform.return_value = MagicMock()
    sc.predict.return_value   = [sentiment]

    return AspectSentimentPipeline(
        aspect_vectorizer    = av,
        aspect_clf           = ac,
        sentiment_vectorizer = sv,
        sentiment_clf        = sc,
    )



class TestDetectAspects:
    def test_returns_list(self):
        pl = make_pipeline(["battery"])
        assert isinstance(pl.detect_aspects("battery is good"), list)

    def test_detected_aspect_present(self):
        pl = make_pipeline(["battery"])
        result = pl.detect_aspects("battery is good")
        assert "battery" in result

    def test_no_detection_returns_empty(self):
        pl = make_pipeline([])
        assert pl.detect_aspects("great product") == []

    def test_multiple_aspects(self):
        pl = make_pipeline(["battery", "delivery"])
        result = pl.detect_aspects("battery good but delivery late")
        assert "battery" in result
        assert "delivery" in result

    def test_all_aspects_valid(self):
        pl = make_pipeline(ASPECTS)
        result = pl.detect_aspects("comprehensive review")
        assert all(a in ASPECTS for a in result)



class TestPredictSentiment:
    def test_returns_string(self):
        pl = make_pipeline(sentiment="positive")
        assert isinstance(pl.predict_sentiment("battery is great"), str)

    @pytest.mark.parametrize("sentiment", ["positive", "negative", "neutral"])
    def test_all_sentiment_labels(self, sentiment):
        pl = make_pipeline(sentiment=sentiment)
        assert pl.predict_sentiment("some clause") == sentiment



class TestPredict:
    def test_returns_dict(self):
        pl = make_pipeline(["battery"])
        assert isinstance(pl.predict("battery is good"), dict)

    def test_empty_when_no_aspects(self):
        pl = make_pipeline([])
        assert pl.predict("interesting text") == {}

    def test_keys_are_valid_aspects(self):
        pl = make_pipeline(["battery", "delivery"])
        result = pl.predict("battery good but delivery late")
        assert all(k in ASPECTS for k in result.keys())

    def test_values_are_valid_sentiments(self):
        pl = make_pipeline(["battery"])
        result = pl.predict("battery is great")
        assert all(v in ("positive", "neutral", "negative") for v in result.values())

    def test_hinglish_review(self):
        pl = make_pipeline(["battery", "delivery"], sentiment="positive")
        result = pl.predict("Battery accha hai but delivery late thi")
        assert isinstance(result, dict)



class TestPredictBatch:
    def test_returns_list(self):
        pl = make_pipeline(["battery"])
        assert isinstance(pl.predict_batch(["battery good", "delivery bad"]), list)

    def test_length_preserved(self):
        pl = make_pipeline(["battery"])
        reviews = ["a", "b", "c"]
        assert len(pl.predict_batch(reviews)) == len(reviews)

    def test_each_element_is_dict(self):
        pl = make_pipeline(["battery"])
        result = pl.predict_batch(["battery is fine"])
        assert isinstance(result[0], dict)

    def test_empty_list(self):
        pl = make_pipeline(["battery"])
        assert pl.predict_batch([]) == []


class TestPredictDisplay:
    def test_returns_dict(self):
        pl = make_pipeline(["battery"])
        assert isinstance(pl.predict_display("battery accha"), dict)

    def test_values_contain_emoji(self):
        pl = make_pipeline(["battery"], sentiment="positive")
        result = pl.predict_display("battery is great")
        for val in result.values():
            # Should contain one of the three sentiment emojis
            assert any(e in val for e in SENTIMENT_EMOJI.values())

    def test_no_aspects_returns_empty(self):
        pl = make_pipeline([])
        assert pl.predict_display("some text") == {}



class TestLoadErrors:
    def test_missing_dir_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            AspectSentimentPipeline.load("this_does_not_exist/")



class TestConstants:
    def test_sentiment_emoji_has_three_keys(self):
        assert set(SENTIMENT_EMOJI.keys()) == {"positive", "negative", "neutral"}

    def test_aspect_emoji_covers_all_aspects(self):
        for aspect in ASPECTS:
            assert aspect in ASPECT_EMOJI, f"Missing emoji for: {aspect}"
