import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from preprocessing import (
    clean_text,
    normalize_hinglish,
    handle_negation,
    split_clauses,
    preprocess,
    preprocess_batch,
    map_emojis,
    NEGATION_WORDS,
)



class TestCleanText:
    def test_lowercase(self):
        assert clean_text("GREAT Battery") == "great battery"

    def test_removes_url(self):
        result = clean_text("Good product https://amazon.in/dp/xyz")
        assert "https" not in result
        assert "amazon.in" not in result

    def test_removes_html(self):
        result = clean_text("<b>Good</b> product")
        assert "<b>" not in result
        assert "good" in result

    def test_collapses_whitespace(self):
        result = clean_text("great   product   here")
        assert "  " not in result

    def test_non_string_input_does_not_raise(self):
        result = clean_text(None)
        assert isinstance(result, str)

    def test_preserves_comma(self):
        # Commas are clause-split boundaries — must be preserved
        result = clean_text("Battery good, delivery bad.")
        assert "," in result

    def test_empty_string(self):
        assert clean_text("") == ""



class TestNormalizeHinglish:
    @pytest.mark.parametrize("variant", ["acha", "achha", "achchha", "axha"])
    def test_accha_variants(self, variant):
        result = normalize_hinglish(f"product {variant} hai")
        assert "accha" in result, f"Failed for variant: {variant}"

    @pytest.mark.parametrize("variant", ["bakwaas", "bkwas"])
    def test_bakwas_variants(self, variant):
        result = normalize_hinglish(variant)
        assert "bakwas" in result, f"Failed for variant: {variant}"

    @pytest.mark.parametrize("variant", ["boht", "bohot", "bhut"])
    def test_bahut_variants(self, variant):
        result = normalize_hinglish(variant)
        assert "bahut" in result

    @pytest.mark.parametrize("variant", ["nhi", "nai", "nahin"])
    def test_nahi_variants(self, variant):
        result = normalize_hinglish(variant)
        assert "nahi" in result

    def test_does_not_mangle_normal_words(self):
        result = normalize_hinglish("quality is good")
        assert "quality" in result

    def test_no_mutation_of_unrelated(self):
        text = "delivery was on time"
        assert normalize_hinglish(text) == text


class TestHandleNegation:
    def test_not_marks_next_word(self):
        result = handle_negation("not good product")
        assert "NOT_good" in result

    def test_nahi_marks_next_word(self):
        result = handle_negation("accha nahi hai")
        assert "NOT_hai" in result

    def test_negation_window_length(self):
        result = handle_negation("not a b c d e", window=2)
        tokens = result.split()
        assert tokens[1] == "NOT_a"
        assert tokens[2] == "NOT_b"
        assert tokens[3] == "c"    # outside window

    def test_no_negation_unchanged(self):
        result = handle_negation("good product")
        assert "NOT_" not in result

    def test_returns_string(self):
        result = handle_negation("this is fine")
        assert isinstance(result, str)

    def test_negation_words_not_prefixed(self):
        # The negation word itself should NOT get a NOT_ prefix
        result = handle_negation("product not good")
        assert "NOT_not" not in result
        assert "NOT_good" in result


class TestSplitClauses:
    def test_split_on_but(self):
        clauses = split_clauses("battery good but delivery bad")
        assert len(clauses) == 2
        assert any("battery" in c for c in clauses)
        assert any("delivery" in c for c in clauses)

    def test_split_on_comma(self):
        clauses = split_clauses("battery good, delivery bad")
        assert len(clauses) == 2

    def test_split_on_lekin(self):
        clauses = split_clauses("quality accha hai lekin price zyada")
        assert len(clauses) == 2

    def test_split_on_aur(self):
        clauses = split_clauses("good quality aur fast delivery")
        assert len(clauses) == 2

    def test_single_clause(self):
        clauses = split_clauses("great product overall")
        assert len(clauses) == 1

    def test_empty_string(self):
        assert split_clauses("") == []

    def test_no_empty_clauses(self):
        clauses = split_clauses("good, , bad")
        assert all(c.strip() != "" for c in clauses)

    def test_returns_list(self):
        assert isinstance(split_clauses("some text"), list)



class TestPreprocess:
    def test_returns_string(self):
        result = preprocess("Battery bahut accha hai")
        assert isinstance(result, str)

    def test_lowercased(self):
        result = preprocess("BATTERY IS GOOD")
        assert result == result.lower()

    def test_negation_applied(self):
        result = preprocess("not good product", negation=True)
        assert "NOT_" in result

    def test_negation_disabled(self):
        result = preprocess("not good product", negation=False)
        assert "NOT_" not in result

    def test_hinglish_normalized(self):
        result = preprocess("product acha hai")
        assert "accha" in result

    def test_url_removed(self):
        result = preprocess("buy at https://flipkart.com now")
        assert "http" not in result


class TestPreprocessBatch:
    def test_returns_list(self):
        result = preprocess_batch(["text one", "text two"])
        assert isinstance(result, list)

    def test_length_preserved(self):
        texts = ["a", "b", "c", "d"]
        assert len(preprocess_batch(texts)) == len(texts)

    def test_handles_none_values(self):
        result = preprocess_batch([None, "good product"])
        assert len(result) == 2
        assert isinstance(result[0], str)


class TestMapEmojis:
    def test_thumbs_up_mapped(self):
        result = map_emojis("\U0001f44d great product")
        assert "good" in result
        assert "\U0001f44d" not in result

    def test_thumbs_down_mapped(self):
        result = map_emojis("very \U0001f44e quality")
        assert "bad" in result

    def test_no_emoji_unchanged(self):
        text = "good product"
        assert map_emojis(text) == text

    def test_multiple_emojis(self):
        result = map_emojis("\U0001f44d product \U0001f44e delivery")
        assert "good" in result
        assert "bad" in result



class TestNegationWords:
    def test_contains_english_words(self):
        assert "not" in NEGATION_WORDS
        assert "no" in NEGATION_WORDS
        assert "never" in NEGATION_WORDS

    def test_contains_hinglish_words(self):
        assert "nahi" in NEGATION_WORDS
        assert "nhi" in NEGATION_WORDS
        assert "mat" in NEGATION_WORDS
