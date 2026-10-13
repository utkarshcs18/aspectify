import re
import string

HINGLISH_NORM_MAP = {
    "acha":          "accha",
    "achha":         "accha",
    "achchha":       "accha",
    "axha":          "accha",
    "acchaa":        "accha",
    "bekaar":        "bekar",
    "bakwaas":       "bakwas",
    "bkwas":         "bakwas",
    "boht":          "bahut",
    "bohot":         "bahut",
    "bhut":          "bahut",
    "dillivery":     "delivery",
    "delivry":       "delivery",
    "delievery":     "delivery",
    "sastaa":        "sasta",
    "mehnga":        "mehenga",
    "mahnga":        "mehenga",
    "paisa wasool":  "paisa vasool",
    "nhi":           "nahi",
    "nai":           "nahi",
    "nahin":         "nahi",
    "qualiy":        "quality",
    "qulaity":       "quality",
    "packging":      "packaging",
    "packagin":      "packaging",
    "khrb":          "kharab",
    "ghatia":        "ghatiya",
    "zabrdast":      "zabardast",
    "badhia":        "badhiya",
    "mast!":         "mast",
}


NEGATION_WORDS = frozenset({
    "not", "no", "never", "neither", "nor",
    "nahi", "nhi", "nai", "nahin", "mat", "na",
})


CLAUSE_SPLIT_PATTERN = re.compile(
    r"\bbut\b|\blekin\b|\bpar\b|\baur\b|\band\b"
    r"|\bhalanki\b|\bmagar\b|\bhowever\b|\bfir bhi\b|,",
    flags=re.IGNORECASE,
)


EMOJI_MAP = {
    "\U0001f621": " angry ",
    "\U0001f620": " angry ",
    "\U0001f624": " frustrated ",
    "\U0001f60d": " love ",
    "\u2764\ufe0f": " love ",
    "\U0001f495": " love ",
    "\U0001f60a": " happy ",
    "\U0001f601": " happy ",
    "\U0001f970": " happy ",
    "\U0001f622": " sad ",
    "\U0001f62d": " sad ",
    "\U0001f494": " sad ",
    "\U0001f44d": " good ",
    "\U0001f44e": " bad ",
    "\u2b50": " star ",
    "\U0001f31f": " star ",
    "\U0001f525": " hot ",
    "\U0001f4af": " perfect ",
    "\U0001f92e": " disgusting ",
    "\U0001f922": " bad ",
    "\U0001f612": " disappointed ",
    "\U0001f61e": " disappointed ",
    "\U0001f44f": " great ",
    "\u2705": " good ",
    "\u274c": " bad ",
}


def map_emojis(text: str) -> str:
    for emoji, word in EMOJI_MAP.items():
        text = text.replace(emoji, word)
    return text


def clean_text(text: str) -> str:

    if not isinstance(text, str):
        text = str(text)
    text = map_emojis(text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_hinglish(text: str) -> str:

    for variant, canonical in HINGLISH_NORM_MAP.items():
        text = re.sub(r"\b" + re.escape(variant) + r"\b", canonical, text)
    return text


def handle_negation(text: str, window: int = 3) -> str:

    tokens = text.split()
    result: list = []
    negate_remaining = 0

    for token in tokens:
        bare = token.strip(string.punctuation)
        if bare in NEGATION_WORDS:
            result.append(token)
            negate_remaining = window
        elif negate_remaining > 0:
            result.append("NOT_" + token)
            negate_remaining -= 1
            if bare in {",", ".", "!", "?", "but", "lekin", "par"}:
                negate_remaining = 0
        else:
            result.append(token)

    return " ".join(result)


def split_clauses(text: str) -> list:

    parts = CLAUSE_SPLIT_PATTERN.split(text)
    return [p.strip() for p in parts if p.strip()]


def preprocess(text: str, negation: bool = True) -> str:

    text = clean_text(text)
    text = normalize_hinglish(text)
    if negation:
        text = handle_negation(text)
    return text


def preprocess_batch(texts, negation: bool = True) -> list:
    return [preprocess(str(t), negation=negation) for t in texts]


# Alias for compatibility
split_into_clauses = split_clauses
