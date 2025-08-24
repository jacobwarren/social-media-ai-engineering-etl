import os
import nltk
import spacy
from dataclasses import dataclass
from typing import Optional, Any, Set
from nltk.sentiment import SentimentIntensityAnalyzer

@dataclass
class NLPContext:
    lemmatizer: nltk.stem.WordNetLemmatizer
    stop_words: Set[str]
    sia: Optional[SentimentIntensityAnalyzer]
    nlp: Optional[Any]


def _maybe_download_nltk():
    if os.environ.get("PIPE_DOWNLOAD_NLTK", "0") == "1":
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except Exception:
            pass


def init_nlp_context() -> NLPContext:
    """Initialize NLTK + spaCy in one place and return a context.
    Mirrors the behavior in 26-train-grpo.py.
    """
    _maybe_download_nltk()

    # NLTK components
    lemmatizer = nltk.stem.WordNetLemmatizer()
    try:
        stop_words = set(nltk.corpus.stopwords.words('english'))
    except Exception:
        stop_words = set()

    try:
        sia = SentimentIntensityAnalyzer()
    except Exception:
        sia = None

    # spaCy model (best-effort)
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Loaded spaCy model for enhanced reward functions")
    except Exception:
        print("Warning: spaCy model not found. Falling back to heuristics.")
        nlp = None

    return NLPContext(
        lemmatizer=lemmatizer,
        stop_words=stop_words,
        sia=sia,
        nlp=nlp,
    )

