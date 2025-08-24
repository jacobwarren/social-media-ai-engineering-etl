from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set
import re

try:
    from nltk.corpus import stopwords as _stopwords
    from nltk.sentiment import SentimentIntensityAnalyzer as _SIA
except Exception:
    _stopwords = None
    _SIA = None


@dataclass
class FeatureContext:
    nlp: any
    stopwords_en: Set[str]
    sia: Optional[any]
    # Precompiled regexes
    bul_numbered_re: any
    bul_lettered_re: any
    bul_symbolic_re: any
    bul_indent_re: any
    divider_re: any

    @staticmethod
    def ensure_senter(nlp) -> None:
        """Ensure sentence segmentation when parser is disabled."""
        try:
            if 'senter' not in nlp.pipe_names and 'parser' not in nlp.pipe_names:
                nlp.add_pipe('senter')
        except Exception:
            pass

    @classmethod
    def from_spacy(cls, nlp) -> "FeatureContext":
        # Stopwords
        try:
            stop_en = set(_stopwords.words('english')) if _stopwords else set()
        except Exception:
            stop_en = set()
        # SIA
        try:
            sia = _SIA() if _SIA else None
        except Exception:
            sia = None
        # Regexes
        bul_numbered = re.compile(r'^\s*\d+[\.\)]\s+')
        bul_lettered = re.compile(r'^\s*[a-zA-Z]+[\.\)]\s+')
        bul_symbolic = re.compile(r'^\s*([^\w\s])')
        bul_indent = re.compile(r'^ {4,}([^\w\s])')
        divider = re.compile(r'^\s*([^\w\s])\1{3,}\s*$')
        # Ensure sentence segmentation if needed
        cls.ensure_senter(nlp)
        return cls(
            nlp=nlp,
            stopwords_en=stop_en,
            sia=sia,
            bul_numbered_re=bul_numbered,
            bul_lettered_re=bul_lettered,
            bul_symbolic_re=bul_symbolic,
            bul_indent_re=bul_indent,
            divider_re=divider,
        )

