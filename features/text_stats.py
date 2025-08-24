from __future__ import annotations

from typing import List, Dict, Tuple
from nltk.tokenize import sent_tokenize, word_tokenize


def analyze_vocabulary_usage(text: str) -> int:
    words = word_tokenize(text)
    return len(set(words))


def analyze_sentence_structure(text: str) -> List[int]:
    sentences = sent_tokenize(text)
    return [len(word_tokenize(sentence)) for sentence in sentences]


def analyze_line_breaks(text: str) -> Tuple[int, float]:
    line_breaks = text.count('\n')
    lines = text.split('\n')
    avg_line_breaks = sum(len(line) == 0 for line in lines) / (len(lines) - 1) if len(lines) > 1 else 0.0
    return line_breaks, float(avg_line_breaks)


def punctuation_counts(text: str) -> Dict[str, int]:
    return {punc: text.count(punc) for punc in '.,;!?'}

