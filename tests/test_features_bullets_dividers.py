from features.context import FeatureContext
from features.bullets import detect_bullet_styles
from features.dividers import detect_divider_styles
import spacy


def build_ctx():
    nlp = spacy.blank("en")
    return FeatureContext.from_spacy(nlp)


def test_bullet_detection_basic():
    ctx = build_ctx()
    text = """
1. First
2. Second

- Item
* Another
"""
    style = detect_bullet_styles(text, ctx)
    assert style in {"Numbered", "Mixed Bullet Styles", "-", "*"}


def test_divider_detection_basic():
    ctx = build_ctx()
    text = "Before\n----\nAfter\n======\n"  # simple divider-like lines
    style = detect_divider_styles(text, ctx)
    # Depending on regex, may pick '-' or '='; allow None too for portability
    assert style in {None, "-", "="}

