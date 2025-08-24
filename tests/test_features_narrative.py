import numpy as np
from features.narrative import analyze_narrative_flow, analyze_pacing, analyze_sentiment_arc


def test_narrative_flow_labels_intro_outro():
    text = "Today we announce a thing. Lots of details here. Please follow and sign up."
    labels = analyze_narrative_flow(text)
    assert labels[0] in {"Introduction/Setup", "Content"}
    assert labels[-1] in {"Outro/CTA", "Content"}


def test_analyze_pacing_percentiles():
    fast_text = "Short. Words are few. Tiny sentence."
    slow_text = "This sentence contains a significantly greater number of words for demonstration purposes. " * 3
    assert analyze_pacing(fast_text) == "Fast"
    assert analyze_pacing(slow_text) in {"Moderate", "Slow"}


def test_sentiment_arc_slope():
    rising = [-0.2, -0.1, 0.0, 0.1, 0.3]
    falling = [0.3, 0.1, 0.0, -0.1, -0.2]
    flat = [0.1, 0.09, 0.11, 0.1]
    assert analyze_sentiment_arc(rising) == "Rising"
    assert analyze_sentiment_arc(falling) == "Falling"
    assert analyze_sentiment_arc(flat) == "Flat"

