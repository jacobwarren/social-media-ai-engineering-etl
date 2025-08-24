import math
from training.rewards.bullet_style import bullet_style_reward_func
from training.rewards.tone import tone_alignment_reward_func
from training.rewards.hashtags import hashtag_limit_reward_func
from training.rewards.emoji import enhanced_emoji_usage
from training.rewards.structure import sentence_structure_reward_func
from training.rewards.semantic import semantic_coherence_reward



def test_bullet_style():
    prompts = ["Bullet Styles: Numbered"]
    completions = ["1. First\n2. Second\n3. Third"]
    s = bullet_style_reward_func(prompts, completions)
    assert s and s[0] >= 0.9


def test_tone_alignment():
    prompts = ["**Tone**: Serious"]
    completions = ["This is a serious note. It is not playful."]
    s = tone_alignment_reward_func(prompts, completions)
    assert s and 0.3 <= s[0] <= 1.0


def test_hashtag_limit():
    prompts = [""]
    completions = ["Great launch! #ai #ml #prod"]
    s = hashtag_limit_reward_func(prompts, completions)
    assert s and s[0] >= 0.7


def test_length_precise():
    prompts = ["**Suggested Post Length**: Up to 50 characters"]
    completions = ["Short post"]
    from training.rewards.length import precise_post_length
    s = precise_post_length(prompts, completions)
    assert s and s[0] >= 0.9


def test_emoji_enhanced():
    prompts = ["**Emoji Usage**: none"]
    completions = ["No emoji here."]
    s = enhanced_emoji_usage(prompts, completions)
    assert s and s[0] >= 0.9


def test_structure():
    prompts = [""]
    completions = ["Para one. It has two sentences.\n\nPara two. Also two."]
    s = sentence_structure_reward_func(prompts, completions)
    assert s and s[0] >= 0.9


def test_coherence():
    prompts = ["Launch an AI product for developers with features X and Y"]
    completions = ["Our AI product for developers includes features X and Y."]
    s = semantic_coherence_reward(prompts, completions)
    assert s and s[0] >= 0.6

