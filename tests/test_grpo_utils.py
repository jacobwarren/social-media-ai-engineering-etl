import re
from training.grpo.prompt_parsing import (
    extract_prompt_content,
    extract_analysis_content,
    parse_writing_style_block,
    detect_urls,
)
from training.grpo.scenarios import get_scenario_type, normalize_scenario_score


PROMPT = (
    "# Request\n"
    "Create a LinkedIn post that **shares a step-by-step guide** on the topic of: `Content Marketing`\n\n"
    "### Key Message\n```\nBe helpful. Share concrete steps.\n```\n"
    "### Writing Constraints\n"
    "- **Suggested Post Length**: up to 1,100 characters\n"
    "- **Emoji Usage**: infrequent\n"
    "- **Tone**: professional\n"
    "- **Bullet Styles**: -\n"
)


def test_extract_prompt_content():
    c = extract_prompt_content(PROMPT)
    assert c['topic'] == 'Content Marketing'
    assert 'Be helpful' in c['key_message']
    assert c['constraints']['length'].startswith('up to')


def test_parse_writing_style_block():
    d = parse_writing_style_block(PROMPT)
    assert d['post_length_requirement'].startswith('up to')
    assert d['emoji_usage_requirement'] == 'infrequent'
    assert d['tone_requirement'] == 'professional'


def test_get_scenario_type_and_normalize():
    sid = get_scenario_type(PROMPT)
    assert sid == 0
    assert normalize_scenario_score(5.0, sid) == 0.5


def test_detect_urls_none():
    assert detect_urls(PROMPT) == []

