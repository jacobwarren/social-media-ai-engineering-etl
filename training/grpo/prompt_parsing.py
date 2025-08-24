import re
from typing import List, Dict
from nltk.tokenize import word_tokenize


def extract_prompt_content(prompt: str) -> Dict:
    """
    Extract key content from the formatted markdown prompt.
    Handles the specific structure used in the writing style summary.
    """
    content = {
        'topic': '',
        'key_message': '',
        'common_phrases': [],
        'full_text': prompt,
        'constraints': {}
    }

    # Extract topic from the request line
    topic_pattern = r"on the topic of\`?\:?\s*\`?([^`\n]+)"
    topic_match = re.search(topic_pattern, prompt, flags=re.IGNORECASE)
    if topic_match:
        content['topic'] = topic_match.group(1).strip()

    # Extract key message from between triple backticks
    key_message_pattern = r"### Key Message\s*```\s*(.*?)\s*```"
    key_message_match = re.search(key_message_pattern, prompt, flags=re.IGNORECASE | re.DOTALL)
    if key_message_match:
        content['key_message'] = key_message_match.group(1).strip()

    # Extract common phrases
    common_phrases_pattern = r"\*\*Common Phrases\*\*\:\s*(.*?)(?:\n|\r\n|$)"
    phrases_match = re.search(common_phrases_pattern, prompt, flags=re.IGNORECASE)
    if phrases_match:
        phrases_text = phrases_match.group(1).strip()
        phrases = [p.strip() for p in phrases_text.split(',') if p.strip()]
        content['common_phrases'] = phrases

    # Extract writing constraints
    # Post length
    length_pattern = r"\*\*Suggested Post Length\*\*\:\s*(.*?)(?:\n|\r\n|$)"
    length_match = re.search(length_pattern, prompt, flags=re.IGNORECASE)
    if length_match:
        content['constraints']['length'] = length_match.group(1).strip()

    # Emoji usage
    emoji_pattern = r"\*\*Emoji Usage\*\*\:\s*(.*?)(?:\n|\r\n|$)"
    emoji_match = re.search(emoji_pattern, prompt, flags=re.IGNORECASE)
    if emoji_match:
        content['constraints']['emoji_usage'] = emoji_match.group(1).strip()

    # Tone
    tone_pattern = r"\*\*Tone\*\*\:\s*(.*?)(?:\n|\r\n|$)"
    tone_match = re.search(tone_pattern, prompt, flags=re.IGNORECASE)
    if tone_match:
        content['constraints']['tone'] = tone_match.group(1).strip()

    # Bullet style
    bullet_pattern = r"\*\*Bullet Styles\*\*\:\s*(.*?)(?:\n|\r\n|$)"
    bullet_match = re.search(bullet_pattern, prompt, flags=re.IGNORECASE)
    if bullet_match:
        content['bullet_style'] = bullet_match.group(1).strip()

    return content


def extract_analysis_content(prompt: str) -> Dict:
    """
    Extract content to analyze from structured markdown analysis prompts.
    """
    content = {
        'text_to_analyze': '',
        'categories': [],
        'constraints': {}
    }

    # Extract content between triple backticks (the text to analyze)
    content_pattern = r"```\s*(.*?)\s*```"
    content_matches = re.findall(content_pattern, prompt, flags=re.DOTALL)

    if content_matches:
        content['text_to_analyze'] = content_matches[0].strip()

    # Extract categories for classification tasks
    if "Structure Categories" in prompt:
        categories = []
        category_section = prompt.split("## Structure Categories")[1].split("##")[0]
        category_lines = category_section.strip().split("\n")
        for line in category_lines:
            if "**" in line:
                m = re.search(r"\*\*(.*?)\*\*", line)
                if m:
                    categories.append(m.group(1).lower())
        content['categories'] = categories

    elif "Available Tones" in prompt:
        tones_section = prompt.split("## Available Tones")[1].split("##")[0]
        tones = [t.strip().lower() for t in tones_section.strip().split(",")]
        content['categories'] = tones

    # Extract writing constraints
    constraints_pattern = r"## Writing Constraints\s*(.*?)(?=##|$)"
    constraints_match = re.search(constraints_pattern, prompt, flags=re.DOTALL)

    if constraints_match:
        constraints_text = constraints_match.group(1)
        # Response Type
        m = re.search(r"\*\*Response Type\*\*:\s*(.*?)(?:\n|$)", constraints_text)
        if m:
            content['constraints']['response_type'] = m.group(1).strip()
        # Format
        m = re.search(r"\*\*Format\*\*:\s*(.*?)(?:\n|$)", constraints_text)
        if m:
            content['constraints']['format'] = m.group(1).strip()
        # Length
        m = re.search(r"\*\*Length\*\*:\s*(.*?)(?:\n|$)", constraints_text)
        if m:
            content['constraints']['length'] = m.group(1).strip()

    return content


def parse_writing_style_block(user_prompt: str) -> dict:
    """
    Extract writing style requirements from the prompt.
    Enhanced to detect more style elements.
    """
    # Post length
    m = re.search(r"(?i)\-\s*Post\s+length:\s*(up to [\d,]+ characters)", user_prompt)
    if not m:
        m = re.search(r"(?i)\*\*Suggested Post Length\*\*:\s*(.*?)(?:\n|$)", user_prompt)
    post_length_req = m.group(1).strip().lower() if m else None

    # Emoji usage
    m = re.search(r"(?i)\-\s*Emoji\s+Usage:\s*(none|infrequent|frequent)", user_prompt)
    if not m:
        m = re.search(r"(?i)\*\*Emoji Usage\*\*:\s*(.*?)(?:\n|$)", user_prompt)
    emoji_usage_req = m.group(1).strip().lower() if m else None

    # Bullet style
    m = re.search(r"(?i)\-\s*Bullet\s+Styles?:\s*([^\n]+)", user_prompt)
    if not m:
        m = re.search(r"(?i)\*\*Bullet Styles\*\*:\s*(.*?)(?:\n|$)", user_prompt)
    bullet_style_req = m.group(1).strip() if m else None

    # Tone
    m = re.search(r"(?i)\-\s*Tone:\s*([^\n]+)", user_prompt)
    if not m:
        m = re.search(r"(?i)\*\*Tone\*\*:\s*(.*?)(?:\n|$)", user_prompt)
    tone_req = m.group(1).strip() if m else None

    return {
        "post_length_requirement": post_length_req,
        "emoji_usage_requirement": emoji_usage_req,
        "bullet_style_requirement": bullet_style_req,
        "tone_requirement": tone_req,
    }


def detect_urls(text: str) -> List[str]:
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(pattern, text)


def detect_potential_people_names(text: str, nlp=None) -> List[str]:
    """Detect potential people names using spaCy if available, else heuristic fallback."""
    if not nlp:
        words = word_tokenize(text)
        potential_names: List[str] = []
        for i in range(len(words) - 1):
            if (
                words[i][0:1].isalpha() and words[i][0].isupper() and len(words[i]) > 1 and
                words[i+1][0:1].isalpha() and words[i+1][0].isupper() and len(words[i+1]) > 1
            ):
                potential_names.append(f"{words[i]} {words[i+1]}")
        return potential_names
    doc = nlp(text[:10000])
    return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]


def detect_organization_names(text: str, nlp=None) -> List[str]:
    """Detect organization names using spaCy if available; otherwise, return []."""
    if not nlp:
        return []
    doc = nlp(text[:10000])
    return [ent.text for ent in doc.ents if ent.label_ == "ORG"]

