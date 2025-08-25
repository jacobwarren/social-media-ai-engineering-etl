import argparse
parser = argparse.ArgumentParser(description="GRPO training")
parser.add_argument("--run-id", default=None)
parser.add_argument("--base-dir", default="data/processed")
parser.add_argument("--models-dir", default="models")
parser.add_argument("--use-aggregator", action="store_true", default=False)
parser.add_argument("--weights", default=None, help="Path to JSON file with reward weights")
parser.add_argument("--seed", type=int, default=3407)
args, _ = parser.parse_known_args()


# ETL cohesion: logging, manifest, run-id latest resolution
from utils.logging_setup import init_pipeline_logging
from utils.manifest import read_manifest, discover_input, should_skip, compute_hash, update_stage
from utils.run_id import get_last_run_id

logger = init_pipeline_logging("phase3.train_grpo", None, "26-train-grpo")

if args.run_id == "latest":
    latest = get_last_run_id(args.base_dir)
    if latest:
        args.run_id = latest

# Early idempotent skip if model already exists when run-id provided
if args.run_id:
    out_dir = os.path.join(args.models_dir, args.run_id, "grpo-model")
    if os.path.isdir(out_dir):
        logger.info(f"Model already exists at {out_dir}; skipping training")
        raise SystemExit(0)


# -*- coding: utf-8 -*-
import os
import re
import torch
import nltk
import emojis
import numpy as np
import spacy
from thefuzz import process, fuzz
from utils.seed import set_global_seed
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import wandb

# Patching TRL for GRPO must happen before trainer/model construction
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from unsloth import is_bfloat16_supported

# Centralized NLP setup (preserve globals for downstream calls)
from training.grpo.nlp_setup import init_nlp_context
_ctx = init_nlp_context()
lemmatizer = _ctx.lemmatizer
stop_words = _ctx.stop_words
sia = _ctx.sia
nlp = _ctx.nlp

set_global_seed(3407)
# --- 2) Load the Base Model with Unsloth (4-bit + LoRA for GRPO) ---

# Optional modular rewards import (non-breaking). If not available, training continues.
try:
    # Temporary adapter: expose current reward funcs via a module import path
    # so training/rewards can import them. This keeps current code intact.
    import types
    _rewards_impl = types.SimpleNamespace(
        bullet_style_reward_func=bullet_style_reward_func,
        tone_alignment_reward_func=tone_alignment_reward_func,
        hashtag_limit_reward_func=hashtag_limit_reward_func,
        chinese_character_reward_func=chinese_character_reward_func,
        semantic_coherence_reward=semantic_coherence_reward,
        sentence_structure_reward_func=sentence_structure_reward_func,
        vocabulary_usage_reward_func=vocabulary_usage_reward_func,
        emoji_variety_reward=emoji_variety_reward,
    )
except Exception:
    pass

# --- 3) Data Preparation ---
from datasets import load_dataset, Dataset

from training.grpo.prompt_parsing import extract_prompt_content, extract_analysis_content, parse_writing_style_block, detect_urls, detect_potential_people_names, detect_organization_names
    """
    Extract key content from the formatted markdown prompt.
    Handles the specific structure used in the writing style summary.
    """
    content = {
        'topic': '',
        'key_message': '',
        'common_phrases': [],
        'full_text': prompt
    }

    # Extract topic from the request line
    topic_pattern = r"on the topic of\`?\:?\s*\`?([^`\n]+)"
    topic_match = re.search(topic_pattern, prompt, flags=re.IGNORECASE)
    if topic_match:
        content['topic'] = topic_match.group(1).strip()

    # Extract key message from between triple backticks
    key_message_pattern = r"### Key Message\s*```\s*(.*?)\s*```"
    key_message_match = re.search(key_message_pattern, prompt, flags=re.IGNORECASE|re.DOTALL)
    if key_message_match:
        content['key_message'] = key_message_match.group(1).strip()

    # Extract common phrases
    common_phrases_pattern = r"\*\*Common Phrases\*\*\:\s*(.*?)(?:\n|\r\n|$)"
    phrases_match = re.search(common_phrases_pattern, prompt, flags=re.IGNORECASE)
    if phrases_match:
        phrases_text = phrases_match.group(1).strip()
        # Split phrases by comma and clean them
        phrases = [p.strip() for p in phrases_text.split(',') if p.strip()]
        content['common_phrases'] = phrases

    # Extract writing constraints
    constraints = {}

    # Extract post length
    length_pattern = r"\*\*Suggested Post Length\*\*\:\s*(.*?)(?:\n|\r\n|$)"
    length_match = re.search(length_pattern, prompt, flags=re.IGNORECASE)
    if length_match:
        constraints['length'] = length_match.group(1).strip()

    # Extract emoji usage
    emoji_pattern = r"\*\*Emoji Usage\*\*\:\s*(.*?)(?:\n|\r\n|$)"
    emoji_match = re.search(emoji_pattern, prompt, flags=re.IGNORECASE)
    if emoji_match:
        constraints['emoji_usage'] = emoji_match.group(1).strip()

    # Extract tone
    tone_pattern = r"\*\*Tone\*\*\:\s*(.*?)(?:\n|\r\n|$)"
    tone_match = re.search(tone_pattern, prompt, flags=re.IGNORECASE)
    if tone_match:
        constraints['tone'] = tone_match.group(1).strip()

    content['constraints'] = constraints

    # Extract any bullet style information
    bullet_pattern = r"\*\*Bullet Styles\*\*\:\s*(.*?)(?:\n|\r\n|$)"
    bullet_match = re.search(bullet_pattern, prompt, flags=re.IGNORECASE)
    if bullet_match:
        content['bullet_style'] = bullet_match.group(1).strip()

    return content


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
        # The first match is typically the content to analyze
        content['text_to_analyze'] = content_matches[0].strip()

    # Extract categories for classification tasks
    if "Structure Categories" in prompt:
        categories = []
        category_section = prompt.split("## Structure Categories")[1].split("##")[0]
        category_lines = category_section.strip().split("\n")
        for line in category_lines:
            if "**" in line:
                category = re.search(r"\*\*(.*?)\*\*", line)
                if category:
                    categories.append(category.group(1).lower())
        content['categories'] = categories

    # Extract available tones for tone analysis
    elif "Available Tones" in prompt:
        tones_section = prompt.split("## Available Tones")[1].split("##")[0]
        tones = [t.strip().lower() for t in tones_section.strip().split(",")]
        content['categories'] = tones

    # Extract writing constraints
    constraints_pattern = r"## Writing Constraints\s*(.*?)(?=##|$)"
    constraints_match = re.search(constraints_pattern, prompt, flags=re.DOTALL)

    if constraints_match:
        constraints_text = constraints_match.group(1)

        # Extract response type
        response_type_match = re.search(r"\*\*Response Type\*\*:\s*(.*?)(?:\n|$)", constraints_text)
        if response_type_match:
            content['constraints']['response_type'] = response_type_match.group(1).strip()

        # Extract format
        format_match = re.search(r"\*\*Format\*\*:\s*(.*?)(?:\n|$)", constraints_text)
        if format_match:
            content['constraints']['format'] = format_match.group(1).strip()

        # Extract length constraint
        length_match = re.search(r"\*\*Length\*\*:\s*(.*?)(?:\n|$)", constraints_text)
        if length_match:
            content['constraints']['length'] = length_match.group(1).strip()

    return content


    """
    Extract writing style requirements from the prompt.
    Enhanced to detect more style elements.
    """
    # Extract post length requirement
    post_length_match = re.search(r"(?i)\-\s*Post\s+length:\s*(up to [\d,]+ characters)", user_prompt)
    if not post_length_match:
        post_length_match = re.search(r"(?i)\*\*Suggested Post Length\*\*:\s*(.*?)(?:\n|$)", user_prompt)
    post_length_req = post_length_match.group(1).strip().lower() if post_length_match else None

    # Extract emoji usage requirement
    emoji_usage_match = re.search(r"(?i)\-\s*Emoji\s+Usage:\s*(none|infrequent|frequent)", user_prompt)
    if not emoji_usage_match:
        emoji_usage_match = re.search(r"(?i)\*\*Emoji Usage\*\*:\s*(.*?)(?:\n|$)", user_prompt)
    emoji_usage_req = emoji_usage_match.group(1).strip().lower() if emoji_usage_match else None

    # Extract bullet style requirement
    bullet_style_match = re.search(r"(?i)\-\s*Bullet\s+Styles?:\s*([^\n]+)", user_prompt)
    if not bullet_style_match:
        bullet_style_match = re.search(r"(?i)\*\*Bullet Styles\*\*:\s*(.*?)(?:\n|$)", user_prompt)
    bullet_style_req = bullet_style_match.group(1).strip() if bullet_style_match else None

    # Extract tone requirement
    tone_match = re.search(r"(?i)\-\s*Tone:\s*([^\n]+)", user_prompt)
    if not tone_match:
        tone_match = re.search(r"(?i)\*\*Tone\*\*:\s*(.*?)(?:\n|$)", user_prompt)
    tone_req = tone_match.group(1).strip() if tone_match else None

    return {
        "post_length_requirement": post_length_req,
        "emoji_usage_requirement": emoji_usage_req,
        "bullet_style_requirement": bullet_style_req,
        "tone_requirement": tone_req
    }


    """
    Detect URLs in text.
    """
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    return urls


    """
    Detect potential people names using spaCy's entity recognition.
    """
    if not nlp:
        # Fallback if spaCy not available
        words = word_tokenize(text)
        potential_names = []
        for i in range(len(words) - 1):
            # Look for capitalized word pairs that might be names
            if (words[i][0].isupper() and len(words[i]) > 1 and
                words[i+1][0].isupper() and len(words[i+1]) > 1):
                potential_names.append(f"{words[i]} {words[i+1]}")
        return potential_names

    # Use spaCy for named entity recognition
    doc = nlp(text[:10000])  # Limit for performance
    people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    return people


    """
    Detect organization names using spaCy's entity recognition.
    """
    if not nlp:
        return []

    doc = nlp(text[:10000])  # Limit for performance
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return orgs

from training.grpo.scenarios import normalize_scenario_score, get_scenario_type
    """
    Standardize scores across different scenarios to a 0-1 range.

    Args:
        score: The raw score from the scenario reward function
        scenario_id: The scenario type ID

    Returns:
        float: Normalized score between 0 and 1
    """
    # Define the exact maximum possible score for each scenario
    # based on the weights used in scenario_reward_func
    max_scores = {
        0: 10.0,  # LinkedIn post (sum of all weights)
        1: 10.0,  # Topic classification
        2: 10.0,  # Opinion extraction
        3: 10.0,  # Tone analysis
        4: 10.0,  # Structure classification
        5: 10.0   # Generic scenario
    }

    # Normalize to 0-1 range
    normalized = score / max_scores.get(scenario_id, 10.0)
    return min(normalized, 1.0)  # Cap at 1.0


    """
    Identify the scenario type from the prompt.
    Updated to handle markdown-formatted prompts where actual content starts on third line.
    """
    # Split the prompt into lines and get the third line (index 2)
    lines = user_prompt.strip().split("\n")

    # Make sure there are at least 3 lines
    if len(lines) >= 4:
        relevant_line = lines[4].lower()
    else:
        # If there aren't enough lines, just use the first available line
        relevant_line = lines[0].lower() if lines else ""

    # LinkedIn post creation scenarios
    if "create a linkedin post that" in relevant_line:
        return 0

    # For other scenario types, we should check the entire prompt text
    # since the identifying patterns might not be in just the third line

    # Look for topic classification patterns (scenario 1)
    if any(pattern in user_prompt.lower() for pattern in [
        "Analyze the following social media post and identify its primary topic"
    ]):
        return 1

    # Look for opinion extraction patterns (scenario 2)
    if any(pattern in user_prompt.lower() for pattern in [
        "Extract the core opinion from this social media post and present it in first person"
    ]):
        return 2

    # Look for tone analysis patterns (scenario 3)
    if any(pattern in user_prompt.lower() for pattern in [
        "Analyze this social media post and identify up to three primary tones"
    ]):
        return 3

    # Look for structure classification patterns (scenario 4)
    if any(pattern in user_prompt.lower() for pattern in [
        "Classify the structural format of this social media post"
    ]):
        return 4

    # Default to unknown scenario
    return 5

def extract_keywords(text, max_keywords=15):
    """
    Extract important keywords from text using spaCy entity recognition
    and frequency analysis.
    """
    if not nlp:
        # Fallback if spaCy not available
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 3]
        return set(tokens[:max_keywords])

    # Process with spaCy
    doc = nlp(text[:10000])  # Limit for performance

    # Extract named entities
    entities = [ent.text.lower() for ent in doc.ents if len(ent.text) > 3]

    # Extract other important words (nouns, etc.)
    important_words = [token.text.lower() for token in doc
                      if token.pos_ in ["NOUN", "PROPN", "ADJ"]
                      and token.text.lower() not in stop_words
                      and len(token.text) > 3]

    # Combine and prioritize entities
    all_keywords = entities + important_words
    counter = Counter(all_keywords)

    # Return top keywords
    return set([word for word, _ in counter.most_common(max_keywords)])

def detect_bullet_styles(text):
    """
    Enhanced bullet style detection from the writing style extraction script.
    Returns the dominant bullet style used or None if no bullets detected.
    """
    lines = text.split('\n')
    bullet_list = []
    numbered_pattern = re.compile(r'^\s*\d+[\.\)]\s+')
    lettered_pattern = re.compile(r'^\s*[a-zA-Z]+[\.\)]\s+')
    symbolic_pattern = re.compile(r'^\s*([^\w\s])')

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if numbered_pattern.match(line):
            bullet_list.append("Numbered")
            continue
        if lettered_pattern.match(line):
            bullet_list.append("Lettered")
            continue
        indent_bullet = re.match(r'^ {4,}([^\w\s])', line)
        if indent_bullet:
            bullet_list.append(indent_bullet.group(1))
            continue
        sym_match = symbolic_pattern.match(line)
        if sym_match:
            bullet_list.append(sym_match.group(1))
            continue
        first_word = line.split()[0] if line.split() else ''
        if first_word and all(emojis.count(char) for char in first_word):
            bullet_list.append("EmojiBullets" if len(first_word) > 1 else "Emoji")
            continue

    if not bullet_list:
        return None

    counts = Counter(bullet_list)
    emoji_entries = [b for b in counts if "Emoji" in b or emojis.count(b) > 0]
    if len(emoji_entries) > 1:
        return "Differing Emojis"
    if len(counts) > 1:
        return "Mixed Bullet Styles"
    style, _ = counts.most_common(1)[0]
    return style

def get_sentiment_scores(text):
    """
    Get sentiment scores for text, analyzing by sentence.
    """
    sentences = sent_tokenize(text[:5000])  # Limit text length for performance
    if len(sentences) > 10:
        # Analyze a subset for long texts
        indices = sorted(np.random.choice(len(sentences), min(10, len(sentences)), replace=False))
        sentences = [sentences[i] for i in indices]

    sentiment_scores = []
    for sentence in sentences:
        score = sia.polarity_scores(sentence)['compound']
        sentiment_scores.append(score)

    return sentiment_scores

def analyze_sentiment_arc(sentiment_scores):
    """
    Analyze the sentiment arc to determine overall tone.
    """
    if len(sentiment_scores) < 3:
        return "Neutral"  # Not enough data

    avg_score = np.mean(sentiment_scores)
    variance = np.var(sentiment_scores)

    # Classify the overall tone
    if avg_score > 0.4:
        return "Very Positive"
    elif avg_score > 0.2:
        return "Positive"
    elif avg_score < -0.4:
        return "Very Negative"
    elif avg_score < -0.2:
        return "Negative"
    elif variance > 0.2:
        return "Mixed"
    else:
        return "Neutral"

def analyze_narrative_flow(text, window_size=3):
    """
    Analyze the narrative flow/structure of the text.
    """
    # Simple approximation for speed
    sentences = sent_tokenize(text[:5000])  # Limit for performance
    if not sentences:
        return []

    sentiment_scores = [sia.polarity_scores(s)['compound'] for s in sentences]

    # If very short, just return basic flow
    if len(sentences) <= 4:
        return ["Introduction/Setup"]

    # Identify potential segments based on sentiment shifts
    flow_markers = []
    prev_score = sentiment_scores[0]
    for i, score in enumerate(sentiment_scores[1:], 1):
        # Detect significant shifts
        if abs(score - prev_score) > 0.4:
            if i < len(sentences) // 3:
                flow_markers.append("Introduction/Setup")
            elif i < 2 * len(sentences) // 3:
                flow_markers.append("Transition/Development")
            else:
                flow_markers.append("Conclusion/Reflection")
        prev_score = score

    # Ensure we have at least some basic flow markers
    if not flow_markers:
        flow_markers = ["Introduction/Setup"]
        if len(sentences) > 8:
            flow_markers.append("Development")
            flow_markers.append("Conclusion")

    return flow_markers

def analyze_pacing(text):
    """
    Determine the pacing of the text (Fast, Moderate, Slow).
    """
    sentences = sent_tokenize(text[:5000])  # Limit for performance
    if len(sentences) < 3:
        return "Short/Not Enough Data"

    sentence_lengths = [len(word_tokenize(s)) for s in sentences]
    avg_len = np.mean(sentence_lengths)
    stddev = np.std(sentence_lengths)

    # More sophisticated pacing analysis
    if stddev > 7:  # High variation
        return "Variable"
    elif avg_len < 10:
        return "Fast"
    elif avg_len > 20:
        return "Slow"
    else:
        return "Moderate"

def analyze_narrative_structure(text):
    """
    Analyze the narrative structure of the text.
    Returns a tuple of (flow, pacing, sentiment_arc).
    """
    flow = analyze_narrative_flow(text)
    pacing = analyze_pacing(text)
    sentiment_scores = get_sentiment_scores(text)
    sentiment_arc = analyze_sentiment_arc(sentiment_scores)
    return flow, pacing, sentiment_arc

# --- 5) Rewards moved to training.grpo.rewards (PR3) ---
from training.rewards.bullet_style import bullet_style_reward_func
from training.rewards.tone import tone_alignment_reward_func
from training.rewards.hashtags import hashtag_limit_reward_func
from training.rewards.language import chinese_character_reward_func

    """
    Enhanced bullet style reward function using more sophisticated detection.
    """
    scores = []

    for i, c in enumerate(completions):
        user_prompt = prompts[i][-1] if isinstance(prompts[i], list) else prompts[i]

        # Try to parse desired bullet style from prompt
        bullet_style_match = re.search(r"(?i)Bullet\s+Styles?:\s*(.*)", user_prompt)
        if not bullet_style_match:
            bullet_style_match = re.search(r"(?i)\*\*Bullet Styles\*\*:\s*(.*?)(?:\n|$)", user_prompt)

        bullet_style_info = bullet_style_match.group(1).lower().strip() if bullet_style_match else ""

        # Identify the desired styles
        desired_styles = []
        if "•" in bullet_style_info or "dot" in bullet_style_info:
            desired_styles.append("•")
        if "differing emojis" in bullet_style_info:
            desired_styles.append("Differing Emojis")
        if "emoji" in bullet_style_info:
            desired_styles.append("Emoji")
        if "-" in bullet_style_info or "dash" in bullet_style_info:
            desired_styles.append("-")
        if "ordered" in bullet_style_info or "numbered" in bullet_style_info:
            desired_styles.append("Numbered")
        if "lettered" in bullet_style_info or "alphabetical" in bullet_style_info:
            desired_styles.append("Lettered")

        # Get content to analyze
        raw_text = c if isinstance(c, str) else c[0]
        answer_text = raw_text
        if not answer_text.strip():
            answer_text = raw_text

        # Use the enhanced bullet style detection
        detected_style = detect_bullet_styles(answer_text)

        # Calculate score based on matched styles
        if not detected_style:
            # No bullets found
            score = 0.0
        elif not desired_styles:
            # No style specified but bullets found
            score = 0.7  # Partial credit for any bullet usage
        else:
            # Check if detected style matches any desired style
            if detected_style == "Mixed Bullet Styles":
                # Partial credit for mixed styles
                score = 0.5
            elif any(desired in detected_style for desired in desired_styles):
                # Full credit for matching style
                score = 1.0
            else:
                # Some bullet style but not matching
                score = 0.3

        scores.append(score)

    return scores

# --- 5c) Tone Alignment Reward ---
# Tone mapping for reference
TONE_TO_SENTIMENT = {
    # Positive tones map to positive sentiment
    "adventurous": "positive", "artistic": "positive", "assertive": "positive",
    "authoritative": "positive", "bold": "positive", "bright": "positive",
    "capable": "positive", "caring": "positive", "casual": "positive",
    "charming": "positive", "cheerful": "positive", "clever": "positive",
    "colorful": "positive", "conversational": "positive", "creative": "positive",
    # (truncated for brevity - refer to the full list in the detailed code)

    # Neutral tones
    "calm": "neutral", "comfortable": "neutral", "dry": "neutral",
    "formal": "neutral", "industrial": "neutral", "informative": "neutral",
    "no-nonsense": "neutral", "stable": "neutral", "unconventional": "neutral",
    "versatile": "neutral",

    # Negative tones
    "cocky": "negative", "rebellious": "negative", "sarcastic": "negative",
    "serious": "negative",
}


    """
    Enhanced tone alignment reward function using more sophisticated sentiment analysis.
    """
    scores = []

    for i, completion in enumerate(completions):
        # Get user prompt
        user_prompt = prompts[i][-1] if isinstance(prompts[i], list) else prompts[i]

        # Parse tone requirements from prompt
        tone_field = re.search(r"[Tt]one:\s*(.+)", user_prompt)
        if not tone_field:
            tone_field = re.search(r"\*\*Tone\*\*:\s*(.*?)(?:\n|$)", user_prompt)

        if not tone_field:
            # No tone specified
            raw_text = completion if isinstance(completion, str) else completion[0]
            answer_text = raw_text
            if not answer_text.strip():
                answer_text = raw_text

            # Give partial credit based on sentiment neutrality/positivity
            sentiment_scores = get_sentiment_scores(answer_text)
            sentiment_arc = analyze_sentiment_arc(sentiment_scores)

            # Simple reward when no tone specified
            if sentiment_arc in ["Positive", "Very Positive", "Neutral"]:
                scores.append(0.7)
            else:
                scores.append(0.3)
            continue

        # Parse requested tones
        tone_str = tone_field.group(1).lower()
        multi_tones = tone_str.replace(" and ", ",")
        tone_list = [t.strip() for t in multi_tones.split(",") if t.strip()]
        tone_list = tone_list[:3]  # Limit to 3 tones

        if not tone_list:
            scores.append(0.2)
            continue

        # Extract text to analyze
        raw_text = completion if isinstance(completion, str) else completion[0]
        answer_text = raw_text
        if not answer_text.strip():
            answer_text = raw_text

        # Get sentence-level sentiment scores
        sentiment_scores = get_sentiment_scores(answer_text)

        # Calculate overall sentiment metrics
        avg_score = np.mean(sentiment_scores) if sentiment_scores else 0
        variance = np.var(sentiment_scores) if len(sentiment_scores) > 1 else 0
        sentiment_arc = analyze_sentiment_arc(sentiment_scores)

        # Calculate per-tone rewards
        per_tone_rewards = []
        for user_tone in tone_list:
            # Match the tone to our known list
            best_match, best_score = process.extractOne(
                query=user_tone,
                choices=list(TONE_TO_SENTIMENT.keys()),
                scorer=fuzz.ratio
            )

            if best_score < 60:
                # Can't match well
                per_tone_rewards.append(0.2)
                continue

            # Get the expected sentiment for this tone
            sentiment_target = TONE_TO_SENTIMENT[best_match]

            # Enhanced tone matching using both average and variance
            if sentiment_target == "positive":
                if avg_score > 0.3:
                    reward = 1.0
                elif avg_score > 0.1:
                    reward = 0.8
                elif avg_score > -0.1:
                    reward = 0.4
                else:
                    reward = 0.0

            elif sentiment_target == "negative":
                if avg_score < -0.3:
                    reward = 1.0
                elif avg_score < -0.1:
                    reward = 0.8
                elif avg_score < 0.1:
                    reward = 0.4
                else:
                    reward = 0.0

            elif sentiment_target == "neutral":
                # For neutral tones, low variance is key
                if abs(avg_score) < 0.1 and variance < 0.1:
                    reward = 1.0
                elif abs(avg_score) < 0.2 and variance < 0.2:
                    reward = 0.8
                elif abs(avg_score) < 0.3:
                    reward = 0.4
                else:
                    reward = 0.2
            else:
                reward = 0.3  # Fallback

            per_tone_rewards.append(reward)

        # Average the per-tone rewards
        final_score = sum(per_tone_rewards) / len(per_tone_rewards)
        scores.append(final_score)

    return scores


    """
    Reward function that encourages posts to have no more than 3 hashtags at the end.
    Returns 1.0 for completions with 0-3 hashtags at the end, and a lower score
    when more hashtags are present.
    """
    scores = []

    for completion in completions:
        # Extract text to analyze
        raw_text = completion if isinstance(completion, str) else completion[0]
        answer_text = raw_text
        if not answer_text.strip():
            answer_text = raw_text

        # Find hashtags at the end of the text
        # We'll consider hashtags at the end if they're in the last paragraph or line
        paragraphs = answer_text.strip().split('\n\n')
        last_paragraph = paragraphs[-1] if paragraphs else ""

        lines = last_paragraph.split('\n')
        last_line = lines[-1] if lines else ""

        # Regex to find hashtags - words that start with # and contain word characters
        hashtags = re.findall(r'#\w+', last_line)
        hashtag_count = len(hashtags)

        # Alternative approach: look in the last 20% of the text for hashtags
        if not hashtags and answer_text:
            last_fifth = answer_text[int(len(answer_text) * 0.8):]
            hashtags = re.findall(r'#\w+', last_fifth)
            hashtag_count = len(hashtags)

        # Score based on hashtag count
        if hashtag_count <= 3:
            score = 1.0  # Optimal: 0-3 hashtags
        elif hashtag_count <= 5:
            score = 0.7  # Acceptable: 4-5 hashtags
        elif hashtag_count <= 7:
            score = 0.4  # Too many: 6-7 hashtags
        else:
            score = 0.2  # Excessive: 8+ hashtags

        scores.append(score)

    return scores


    """
    Reward function that discourages the use of Chinese characters in completions.
    Returns 1.0 for completions with no Chinese characters, and a lower score
    when Chinese characters are present.
    """
    scores = []

    for completion in completions:
        # Extract text to analyze
        raw_text = completion if isinstance(completion, str) else completion[0]
        answer_text = raw_text
        if not answer_text.strip():
            answer_text = raw_text

        # Check for Chinese characters
        # Main CJK Unified Ideographs block (covers most common Chinese characters)
        chinese_char_count = 0
        total_chars = len(answer_text)

        for char in answer_text:
            # Check if character is in the main CJK Unified Ideographs block
            # or in the CJK Unified Ideographs Extension blocks
            if (0x4E00 <= ord(char) <= 0x9FFF or  # Main CJK block
                0x3400 <= ord(char) <= 0x4DBF or  # Extension A
                0x20000 <= ord(char) <= 0x2A6DF or  # Extension B
                0x2A700 <= ord(char) <= 0x2B73F or  # Extension C
                0x2B740 <= ord(char) <= 0x2B81F or  # Extension D
                0x2B820 <= ord(char) <= 0x2CEAF or  # Extension E
                0x2CEB0 <= ord(char) <= 0x2EBEF):  # Extension F
                chinese_char_count += 1

        if total_chars == 0:
            scores.append(1.0)  # No text to analyze
            continue

        # Calculate percentage of Chinese characters
        chinese_percentage = chinese_char_count / total_chars

        # Score is inverse of percentage (1.0 = no Chinese, 0.0 = all Chinese)
        if chinese_char_count == 0:
            score = 1.0  # No Chinese characters
        elif chinese_percentage < 0.01:
            score = 0.8  # Very few Chinese characters (possibly names/loan words)
        elif chinese_percentage < 0.05:
            score = 0.5  # Some Chinese characters
        elif chinese_percentage < 0.2:
            score = 0.2  # Many Chinese characters
        else:
            score = 0.0  # Text with significant Chinese content

        scores.append(score)

    return scores

# --- 5e) Post Length and Emoji Usage Rewards ---
def categorize_emoji_usage(frequency, bins=None):
    """
    Return a label based on frequency thresholds.
    More precise categorization from your provided code.
    """
    if bins is None:
        bins = [
            (0, "none"),
            (0.0005, "very low"),   # up to 0.05%
            (0.001, "low"),         # up to 0.1%
            (0.005, "medium"),      # up to 0.5%
            (0.01, "high"),         # up to 1%
            (1.0, "extreme")        # anything above 1% or up to 100%
        ]

    last_label = "none"
    for threshold, label in bins:
        if frequency <= threshold:
            return label
        last_label = label

    return last_label

def emoji_frequency_analysis(post, threshold=0.002):
    """
    Analyze emoji usage frequency in text.
    Returns a dict with emoji counts and categorization.
    """
    emoji_count = emojis.count(post)
    text_length = len(post)

    if text_length == 0:
        frequency = 0.0
    else:
        frequency = emoji_count / text_length

    usage_label = categorize_emoji_usage(frequency)

    return {
        "emoji_count": emoji_count,
        "text_length": text_length,
        "frequency": frequency,
        "usage": usage_label
    }

def enhanced_emoji_usage_reward(required_usage: str, completion_text: str) -> float:
    """
    More sophisticated emoji usage reward that uses the precise categories
    and gives partial credit for near matches.
    """
    if not required_usage:
        return 0.5  # No requirement specified

    # Normalize required usage to match our categories
    required_lower = required_usage.lower()
    if required_lower in ["none", "very low", "low", "medium", "high", "extreme"]:
        required_category = required_lower
    elif required_lower == "infrequent":
        required_category = "low"
    elif required_lower == "frequent":
        required_category = "high"
    else:
        required_category = "medium"  # Default fallback

    # Get actual emoji usage
    emoji_count = emojis.count(completion_text)
    text_length = len(completion_text)
    if text_length == 0:
        frequency = 0
    else:
        frequency = emoji_count / text_length

    actual_category = categorize_emoji_usage(frequency)

    # Define category ordering for distance calculation
    category_order = ["none", "very low", "low", "medium", "high", "extreme"]

    # Perfect match
    if actual_category == required_category:
        return 1.0

    # Calculate distance between categories
    try:
        required_idx = category_order.index(required_category)
        actual_idx = category_order.index(actual_category)
        distance = abs(required_idx - actual_idx)

        # Convert distance to score (closer = higher score)
        if distance == 1:  # Adjacent categories
            return 0.7
        elif distance == 2:  # Two categories apart
            return 0.4
        else:  # Three or more categories apart
            return 0.0
    except ValueError:
        return 0.3  # If category not found, give minimal credit

def emoji_usage_reward(required_usage: str, completion_text: str) -> float:
    """
    Compare actual emoji usage with required usage.
    Simpler version for backward compatibility.
    """
    if not required_usage:
        return 0.5

    analysis = emoji_frequency_analysis(completion_text)
    actual_usage = analysis["usage"]

    # Map from detailed categories to simple ones
    if actual_usage in ["none"]:
        simple_actual = "none"
    elif actual_usage in ["very low", "low"]:
        simple_actual = "infrequent"
    elif actual_usage in ["medium", "high", "extreme"]:
        simple_actual = "frequent"
    else:
        simple_actual = "infrequent"  # Fallback

    # Map required usage to simple categories
    if required_usage.lower() in ["none"]:
        simple_required = "none"
    elif required_usage.lower() in ["infrequent", "very low", "low"]:
        simple_required = "infrequent"
    elif required_usage.lower() in ["frequent", "medium", "high", "extreme"]:
        simple_required = "frequent"
    else:
        simple_required = "infrequent"  # Fallback

    if simple_required == simple_actual:
        return 1.0

    # Partial credit logic
    if simple_required == "none":
        if simple_actual == "infrequent":
            return 0.5
        else:  # actual_usage=frequent
            return 0.0

    elif simple_required == "infrequent":
        if simple_actual == "none":
            return 0.5
        elif simple_actual == "frequent":
            return 0.3
        else:
            return 1.0  # actual_usage=infrequent (handled above)

    elif simple_required == "frequent":
        if simple_actual == "infrequent":
            return 0.7
        elif simple_actual == "none":
            return 0.0
        else:
            return 1.0  # actual_usage=frequent (handled above)

    return 0.0  # fallback

from training.rewards.base import analyze_narrative_structure
from training.rewards.narrative import narrative_structure_reward_func

from training.rewards.emoji import enhanced_emoji_usage_reward, emoji_usage_reward
from training.rewards.vocabulary import vocabulary_usage_reward_func
from training.rewards.linebreaks import line_break_reward_func
from training.rewards.punctuation import punctuation_usage_reward_func
from training.rewards.divider import divider_style_reward_func
from training.rewards.semantic import semantic_coherence_reward
from training.rewards.structure import sentence_structure_reward_func
from training.rewards.topics import topic_shifts_reward_func


def precise_post_length_reward(required_length: str, completion_text: str) -> float:
    """
    More precise post length reward that gives higher scores
    when closer to the upper bound of the allowed range.
    """
    if not required_length:
        return 0.5  # No requirement specified

    length = len(completion_text)

    # Parse the required length range
    if "up to 750" in required_length.lower():
        max_chars = 750
        # Give higher scores when closer to the limit (but still under)
        if length <= max_chars:
            # Scale from 0.7 to 1.0 based on how close to the limit
            return 0.7 + (0.3 * min(1.0, length / max_chars))
        else:
            # Over the limit - penalty increases with distance
            overage_ratio = (length - max_chars) / max_chars
            if overage_ratio <= 0.1:  # Up to 10% over
                return 0.6
            elif overage_ratio <= 0.25:  # Up to 25% over
                return 0.3
            else:
                return 0.0

    elif "between 750 and 1,500" in required_length.lower():
        min_chars = 750
        max_chars = 1500

        if length < min_chars:
            # Under minimum - partial credit based on how close
            return 0.7 * (length / min_chars)
        elif length <= max_chars:
            # In range - higher score when closer to upper bound
            range_size = max_chars - min_chars
            position_in_range = length - min_chars
            return 0.7 + (0.3 * (position_in_range / range_size))
        else:
            # Over maximum - penalty increases with distance
            overage_ratio = (length - max_chars) / max_chars
            if overage_ratio <= 0.1:  # Up to 10% over
                return 0.6
            elif overage_ratio <= 0.25:  # Up to 25% over
                return 0.3
            else:
                return 0.0

    elif "between 1,500 and 3,000" in required_length.lower():
        min_chars = 1500
        max_chars = 3000

        if length < min_chars:
            # Under minimum - partial credit based on how close
            return 0.7 * (length / min_chars)
        elif length <= max_chars:
            # In range - higher score when closer to upper bound
            range_size = max_chars - min_chars
            position_in_range = length - min_chars
            return 0.7 + (0.3 * (position_in_range / range_size))
        else:
            # Over maximum - penalty increases with distance
            overage_ratio = (length - max_chars) / max_chars
            if overage_ratio <= 0.1:  # Up to 10% over
                return 0.6
            elif overage_ratio <= 0.25:  # Up to 25% over
                return 0.3
            else:
                return 0.0

    else:
        # Use simpler approach for compatibility
        return post_length_reward(required_length, completion_text)


    """
    Return 1.0 if completion meets the required length constraint.
    Simpler version for backward compatibility.
    """
    if not required_length:
        return 0.5

    length = len(completion_text)

    # Parse the numeric range
    match = re.search(r"up\s+to\s+(\d+(,\d+)?)\s+characters", required_length)
    if not match:
        return 0.5

    # Remove commas and convert to int
    max_chars_str = match.group(1).replace(",", "")
    max_chars = int(max_chars_str)

    # Check if within limit
    if length <= max_chars:
        return 1.0
    else:
        # Partial credit if slightly over
        overage_ratio = (length - max_chars) / max_chars
        if overage_ratio <= 0.1:  # Up to 10% over
            return 0.7
        elif overage_ratio <= 0.25:  # Up to 25% over
            return 0.4
        else:
            return 0.0

# --- 5f) Narrative Structure Reward ---
def narrative_structure_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Reward function that evaluates the narrative structure quality.
    """
    scores = []

    for completion in completions:
        raw_text = completion if isinstance(completion, str) else completion[0]
        answer_text = raw_text
        if not answer_text.strip():
            answer_text = raw_text

        # Analyze narrative structure
        flow, pacing, sentiment_arc = analyze_narrative_structure(answer_text)

        # Calculate structure score based on coherence
        structure_score = 0.0

        # Check for coherent pacing
        if pacing not in ["Short/Not Enough Data"]:
            structure_score += 0.4

        # Check for identifiable sentiment arc
        if sentiment_arc not in ["Neutral"]:
            structure_score += 0.3

        # Check for defined flow
        if flow and flow[0] not in ["Short Content", "Mixed Flow"]:
            structure_score += 0.3

        scores.append(structure_score)

    return scores

# --- 5g) Emoji Variety Reward ---
from training.rewards.emoji_variety import emoji_variety_reward

    """
    Rewards using a variety of different emoji rather than repeating the same ones.
    Only applicable when emoji usage is desired.
    """
    # Extract all emoji from the text
    all_emoji = [c for c in completion_text if emojis.count(c) > 0]
    total_emoji = len(all_emoji)

    if total_emoji == 0:
        return 0.0  # No emoji used

    # Count unique emoji
    unique_emoji = set(all_emoji)
    unique_count = len(unique_emoji)

    # Calculate variety ratio
    variety_ratio = unique_count / total_emoji

    # Scale the reward - higher variety gets higher score
    if variety_ratio == 1.0:  # All emoji are unique
        return 1.0
    elif variety_ratio >= 0.8:  # High variety
        return 0.9
    elif variety_ratio >= 0.6:  # Good variety
        return 0.7
    elif variety_ratio >= 0.4:  # Moderate variety
        return 0.5
    elif variety_ratio >= 0.2:  # Low variety
        return 0.3
    else:  # Very low variety
        return 0.1

# --- 5h) Semantic Coherence Reward ---
def semantic_coherence_reward(prompts, completions, **kwargs) -> list[float]:
    """
    Rewards content that maintains proper semantic coherence and topic focus.
    Uses basic NLP techniques to estimate topic shifts.
    """
    scores = []

    for i, completion in enumerate(completions):
        user_prompt = prompts[i][-1] if isinstance(prompts[i], list) else prompts[i]

        # Get answer text
        raw_text = completion if isinstance(completion, str) else completion[0]
        answer_text = raw_text
        if not answer_text.strip():
            answer_text = raw_text

        # Extract prompt content to determine if coherence is desired
        prompt_content = extract_prompt_content(user_prompt)

        # If there's a "Topic Shifts" indicator in the prompt, use it
        topic_shift_expected = False
        if "**Topic Shifts**:" in user_prompt:
            topic_shift_match = re.search(r"\*\*Topic Shifts\*\*:\s*(.*?)(?:\n|\r\n|$)", user_prompt)
            if topic_shift_match:
                topic_shift_text = topic_shift_match.group(1).lower()
                # If consistent focus is mentioned, then shifts are not desired
                topic_shift_expected = "consistent" not in topic_shift_text

        # Split text into sentences and paragraphs
        sentences = sent_tokenize(answer_text)
        paragraphs = [p.strip() for p in answer_text.split('\n\n') if p.strip()]

        # Too short to analyze
        if len(sentences) < 3 or len(paragraphs) < 2:
            scores.append(0.5)  # Neutral score for very short content
            continue

        # 1. Check for abrupt sentiment shifts
        sentiment_scores = []
        for sentence in sentences:
            score = sia.polarity_scores(sentence)['compound']
            sentiment_scores.append(score)

        # Calculate sentiment volatility
        sentiment_shifts = 0
        for i in range(1, len(sentiment_scores)):
            if abs(sentiment_scores[i] - sentiment_scores[i-1]) > 0.5:  # Significant shift
                sentiment_shifts += 1

        sentiment_volatility = sentiment_shifts / (len(sentiment_scores) - 1)

        # 2. Check for topic consistency across paragraphs
        paragraph_similarities = []
        if nlp:
            # Create embeddings for each paragraph
            paragraph_docs = [nlp(p[:1000]) for p in paragraphs]  # Limit size for performance

            # Calculate similarities between consecutive paragraphs
            for i in range(1, len(paragraph_docs)):
                similarity = paragraph_docs[i].similarity(paragraph_docs[i-1])
                paragraph_similarities.append(similarity)

            # Average similarity (higher = more consistent)
            avg_similarity = sum(paragraph_similarities) / len(paragraph_similarities) if paragraph_similarities else 0.5
        else:
            # Fallback without spaCy
            avg_similarity = 0.5  # Neutral

        # 3. Determine overall coherence score
        if topic_shift_expected:
            # If shifts are expected/desired, reward moderate values
            coherence_score = 1.0 - abs(0.5 - sentiment_volatility) - abs(0.5 - avg_similarity)
        else:
            # If consistency is desired, reward low volatility and high similarity
            coherence_score = 1.0 - sentiment_volatility + (avg_similarity - 0.5)

        # Normalize score
        normalized_score = max(0.0, min(1.0, coherence_score))
        scores.append(normalized_score)

    return scores

# --- 5j) Sentence Structure Reward ---
def sentence_structure_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Rewards completions that match the requested sentence structure
    (short sentences, long sentences, or mix).
    """
    scores = []

    for i, completion in enumerate(completions):
        user_prompt = prompts[i][-1] if isinstance(prompts[i], list) else prompts[i]

        # Extract the requested sentence structure from the prompt
        structure_match = re.search(r"\*\*Sentence Structure\*\*:\s*(.*?)(?:\n|$)", user_prompt)
        requested_structure = None
        if structure_match:
            structure_text = structure_match.group(1).lower()

            if "short" in structure_text and "sentences" in structure_text:
                requested_structure = "short"
            elif "long" in structure_text and "complex" in structure_text:
                requested_structure = "long"
            elif "mix" in structure_text or "balanced" in structure_text:
                requested_structure = "balanced"

        # Extract text to analyze
        raw_text = completion if isinstance(completion, str) else completion[0]
        answer_text = raw_text
        if not answer_text.strip():
            answer_text = raw_text

        # No sentence structure specified
        if not requested_structure:
            scores.append(0.5)  # Neutral score
            continue

        # Analyze sentence structure
        sentences = sent_tokenize(answer_text)
        if len(sentences) < 2:
            scores.append(0.3)  # Too short to analyze properly
            continue

        # Calculate sentence lengths
        sentence_lengths = [len(word_tokenize(s)) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        length_variance = np.var(sentence_lengths)

        # Score based on requested structure
        if requested_structure == "short":
            # Short sentences: lower average length is better
            if avg_length < 10:
                score = 1.0
            elif avg_length < 15:
                score = 0.7
            elif avg_length < 20:
                score = 0.4
            else:
                score = 0.2

        elif requested_structure == "long":
            # Long sentences: higher average length is better
            if avg_length > 20:
                score = 1.0
            elif avg_length > 15:
                score = 0.7
            elif avg_length > 10:
                score = 0.4
            else:
                score = 0.2

        else:  # balanced/mixed
            # Balance is good avg length with appropriate variance
            if 10 <= avg_length <= 20 and length_variance > 20:
                score = 1.0  # Good mix of lengths
            elif 10 <= avg_length <= 20:
                score = 0.7  # Good average but not enough variety
            elif length_variance > 20:
                score = 0.6  # Good variety but average is off
            else:
                score = 0.4  # Neither good average nor good variety

        scores.append(score)

    return scores

# --- 5k) Vocabulary Usage Reward ---
def vocabulary_usage_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Rewards completions with rich vocabulary usage when requested.
    """
    scores = []

    for i, completion in enumerate(completions):
        user_prompt = prompts[i][-1] if isinstance(prompts[i], list) else prompts[i]

        # Extract the requested vocabulary richness from the prompt
        vocab_match = re.search(r"\*\*Vocabulary Usage\*\*:\s*(.*?)(?:\n|$)", user_prompt)
        requested_vocab = None
        if vocab_match:
            vocab_text = vocab_match.group(1).lower()

            if "rich" in vocab_text:
                requested_vocab = "rich"
            elif "developed" in vocab_text:
                requested_vocab = "developed"
            elif "normal" in vocab_text:
                requested_vocab = "normal"
            elif "conservative" in vocab_text or "narrow" in vocab_text:
                requested_vocab = "conservative"

        # Extract text to analyze
        raw_text = completion if isinstance(completion, str) else completion[0]
        answer_text = raw_text
        if not answer_text.strip():
            answer_text = raw_text

        # No vocabulary requirement specified
        if not requested_vocab:
            scores.append(0.5)  # Neutral score
            continue

        # Analyze vocabulary usage
        words = [word.lower() for word in word_tokenize(answer_text) if word.isalpha()]
        if not words:
            scores.append(0.2)  # No words to analyze
            continue

        total_words = len(words)
        unique_words = len(set(words))
        vocab_ratio = unique_words / total_words

        # Score based on requested vocabulary richness
        if requested_vocab == "rich":
            if vocab_ratio > 0.5:  # Very rich vocabulary
                score = 1.0
            elif vocab_ratio > 0.4:
                score = 0.8
            elif vocab_ratio > 0.3:
                score = 0.5
            else:
                score = 0.3

        elif requested_vocab == "developed":
            if 0.35 < vocab_ratio <= 0.5:  # Developed vocabulary
                score = 1.0
            elif 0.3 < vocab_ratio <= 0.35 or 0.5 < vocab_ratio <= 0.6:
                score = 0.8
            elif 0.25 < vocab_ratio <= 0.3 or 0.6 < vocab_ratio:
                score = 0.5
            else:
                score = 0.3

        elif requested_vocab == "normal":
            if 0.25 < vocab_ratio <= 0.35:  # Normal vocabulary
                score = 1.0
            elif 0.2 < vocab_ratio <= 0.25 or 0.35 < vocab_ratio <= 0.4:
                score = 0.8
            elif 0.15 < vocab_ratio <= 0.2 or 0.4 < vocab_ratio <= 0.5:
                score = 0.5
            else:
                score = 0.3

        else:  # conservative/narrow
            if vocab_ratio <= 0.25:  # Conservative vocabulary
                score = 1.0
            elif vocab_ratio <= 0.3:
                score = 0.8
            elif vocab_ratio <= 0.35:
                score = 0.5
            else:
                score = 0.3

        scores.append(score)

    return scores

# --- 5l) Line Break Reward ---
def line_break_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Rewards completions that match the requested line break pattern.
    """
    scores = []

    for i, completion in enumerate(completions):
        user_prompt = prompts[i][-1] if isinstance(prompts[i], list) else prompts[i]

        # Extract the requested line break style from the prompt
        linebreak_match = re.search(r"\*\*Line Break Usage\*\*:\s*(.*?)(?:\n|$)", user_prompt)
        requested_style = None
        if linebreak_match:
            style_text = linebreak_match.group(1).lower()

            if "frequent" in style_text:
                requested_style = "frequent"
            elif "fewer" in style_text or "compact" in style_text:
                requested_style = "fewer"
            elif "no " in style_text or "continuous" in style_text:
                requested_style = "none"
            elif "moderate" in style_text:
                requested_style = "moderate"

        # Extract text to analyze
        raw_text = completion if isinstance(completion, str) else completion[0]
        answer_text = raw_text
        if not answer_text.strip():
            answer_text = raw_text

        # No line break style specified
        if not requested_style:
            scores.append(0.5)  # Neutral score
            continue

        # Analyze line breaks
        lines = answer_text.split('\n')
        text_length = len(answer_text)
        line_count = len(lines)

        # Calculate line break metrics
        if text_length == 0 or line_count <= 1:
            line_break_ratio = 0
        else:
            line_break_ratio = (line_count - 1) / text_length * 100  # Line breaks per 100 chars

        # Score based on requested line break style
        if requested_style == "frequent":
            if line_break_ratio > 2:  # Very frequent line breaks
                score = 1.0
            elif line_break_ratio > 1.5:
                score = 0.8
            elif line_break_ratio > 1:
                score = 0.6
            elif line_break_ratio > 0.5:
                score = 0.4
            else:
                score = 0.2

        elif requested_style == "fewer":
            if 0.2 < line_break_ratio <= 0.8:  # Fewer line breaks
                score = 1.0
            elif 0 < line_break_ratio <= 0.2 or 0.8 < line_break_ratio <= 1.2:
                score = 0.7
            elif line_break_ratio > 1.2:
                score = 0.3
            else:
                score = 0.5  # No line breaks at all is not ideal for "fewer"

        elif requested_style == "none":
            if line_break_ratio == 0:  # No line breaks
                score = 1.0
            elif line_break_ratio <= 0.2:
                score = 0.7
            elif line_break_ratio <= 0.5:
                score = 0.4
            else:
                score = 0.2

        else:  # moderate
            if 0.8 < line_break_ratio <= 1.5:  # Moderate line breaks
                score = 1.0
            elif 0.5 < line_break_ratio <= 0.8 or 1.5 < line_break_ratio <= 2:
                score = 0.8
            elif 0.2 < line_break_ratio <= 0.5 or 2 < line_break_ratio <= 2.5:
                score = 0.5
            else:
                score = 0.3

        scores.append(score)

    return scores

# --- 5m) Punctuation Usage Reward ---
def punctuation_usage_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Rewards completions that match the requested punctuation usage pattern.
    """
    scores = []

    for i, completion in enumerate(completions):
        user_prompt = prompts[i][-1] if isinstance(prompts[i], list) else prompts[i]

        # Extract punctuation preferences from the prompt
        punct_match = re.search(r"\*\*Punctuation\*\*:\s*(.*?)(?:\n|$)", user_prompt)
        requested_style = None
        if punct_match:
            punct_text = punct_match.group(1).lower()

            # Check for specific punctuation preferences
            periods_heavy = "heavy use of periods" in punct_text
            commas_heavy = "heavy use of commas" in punct_text
            exclamation_heavy = "heavy use of exclamation" in punct_text
            question_heavy = "heavy use of question" in punct_text
            semicolon_heavy = "heavy use of semicolons" in punct_text

            requested_style = {
                "periods": 2 if periods_heavy else 1,  # 2 = heavy, 1 = normal
                "commas": 2 if commas_heavy else 1,
                "exclamation": 2 if exclamation_heavy else 1,
                "question": 2 if question_heavy else 1,
                "semicolon": 2 if semicolon_heavy else 1
            }

        # Extract text to analyze
        raw_text = completion if isinstance(completion, str) else completion[0]
        answer_text = raw_text
        if not answer_text.strip():
            answer_text = raw_text

        # No punctuation preferences specified
        if not requested_style:
            scores.append(0.5)  # Neutral score
            continue

        # Count punctuation
        text_length = len(answer_text)
        if text_length == 0:
            scores.append(0.3)  # No text to analyze
            continue

        punct_counts = {
            "periods": answer_text.count('.') / text_length,
            "commas": answer_text.count(',') / text_length,
            "exclamation": answer_text.count('!') / text_length,
            "question": answer_text.count('?') / text_length,
            "semicolon": answer_text.count(';') / text_length
        }

        # Define thresholds for punctuation frequency
        thresholds = {
            "periods": {"low": 0.01, "normal": 0.02, "heavy": 0.03},
            "commas": {"low": 0.01, "normal": 0.02, "heavy": 0.03},
            "exclamation": {"low": 0.001, "normal": 0.005, "heavy": 0.01},
            "question": {"low": 0.001, "normal": 0.005, "heavy": 0.01},
            "semicolon": {"low": 0.0005, "normal": 0.001, "heavy": 0.002}
        }

        # Calculate score for each punctuation type
        type_scores = []
        for punct_type, desired_level in requested_style.items():
            actual_freq = punct_counts[punct_type]

            if desired_level == 2:  # Heavy usage desired
                if actual_freq >= thresholds[punct_type]["heavy"]:
                    type_scores.append(1.0)
                elif actual_freq >= thresholds[punct_type]["normal"]:
                    type_scores.append(0.7)
                elif actual_freq >= thresholds[punct_type]["low"]:
                    type_scores.append(0.4)
                else:
                    type_scores.append(0.1)
            else:  # Normal usage desired
                if thresholds[punct_type]["low"] <= actual_freq <= thresholds[punct_type]["normal"]:
                    type_scores.append(1.0)
                elif actual_freq < thresholds[punct_type]["low"]:
                    type_scores.append(0.6)  # Too little
                elif actual_freq < thresholds[punct_type]["heavy"]:
                    type_scores.append(0.8)  # A bit too much
                else:
                    type_scores.append(0.4)  # Way too much

        # Average the scores for each punctuation type
        scores.append(sum(type_scores) / len(type_scores))

    return scores

# --- 5n) Divider Style Reward ---
def divider_style_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Rewards completions that include the requested divider style.
    """
    scores = []

    for i, completion in enumerate(completions):
        user_prompt = prompts[i][-1] if isinstance(prompts[i], list) else prompts[i]

        # Extract the requested divider style from the prompt
        divider_match = re.search(r"\*\*Section Divider\*\*:\s*`([^\`]+)`", user_prompt)
        requested_divider = None
        if divider_match:
            requested_divider = divider_match.group(1)

        # Extract text to analyze
        raw_text = completion if isinstance(completion, str) else completion[0]
        answer_text = raw_text
        if not answer_text.strip():
            answer_text = raw_text

        # No divider style specified
        if not requested_divider:
            scores.append(0.5)  # Neutral score
            continue

        # Look for dividers in the text
        lines = answer_text.split('\n')
        found_dividers = []

        for line in lines:
            line = line.strip()
            # Look for repeated character patterns (e.g. "---", "***", "===")
            if line and len(line) >= 3:
                char = line[0]
                if all(c == char for c in line) and len(line) >= 3:
                    found_dividers.append(char)
                # Also match patterns with spaces (e.g. "- - -")
                elif len(line) >= 5 and line[0] == line[2] and all(line[i] == ' ' for i in range(1, len(line), 2)):
                    found_dividers.append(line[0])

        # Score based on divider style match
        if not found_dividers:
            scores.append(0.0)  # No dividers found
        elif requested_divider in found_dividers:
            scores.append(1.0)  # Exact match
        else:
            scores.append(0.3)  # Some divider but not the requested one

    return scores

# --- 5o) Topic Shifts Reward ---
def topic_shifts_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Rewards completions that match the requested topic shift pattern.
    Uses an approach similar to analyze_topic_transitions from writing style extraction.
    """
    scores = []

    for i, completion in enumerate(completions):
        user_prompt = prompts[i][-1] if isinstance(prompts[i], list) else prompts[i]

        # Extract the requested topic shift style from the prompt
        shift_match = re.search(r"\*\*Topic Shifts\*\*:\s*(.*?)(?:\n|$)", user_prompt)
        requested_style = None
        if shift_match:
            shift_text = shift_match.group(1).lower()

            if "dynamic" in shift_text or "highly versatile" in shift_text:
                requested_style = "dynamic"
            elif "regular" in shift_text or "balanced" in shift_text:
                requested_style = "regular"
            elif "moderate" in shift_text or "well-rounded" in shift_text:
                requested_style = "moderate"
            elif "conservative" in shift_text or "cautious" in shift_text:
                requested_style = "conservative"
            elif "consistent" in shift_text or "thorough" in shift_text:
                requested_style = "consistent"

        # Extract text to analyze
        raw_text = completion if isinstance(completion, str) else completion[0]
        answer_text = raw_text
        if not answer_text.strip():
            answer_text = raw_text

        # No topic shift style specified
        if not requested_style:
            scores.append(0.5)  # Neutral score
            continue

        # Analyze topic shifts using paragraph embeddings if spaCy is available
        if nlp:
            # Split into paragraphs
            paragraphs = [p.strip() for p in answer_text.split('\n\n') if p.strip()]
            if len(paragraphs) < 2:
                paragraphs = [s.strip() for s in answer_text.split('\n') if s.strip()]

            if len(paragraphs) < 2:
                scores.append(0.3)  # Not enough paragraphs to analyze
                continue

            # Get embeddings for each paragraph
            paragraph_docs = []
            for p in paragraphs:
                # Limit size for performance
                if len(p) > 1000:
                    p = p[:1000]
                paragraph_docs.append(nlp(p))

            # Calculate semantic similarity between adjacent paragraphs
            similarities = []
            for i in range(len(paragraph_docs) - 1):
                similarity = paragraph_docs[i].similarity(paragraph_docs[i+1])
                similarities.append(similarity)

            # Convert similarities to shift scores (higher similarity = lower shift)
            shift_scores = [1.0 - sim for sim in similarities]

            # Calculate overall statistics
            avg_shift = sum(shift_scores) / len(shift_scores) if shift_scores else 0
            max_shift = max(shift_scores) if shift_scores else 0
        else:
            # Fallback method without spaCy: use sentence-level analysis
            sentences = sent_tokenize(answer_text)
            if len(sentences) < 5:
                scores.append(0.3)  # Not enough sentences
                continue

            # Group sentences into chunks of 2-3
            chunks = [sentences[i:i+3] for i in range(0, len(sentences), 3)]
            if len(chunks) < 2:
                scores.append(0.3)  # Not enough chunks
                continue

            # Calculate "bag of words" for each chunk
            chunk_words = []
            for chunk in chunks:
                chunk_text = " ".join(chunk)
                words = set(word.lower() for word in word_tokenize(chunk_text)
                            if word.isalpha() and word.lower() not in stop_words)
                chunk_words.append(words)

            # Calculate Jaccard similarity between adjacent chunks
            similarities = []
            for i in range(len(chunk_words) - 1):
                intersection = len(chunk_words[i] & chunk_words[i+1])
                union = len(chunk_words[i] | chunk_words[i+1])
                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)

            # Convert similarities to shift scores
            shift_scores = [1.0 - sim for sim in similarities]

            # Calculate statistics
            avg_shift = sum(shift_scores) / len(shift_scores) if shift_scores else 0
            max_shift = max(shift_scores) if shift_scores else 0

        # Score based on requested topic shift style
        if requested_style == "dynamic":
            # Dynamic: high avg_shift and max_shift
            if avg_shift > 0.6 and max_shift > 0.8:
                score = 1.0
            elif avg_shift > 0.5 or max_shift > 0.7:
                score = 0.7
            elif avg_shift > 0.4 or max_shift > 0.6:
                score = 0.4
            else:
                score = 0.2

        elif requested_style == "regular":
            # Regular: moderate avg_shift with some variation
            if 0.4 < avg_shift < 0.6 and max_shift > 0.6:
                score = 1.0
            elif 0.3 < avg_shift < 0.7:
                score = 0.7
            else:
                score = 0.4

        elif requested_style == "moderate":
            # Moderate: modest avg_shift
            if 0.2 < avg_shift < 0.5:
                score = 1.0
            elif 0.1 < avg_shift < 0.6:
                score = 0.7
            else:
                score = 0.4

        elif requested_style == "conservative":
            # Conservative: low avg_shift
            if 0.1 < avg_shift < 0.3:
                score = 1.0
            elif avg_shift <= 0.1 or 0.3 <= avg_shift < 0.4:
                score = 0.7
            else:
                score = 0.4

        else:  # consistent
            # Consistent: very low avg_shift
            if avg_shift < 0.2 and max_shift < 0.3:
                score = 1.0
            elif avg_shift < 0.3:
                score = 0.7
            elif avg_shift < 0.4:
                score = 0.4
            else:
                score = 0.2

        scores.append(score)

    return scores

# --- 5p) Fabrication Detection Reward ---
def fabrication_detection_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Reward function that penalizes fabricated content not present in the prompt.
    Lower scores indicate more fabricated content.
    Optimized for the markdown-formatted prompts generated by the writing style script.
    """
    scores = []

    for i, completion in enumerate(completions):
        user_prompt = prompts[i][-1] if isinstance(prompts[i], list) else prompts[i]

        # Extract completion text
        raw_text = completion if isinstance(completion, str) else completion[0]
        answer_text = raw_text
        if not answer_text.strip():
            answer_text = raw_text

        # Extract prompt content using the new function for markdown format
        prompt_content = extract_prompt_content(user_prompt)

        # Initialize fabrication penalty
        fabrication_penalty = 0.0

        # Check for URLs in completion that weren't in prompt
        urls_in_completion = detect_urls(answer_text)
        urls_in_prompt = detect_urls(prompt_content['full_text'])
        urls_in_key_message = detect_urls(prompt_content['key_message'])

        if urls_in_completion:
            if not urls_in_prompt and not urls_in_key_message:
                # Severe penalty for introducing URLs that weren't in prompt
                fabrication_penalty += 0.7
            else:
                # Check if the URLs are actually the same
                # Sometimes the completion might include a URL that was mentioned in the prompt
                if not any(url_c == url_p for url_c in urls_in_completion for url_p in urls_in_prompt + urls_in_key_message):
                    fabrication_penalty += 0.5

        # Check for people names in completion that weren't in prompt
        names_in_completion = detect_potential_people_names(answer_text)
        names_in_topic = detect_potential_people_names(prompt_content['topic'])
        names_in_key_message = detect_potential_people_names(prompt_content['key_message'])
        names_in_prompt = names_in_topic + names_in_key_message

        if names_in_completion:
            new_names = 0
            for name in names_in_completion:
                # Check if this name or something very similar was in the prompt
                if not any(fuzz.ratio(name.lower(), prompt_name.lower()) > 80 for prompt_name in names_in_prompt):
                    new_names += 1

            # Penalty based on number of new names introduced
            if new_names > 0:
                fabrication_penalty += min(0.5, new_names * 0.1)

        # Check for organization names in completion that weren't in prompt
        orgs_in_completion = detect_organization_names(answer_text)
        orgs_in_topic = detect_organization_names(prompt_content['topic'])
        orgs_in_key_message = detect_organization_names(prompt_content['key_message'])
        orgs_in_prompt = orgs_in_topic + orgs_in_key_message

        if orgs_in_completion:
            new_orgs = 0
            for org in orgs_in_completion:
                # Check if this org or something very similar was in the prompt
                if not any(fuzz.ratio(org.lower(), prompt_org.lower()) > 80 for prompt_org in orgs_in_prompt):
                    new_orgs += 1

            # Penalty based on number of new organizations introduced
            if new_orgs > 0:
                fabrication_penalty += min(0.5, new_orgs * 0.1)

        # Check for common promotional phrases that weren't in prompt
        newsletter_patterns = [
            r"(sign\s*up|subscribe|join).{0,30}(newsletter)",
            r"(register|join).{0,30}(webinar|event)",
            r"link in (bio|profile|comments)",
            r"check out.{0,20}(latest|profile|website)",
            r"follow me",
            r"learn more at"
        ]

        for pattern in newsletter_patterns:
            if (re.search(pattern, answer_text, flags=re.IGNORECASE) and
                not re.search(pattern, prompt_content['topic'], flags=re.IGNORECASE) and
                not re.search(pattern, prompt_content['key_message'], flags=re.IGNORECASE)):
                # Penalty for introducing promotional language not in prompt
                fabrication_penalty += 0.3
                break

        # Check for specific phrases related to contact collection or off-platform actions
        action_patterns = [
            r"link in (bio|comments|description)",
            r"dm me for",
            r"call me at",
            r"email me at",
            r"contact me",
            r"reach out to me",
            r"visit (my|our) website",
            r"download (my|our|the) (app|guide|resources)"
        ]

        for pattern in action_patterns:
            if (re.search(pattern, answer_text, flags=re.IGNORECASE) and
                not re.search(pattern, prompt_content['topic'], flags=re.IGNORECASE) and
                not re.search(pattern, prompt_content['key_message'], flags=re.IGNORECASE)):
                # Higher penalty for action-driving phrases
                fabrication_penalty += 0.4
                break

        # Convert penalty to reward (1.0 = no fabrication, 0.0 = maximum fabrication)
        fabrication_score = max(0.0, 1.0 - fabrication_penalty)
        scores.append(fabrication_score)

    return scores

# --- 5q) Scenario-specific Check Functions ---
def scenario_1_check(prompt: str, completion: str) -> float:
    """
    Enhanced check for scenario 1: Classification of overarching topic.
    Now handles markdown-formatted prompts.
    """
    # First check if it's a markdown prompt
    if "# Request" in prompt or "## Content to Analyze" in prompt:
        analysis_content = extract_analysis_content(prompt)
        text_to_analyze = analysis_content['text_to_analyze']

        # Get length constraints
        length_constraint = analysis_content.get('constraints', {}).get('length', '')
        max_words = 8  # Default

        if "under 10 words" in length_constraint:
            max_words = 10
        elif "under 8 words" in length_constraint:
            max_words = 8
        elif "under 5 words" in length_constraint:
            max_words = 5
    else:
        # Use original approach for non-markdown prompts
        text_to_analyze = prompt
        max_words = 8  # Default

    # Check completion
    lines = completion.strip().split("\n")

    # Must be only 1 line
    if len(lines) > 1:
        return 0.0

    tokens = completion.strip().split()
    # Must be <= max_words
    if len(tokens) > max_words:
        return 0.0

    # Check if it's related to the prompt
    best_score = 0
    for t in tokens[:3]:  # check first few tokens
        score = fuzz.partial_ratio(t.lower(), text_to_analyze.lower())
        if score > best_score:
            best_score = score

    return 1.0 if best_score >= 50 else 0.0

def scenario_2_check(prompt: str, completion: str) -> float:
    """
    Enhanced check for scenario 2: Extract underlying opinion.
    Now handles markdown-formatted prompts.
    """
    # Check if it's a markdown prompt
    if "# Request" in prompt or "## Content to Analyze" in prompt:
        analysis_content = extract_analysis_content(prompt)
        text_to_analyze = analysis_content['text_to_analyze']

        # Extract key concepts using spaCy
        if nlp:
            doc = nlp(text_to_analyze[:5000])
            key_entities = [ent.text.lower() for ent in doc.ents if len(ent.text) > 3]
            key_phrases = []

            # Extract noun phrases or key sentences
            for sentence in doc.sents:
                # Look for sentences with stronger sentiment
                sentiment = sia.polarity_scores(sentence.text)['compound']
                if abs(sentiment) > 0.3:  # Significant sentiment
                    key_phrases.append(sentence.text.lower())

            concepts = key_entities[:5] + key_phrases[:3]  # Use top 5 entities and 3 key phrases
        else:
            # Simple keyword extraction fallback
            words = word_tokenize(text_to_analyze.lower())
            filtered_words = [w for w in words if w not in stop_words and len(w) > 4]
            concepts = sorted(set(filtered_words), key=filtered_words.count, reverse=True)[:8]
    else:
        # Use original approach for non-markdown prompts
        match = re.search(r"[Kk]ey\s+[Cc]oncepts?:\s*(.+)", prompt)
        if match:
            concept_line = match.group(1).strip()
            concepts = [c.strip().lower() for c in concept_line.split(",") if c.strip()]
        else:
            # Extract entities from prompt as fallback
            if nlp:
                prompt_doc = nlp(prompt[:5000])
                concepts = [ent.text.lower() for ent in prompt_doc.ents if len(ent.text) > 3]
            else:
                return 0.5  # No entity extraction, give partial credit

    if not concepts:
        return 0.5  # No concepts to check

    # Check if completion is in first person (for opinion extraction)
    first_person = re.search(r'\b(I|me|my|mine|we|us|our)\b', completion.lower())
    if not first_person:
        return 0.3  # Penalty for not using first person

    # Check concept coverage
    found = 0
    for concept in concepts:
        score = fuzz.partial_ratio(concept, completion.lower())
        if score >= 60:  # 60% similarity threshold
            found += 1

    coverage = found / min(len(concepts), 5)  # Only expect a few concepts
    return coverage

def scenario_3_check(completion: str, prompt: str = None) -> float:
    """
    Enhanced check for scenario 3: Analyze tone.
    Now handles markdown-formatted prompts.
    """
    # Convert completion to lowercase and clean
    completion_clean = completion.strip().lower()

    # If we have a prompt in markdown format, extract the valid tones
    if prompt and ("# Request" in prompt or "## Available Tones" in prompt):
        analysis_content = extract_analysis_content(prompt)
        valid_tones = set(tone.lower() for tone in analysis_content.get('categories', []))

        # If categories empty, fallback to the list in the prompt
        if not valid_tones:
            tones_section = ""
            if "## Available Tones" in prompt:
                tones_section = prompt.split("## Available Tones")[1].split("##")[0]
            else:
                tones_match = re.search(r"Available Tones:(.*?)(?:\n\n|\Z)", prompt, re.DOTALL)
                if tones_match:
                    tones_section = tones_match.group(1)

            if tones_section:
                # Remove any formatting and split on commas, whitespace, or line breaks
                cleaned_section = re.sub(r'[*_]', '', tones_section)
                tones = re.split(r'[,\n\s]+', cleaned_section)
                valid_tones = set(t.strip().lower() for t in tones if t.strip())
    else:
        # Fallback to our predefined list
        valid_tones = VALID_TONES

    # Split on commas and clean
    items = [x.strip().lower() for x in completion_clean.split(",") if x.strip()]

    # Must have 1-3 items
    if not (1 <= len(items) <= 3):
        return 0.0

    # Check against valid tones with fuzzy matching
    valid_items = 0
    for tone in items:
        # Exact match
        if tone in valid_tones:
            valid_items += 1
            continue

        # Fuzzy match
        best_match, score = process.extractOne(tone, valid_tones)
        if score >= 85:  # 85% similarity threshold
            valid_items += 0.8  # Partial credit for similar tones

    # Calculate score based on valid items
    if valid_items == 0:
        return 0.0

    return valid_items / len(items)

def scenario_4_check(completion: str, prompt: str = None) -> float:
    """
    Enhanced check for scenario 4: Classify content structure.
    Now handles markdown-formatted prompts.
    """
    candidate = completion.strip().lower()

    # If we have a prompt in markdown format, extract the valid structures
    if prompt and ("# Request" in prompt or "## Structure Categories" in prompt):
        analysis_content = extract_analysis_content(prompt)
        valid_structures = set(structure.lower() for structure in analysis_content.get('categories', []))

        # If categories empty, fallback to the list in the prompt
        if not valid_structures:
            structure_section = ""
            if "## Structure Categories" in prompt:
                structure_section = prompt.split("## Structure Categories")[1].split("##")[0]
            else:
                structures_match = re.search(r"Structure Categories:(.*?)(?:\n\n|\Z)", prompt, re.DOTALL)
                if structures_match:
                    structure_section = structures_match.group(1)

            if structure_section:
                # Extract structure names from bullet points or sections
                structure_matches = re.findall(r"\*\*(.*?)\*\*", structure_section)
                valid_structures = set(s.strip().lower() for s in structure_matches if s.strip())
    else:
        # Fallback to our predefined list
        valid_structures = VALID_STRUCTURES

    # Exact match
    if candidate in valid_structures:
        return 1.0

    # Fuzzy match
    best_match, score = process.extractOne(candidate, valid_structures)
    if score >= 85:  # 85% similarity threshold
        return 0.7  # Partial credit for similar structure

    return 0.0

# --- 5r) Main Scenario Reward Function ---
def scenario_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Comprehensive scenario reward function that handles all scenarios
    and incorporates all writing style elements.
    """
    print("\n----------Prompt----------\n", prompts[0])
    print("\n----------Completion----------\n", completions[0])
    scores = []

    for i, completion in enumerate(completions):
        user_prompt = prompts[i][-1] if isinstance(prompts[i], list) else prompts[i]

        # Get answer text from completion
        raw_text = completion if isinstance(completion, str) else completion[0]
        answer_text = raw_text

        if not answer_text.strip():
            answer_text = raw_text


        # Extract user content from the prompt
        scenario_id = get_scenario_type(user_prompt)
        scenario_score = 0.0

        if scenario_id == 0:  # LinkedIn post creation
            # Core rewards (essential for all posts)
            bullet_score = bullet_style_reward_func([prompts[i]], [[completion]], **kwargs)[0]
            tone_score = tone_alignment_reward_func([prompts[i]], [completion], **kwargs)[0]
            english_score = chinese_character_reward_func([prompts[i]], [completion], **kwargs)[0]
            hashtag_score = hashtag_limit_reward_func([prompts[i]], [completion], **kwargs)[0]

            # Parse writing style requirements
            style_data = parse_writing_style_block(user_prompt)
            required_length = style_data["post_length_requirement"]
            required_emoji_usage = style_data["emoji_usage_requirement"]

            # Post formatting rewards
            length_score = precise_post_length_reward(required_length, answer_text)
            emoji_score = enhanced_emoji_usage_reward(required_emoji_usage, answer_text)

            # Detailed writing style rewards
            sentence_structure_score = sentence_structure_reward_func([prompts[i]], [completion], **kwargs)[0]
            vocabulary_score = vocabulary_usage_reward_func([prompts[i]], [completion], **kwargs)[0]
            line_break_score = line_break_reward_func([prompts[i]], [completion], **kwargs)[0]
            punctuation_score = punctuation_usage_reward_func([prompts[i]], [completion], **kwargs)[0]
            divider_score = divider_style_reward_func([prompts[i]], [completion], **kwargs)[0]
            topic_shifts_score = topic_shifts_reward_func([prompts[i]], [completion], **kwargs)[0]

            # Narrative quality rewards
            narrative_score = narrative_structure_reward_func([prompts[i]], [completion], **kwargs)[0]
            coherence_score = semantic_coherence_reward([prompts[i]], [completion], **kwargs)[0]

            # Additional quality rewards
            emoji_variety_score = 0
            if required_emoji_usage and required_emoji_usage.lower() not in ["none", "very low"]:
                emoji_variety_score = emoji_variety_reward(answer_text)

            # Safety reward (highest weight)
            fabrication_score = fabrication_detection_reward_func([prompts[i]], [completion], **kwargs)[0]

            # Combine all scores with weighted importance
            # Adjusted weights for LinkedIn posts (scenario 0)
            scenario_score = (
                # Core elements (3.0 total weight)
                (bullet_score * 0.9) * 0.4 +       # 0.36 max
                (tone_score * 0.85) * 0.4 +        # 0.34 max
                english_score * 2.0 +              # 2.0 max (priority #2)
                hashtag_score * 0.3 +              # 0.3 max

                # Formatting (2.5 total)
                length_score * 1.5 +               # 1.5 max (priority #3)
                emoji_score * 1.0 +                # 1.0 max (priority #4)

                # Writing style (1.1 total)
                sentence_structure_score * 0.3 +   # 0.3 max
                line_break_score * 0.3 +           # 0.3 max
                divider_score * 0.2 +              # 0.2 max
                topic_shifts_score * 0.3 +         # 0.3 max

                # Narrative quality (0.6 total)
                narrative_score * 0.3 +            # 0.3 max
                coherence_score * 0.3 +            # 0.3 max

                # Additional quality (0.3 total)
                emoji_variety_score * 0.3 +        # 0.3 max

                # Safety (2.5 total)
                fabrication_score * 2.5            # 2.5 max (priority #1)
            )

            # Apply normalization once
            scenario_score = normalize_scenario_score(scenario_score, 0)

        elif scenario_id == 1:  # Classify overarching topic

            topic_score = scenario_1_check(user_prompt, answer_text)

            scenario_score = topic_score * 10
            scenario_score = normalize_scenario_score(scenario_score, 1)
            print(f"\nScenario: Topic Score: {scenario_score}\n")

        elif scenario_id == 2:  # Extract underlying opinion
            opinion_score = scenario_2_check(user_prompt, answer_text)

            scenario_score = opinion_score * 10
            scenario_score = normalize_scenario_score(scenario_score, 2)
            print(f"\nScenario: Opinion: {scenario_score}\n")

        elif scenario_id == 3:  # Analyze tone
            tone_score = scenario_3_check(answer_text, user_prompt)

            scenario_score = tone_score * 10
            scenario_score = normalize_scenario_score(scenario_score, 3)
            print(f"\nScenario: Tone: {scenario_score}\n")

        elif scenario_id == 4:  # Classify content structure
            structure_score = scenario_4_check(answer_text, user_prompt)

            scenario_score = structure_score * 10
            scenario_score = normalize_scenario_score(scenario_score, 4)
            print(f"\nScenario: Structure: {scenario_score}\n")

        else:  # Generic/unknown scenario
            scenario_score = 1

        # Cap and normalize score
        # total = min(scenario_score, 15.0)  # Higher cap for more rewards
        scores.append(scenario_score)

    return scores

# --- 7) Train the Model ---
def main():
    """Main training function to run the enhanced GRPO training."""
    print("Starting GRPO Training...")
    from training.grpo.model import load_base_model_and_tokenizer

    # Match prior defaults
    model, tokenizer = load_base_model_and_tokenizer(
        model_name="./sft-model",
        max_seq_length=4096,
        lora_rank=64,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=64,
        gpu_memory_utilization=0.6,
    )


    def get_balanced_sft(split="train") -> Dataset:
        # Discover DPO CSV via manifest if run-id provided, else fallback
        csv_path = "dpo.csv"
        try:
            if args.run_id:
                manifest = read_manifest(args.run_id, args.base_dir)
                out_paths = []
                try:
                    stage_meta = manifest.get("stages", {}).get("24-add-negatives")
                    if stage_meta:
                        outs = stage_meta.get("outputs") or stage_meta.get("output") or []
                        out_paths.extend(outs if isinstance(outs, list) else [outs])
                except Exception:
                    pass
                if not out_paths:
                    try:
                        stage_meta = manifest.get("stages", {}).get("23-split")
                        if stage_meta:
                            outs = stage_meta.get("outputs") or stage_meta.get("output") or []
                            out_paths.extend(outs if isinstance(outs, list) else [outs])
                    except Exception:
                        pass
                # Prefer 24-dpo-ready.csv, else 23-dpo.csv, else first discovered
                candidates = [p for p in out_paths if isinstance(p, str)]
                preferred = [p for p in candidates if p.endswith("24-dpo-ready.csv")] or [p for p in candidates if p.endswith("23-dpo.csv")] or candidates
                if preferred:
                    csv_path = preferred[0]
                logger.info(f"Using DPO CSV: {csv_path}")
        except Exception:
            pass

        data = load_dataset("csv", data_files=csv_path)["train"]

        # Filter for rows that have a 'prompt' and 'chosen'
        def filter_samples(sample):
            return sample.get("prompt") is not None and sample.get("chosen") is not None

        data = data.filter(filter_samples)

        def map_to_grpo(sample):
            # We'll build a user prompt
            chat_prompt = [
                {"role": "user", "content": sample["prompt"]},
            ]
            # Flatten into a single string
            combined_string = tokenizer.apply_chat_template(
                chat_prompt, tokenize=False, add_generation_prompt=True
            )

            # Instead of adding <|start_header_id|> etc.
            # we'll just store the "answer" as the raw text (sample["chosen"])
            return {
                "prompt": combined_string,
                "answer": sample['chosen']  # raw text
            }

        data = data.map(map_to_grpo)
        return data


    dataset = get_balanced_sft()
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42, shuffle=True)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    from training.grpo.trainer import build_training_args
    training_args = build_training_args(output_dir="grpo-results")
    # Optional: use aggregator to blend reward signals from multiple simple rewards
    if args.use_aggregator:
        try:
            from training.rewards.aggregator import aggregate_rewards
            # Modularized GRPO rewards (available if aggregator installed)

            from training.rewards.context import RewardContext
            ctx = RewardContext()

            def aggregated_reward(prompts, completions, **kwargs):
                funcs = {
                    "bullet": lambda p, c: rw.bullet_style_reward(p, c, ctx=ctx),
                    "tone": lambda p, c: rw.tone_alignment(p, c, ctx=ctx),
                    "hashtags": lambda p, c: rw.hashtag_limit(p, c, ctx=ctx),
                    "length": lambda p, c: rw.precise_post_length(p, c, ctx=ctx),
                    "emoji": lambda p, c: rw.enhanced_emoji_usage(p, c, ctx=ctx),
                    "structure": lambda p, c: rw.sentence_structure(p, c, ctx=ctx),
                    "coherence": lambda p, c: rw.semantic_coherence(p, c, ctx=ctx),
                }
                import json
                weights = {"bullet": 1.0, "tone": 1.0, "hashtags": 1.0, "length": 1.0, "emoji": 1.0, "structure": 0.5, "coherence": 0.5}
                if args.weights:
                    try:
                        with open(args.weights, "r", encoding="utf-8") as f:
                            user_weights = json.load(f)
                        # Only override known keys to keep safety
                        for k in list(weights.keys()):
                            if k in user_weights:
                                weights[k] = float(user_weights[k])
                    except Exception:
                        pass
                return aggregate_rewards(prompts, completions, funcs, weights)

            reward_list = [aggregated_reward]
        except Exception:
            reward_list = [scenario_reward_func]
    else:
        reward_list = [scenario_reward_func]

    from training.grpo.trainer import build_trainer

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_list,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if test_dataset else None,
    )
    # Write a minimal model card in output_dir
    try:
        import json, time
        card = {
            "run_id": args.run_id,
            "seed": 3407,
            "base_model": getattr(model, 'name_or_path', 'unknown'),
            "trainer": "GRPOTrainer",
            "report_to": "wandb",
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        with open(os.path.join(output_dir, "model_card.json"), "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2)
    except Exception:
        pass


    print("Trainer model:", trainer.model)
    print("Trainer tokenizer:", trainer.tokenizer)

    # Train the model
    print("Starting training...")
    trainer.train()

    wandb.finish()

    # Resolve output model path and idempotent skip
    output_dir = "grpo-model"
    if args.run_id:
        os.makedirs(os.path.join(args.models_dir, args.run_id), exist_ok=True)
        output_dir = os.path.join(args.models_dir, args.run_id, output_dir)
        # Manifest idempotent skip was already checked pre-training if you add it above

    # Save the merged model and tokenizer
    print(f"Saving merged model to {output_dir}...")
    model.save_pretrained_merged(output_dir, tokenizer)
    print(f"Successfully saved merged model to {output_dir}")

    # Update manifest
    try:
        if args.run_id:
            m = read_manifest(args.run_id, args.base_dir)
            sig = compute_hash([], {"stage": 26})
            update_stage(args.run_id, args.base_dir, m, "26-train-grpo", input_path=None, outputs=[output_dir], signature=sig, extra={})
    except Exception:
        pass

    return trainer

if __name__ == "__main__":
    main()