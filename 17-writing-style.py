import json
import re
import nltk
import spacy
import emojis
import warnings
import time
import os
from datetime import datetime
from tqdm import tqdm  # For progress bars (install with pip if needed)
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertModel
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
from spacy.matcher import Matcher, PhraseMatcher
import multiprocessing as mp
from functools import partial
import argparse

# Configure performance settings here
PERFORMANCE_CONFIG = {
    'batch_size': 5000,                # Process this many posts at once
    'max_posts_per_author': 20,        # Limit posts per author for phrase analysis
    'min_posts_for_hybrid': 5,         # Only use hybrid approach for authors with 5+ posts
    'sample_percent': 100,             # Process only this percentage of posts (1-100)
    'use_semantic_similarity': False,  # Disable for massive speedup
    'use_bert_for_topics': False,      # Disable for massive speedup
    'use_parallel_processing': True,   # Enable multiprocessing by default (mp-safe bullet detection applied)
    'max_workers': mp.cpu_count() - 1, # Number of parallel workers
    'author_phrases_only': False,      # Set to True to only extract phrases and skip other analyses
    'output_interval': 10000,          # Write results every N posts
    'use_small_spacy': True           # Use smaller spaCy model
}

# Compatibility bootstrap: allow running this file directly or as part of the package
if __package__ is None or __package__ == "":
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from utils.logging_setup import init_pipeline_logging
from utils.manifest import read_manifest, update_stage, compute_hash, should_skip
from utils.cli import add_standard_args, resolve_common_args
from utils.io import resolve_input_path

logger = init_pipeline_logging("phase2.style", None, "17-writing-style")
logger.info(f"Performance configuration: {PERFORMANCE_CONFIG}")
logger.info(f"Available CPUs: {mp.cpu_count()}")

# Start timing
start_time = time.time()

# Optional imports for enhanced functionality
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True and PERFORMANCE_CONFIG['use_semantic_similarity']
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence-transformers not installed. Semantic similarity analysis will be limited.")

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_CLUSTERING_AVAILABLE = True and PERFORMANCE_CONFIG['use_semantic_similarity']
except ImportError:
    SKLEARN_CLUSTERING_AVAILABLE = False
    warnings.warn("scikit-learn DBSCAN not available. Clustering capabilities will be limited.")

# Load the spaCy English model (use small model for speed)
if PERFORMANCE_CONFIG['use_small_spacy']:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Loaded spaCy small model for better performance")
else:
    try:
        nlp = spacy.load("en_core_web_lg")
        logger.info("Loaded spaCy large model")
    except:
        try:
            nlp = spacy.load("en_core_web_md")
            logger.info("Loaded spaCy medium model")
        except:
            nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy small model")

# Configure spaCy pipeline for speed
if PERFORMANCE_CONFIG['use_small_spacy']:
    # Disable components we don't need
    disabled_pipes = []
    for pipe in ['tagger', 'parser', 'attribute_ruler', 'lemmatizer']:
        if pipe in nlp.pipe_names:
            disabled_pipes.append(pipe)
    disabled_pipes = set(disabled_pipes)
    logger.info(f"Disabling unused spaCy pipes: {disabled_pipes}")
    nlp.disable_pipes(*disabled_pipes)

# Build a shared FeatureContext once
try:
    from features.context import FeatureContext as _FeatureContext
    CTX = _FeatureContext.from_spacy(nlp)
except Exception:
    CTX = None

# NLTK resources are expected to be installed via scripts/setup_nlp.py
# Keeping this callable for local convenience; set environment to skip in CI
if os.environ.get("PIPE_DOWNLOAD_NLTK", "0") == "1":
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
    except Exception:
        pass

# Global singletons for performance
try:
    STOPWORDS_EN = set(stopwords.words('english'))
except Exception:
    STOPWORDS_EN = set()

try:
    SIA = SentimentIntensityAnalyzer()
except Exception:
    SIA = None

# Precompiled regexes
BUL_NUMBERED_RE = re.compile(r'^\s*\d+[\.\)]\s+')
BUL_LETTERED_RE = re.compile(r'^\s*[a-zA-Z]+[\.\)]\s+')
BUL_SYMBOLIC_RE = re.compile(r'^\s*([^\w\s])')
BUL_INDENT_RE = re.compile(r'^ {4,}([^\w\s])')
DIVIDER_RE = re.compile(r'^\s*([^\w\s])\1{3,}\s*$')

###################################################
#              BERT Model Setup                   #
###################################################
if PERFORMANCE_CONFIG['use_bert_for_topics']:
    print("Loading BERT model...")
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()  # Set to eval mode for inference
else:
    print("Skipping BERT model loading for performance")
    tokenizer = None
    model = None

###################################################
#             Analysis Functions                  #
###################################################

# moved to features/text_stats.py

def get_action_to_descriptive_ratio(text):
    try:
        doc = nlp(text[:10000])  # Limit text length for performance
        action_verbs = len([token for token in doc if token.pos_ == "VERB"])
        adjectives = len([token for token in doc if token.pos_ == "ADJ"])
        return (action_verbs / adjectives) if adjectives > 0 else 0
    except Exception as e:
        return 0

def get_new_entity_rate(text):
    try:
        doc = nlp(text[:10000])  # Limit text length for performance
        total_entities = len(doc.ents)
        unique_entities = len(set(ent.text for ent in doc.ents))
        if total_entities == 0:
            return 0
        return unique_entities / total_entities
    except Exception as e:
        return 0

def analyze_topic_transitions(text, tokenizer, model, min_segment_length=50, min_text_length=300):
    # Skip if BERT is disabled
    if not PERFORMANCE_CONFIG['use_bert_for_topics']:
        return []

    if len(text) < min_text_length:
        return []
    raw_sections = text.split('\n')
    segments = []
    current_segment = ""
    for section in raw_sections:
        candidate = (current_segment + " " + section).strip() if current_segment else section
        # Accumulate until we exceed min_segment_length
        if len(candidate) < min_segment_length:
            current_segment = candidate
        else:
            if current_segment.strip():
                segments.append(current_segment.strip())
            current_segment = section
    if current_segment.strip():
        segments.append(current_segment.strip())
    if len(segments) < 2:
        return []

    def get_bert_embedding(text_chunk):
        inputs = tokenizer(text_chunk, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[0, 0, :]
        return cls_emb.numpy()

    segment_embeddings = [get_bert_embedding(seg) for seg in segments]
    transitions = []
    for i in range(len(segment_embeddings) - 1):
        emb1 = segment_embeddings[i].reshape(1, -1)
        emb2 = segment_embeddings[i+1].reshape(1, -1)
        sim = cosine_similarity(emb1, emb2)[0][0]
        shift_score = float(1.0 - sim)
        transitions.append({
            "from_segment": i,
            "to_segment": i+1,
            "shift_score": shift_score
        })
    return transitions

def analyze_line_breaks(text):
    line_breaks = text.count('\n')
    lines = text.split('\n')
    avg_line_breaks = sum(len(line) == 0 for line in lines) / (len(lines) - 1) if len(lines) > 1 else 0
    return line_breaks, avg_line_breaks

from features.bullets import detect_bullet_styles as _mod_detect_bullets
from features.dividers import detect_divider_styles as _mod_detect_dividers
from features.profanity import determine_profanity_category as _mod_profanity
from features.text_stats import analyze_vocabulary_usage as _mod_vocab, analyze_sentence_structure as _mod_sent_struct, analyze_line_breaks as _mod_line_breaks, punctuation_counts as _mod_punct

def detect_bullet_styles(text):
    lines = text.split('\n')
    bullet_list = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if BUL_NUMBERED_RE.match(line):
            bullet_list.append("Numbered")
            continue
        if BUL_LETTERED_RE.match(line):
            bullet_list.append("Lettered")
            continue
        indent_bullet = BUL_INDENT_RE.match(line)
        if indent_bullet:
            bullet_list.append(indent_bullet.group(1))
            continue
        sym_match = BUL_SYMBOLIC_RE.match(line)
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

def rolling_average(values, window=3):
    if len(values) < window:
        return values
    return [np.mean(values[i:i+window]) for i in range(len(values) - window + 1)]

def analyze_narrative_flow(text, window_size: int = 3):
    """Backup behavior: Intro + Content labels, computed over sentences."""
    sentences = sent_tokenize(text[:5000])  # Limit for performance
    if not sentences:
        return []
    # Compute sentiment smoothing as in backup (not used for labeling here)
    sia = SentimentIntensityAnalyzer()
    raw_scores = [sia.polarity_scores(s)['compound'] for s in sentences]
    _ = rolling_average(raw_scores, window=window_size)
    # Simplified labels for speed (backup): first Intro, rest Content
    return ["Introduction/Setup"] + ["Content"] * (len(sentences) - 1)

def analyze_pacing(text):
    """Backup behavior: mean sentence length thresholds."""
    sentences = sent_tokenize(text[:5000])  # Limit for performance
    if len(sentences) < 3:
        return "Short/Not Enough Data"
    sentence_lengths = [len(word_tokenize(s)) for s in sentences]
    avg_len = np.mean(sentence_lengths)
    if avg_len < 10:
        return "Fast"
    elif avg_len > 20:
        return "Slow"
    else:
        return "Moderate"

def get_sentiment_scores(text):
    """Backup behavior: sample up to 10 sentences and compute VADER compound scores."""
    sentences = sent_tokenize(text[:5000])  # Limit text length
    if len(sentences) > 10:
        # Sample at most 10 sentences for long texts
        indices = sorted(random.sample(range(len(sentences)), 10))
        sentences = [sentences[i] for i in indices]
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = []
    for sentence in sentences:
        score = sia.polarity_scores(sentence)['compound']
        sentiment_scores.append(score)
    return sentiment_scores

def analyze_sentiment_arc(sentiment_scores, window_size: int = 3, short_text_threshold: int = 3):
    """Backup behavior: classify by average sentiment."""
    if len(sentiment_scores) < short_text_threshold:
        return "Short/Not Enough Data for Arc"
    avg_score = np.mean(sentiment_scores)
    if avg_score > 0.2:
        return "Positive"
    elif avg_score < -0.2:
        return "Negative"
    else:
        return "Neutral"

def analyze_narrative_structure(text):
    flow = analyze_narrative_flow(text)
    pacing = analyze_pacing(text)
    sentiment_scores = get_sentiment_scores(text)
    sentiment_arc = analyze_sentiment_arc(sentiment_scores)
    return flow, pacing, sentiment_arc

def detect_divider_styles(text):
    lines = text.split('\n')
    dividers = []
    for line in lines:
        match = DIVIDER_RE.match(line)
        if match:
            dividers.append(match.group(1))
    divider_counts = Counter(dividers)
    return divider_counts.most_common(1)[0][0] if divider_counts else None

def determine_profanity_category(text):
    categorized_words = {
        "apeshit": "moderate",
        "arsehole": "light",
        "ass": "light",
        "asshole": "light",
        "bastard": "moderate",
        "bullshit": "moderate",
        "bitch": "moderate",
        "clusterfuck": "heavy",
        "damn": "moderate",
        "damnit": "moderate",
        "fuck": "heavy",
        "fucker": "heavy",
        "fuckin": "heavy",
        "fucking": "heavy",
        "goddamn": "heavy",
        "bollocks": "light",
        "hell": "light",
        "holy shit": "moderate",
        "horseshit": "moderate",
        "motherfucker": "heavy",
        "mother fucker": "heavy",
        "piss": "light",
        "pissed": "light",
        "shit": "moderate",
    }
    words_in_text = text.lower().split()
    highest_severity = "none"
    severity_mapping = {"none": 0, "light": 1, "moderate": 2, "heavy": 3}

    # Check first 1000 words for performance
    for word in words_in_text[:1000]:
        category = categorized_words.get(word, "none")
        if severity_mapping[category] > severity_mapping[highest_severity]:
            highest_severity = category
    return highest_severity

###################################################
#      DISTINCTIVE PHRASE DETECTION              #
###################################################

def similar(phrase1, phrase2):
    """Check if two phrases are similar based on character overlap"""
    # If one is a subset of the other
    if phrase1 in phrase2 or phrase2 in phrase1:
        return True

    # Calculate overlap
    min_len = min(len(phrase1), len(phrase2))
    if min_len < 10:  # Very short phrases
        return False

    # Count matching characters
    matches = sum(1 for a, b in zip(phrase1, phrase2) if a == b)
    match_ratio = matches / min_len

    return match_ratio > 0.8  # 80% similarity threshold

def get_author_distinctive_phrases_hybrid(author_texts, max_phrases=15, min_phrase_freq=2):
    """
    Hybrid approach combining entity recognition and semantic similarity
    for identifying distinctive author phrases - optimized for performance.
    """
    # Limit number of texts per author for performance
    if len(author_texts) > PERFORMANCE_CONFIG['max_posts_per_author']:
        # Use the longest posts as they likely contain more distinctive phrases
        author_texts = sorted(author_texts, key=len, reverse=True)[:PERFORMANCE_CONFIG['max_posts_per_author']]

    # Dictionary to store our results from different methods
    results = {
        'entities': [],      # From spaCy NER
        'patterns': [],      # From custom pattern matching
        'semantic': [],      # From transformer embeddings
        'position': []       # From positional analysis (first/last sentences)
    }

    # Skip processing if no texts
    if not author_texts:
        return []

    ###############################
    # PART 1: ENTITY RECOGNITION
    ###############################
    try:
        # Process all author texts with spaCy (limit text size for performance)
        # Build processed texts and batch with spaCy for speed
        processed_texts = []
        for text in author_texts:
            if len(text) > 5000:
                processed_texts.append(text[:2000] + " " + text[-2000:])
            else:
                processed_texts.append(text)
        spacy_docs = list(nlp.pipe(
            processed_texts,
            n_process=PERFORMANCE_CONFIG['max_workers'],
            batch_size=32
        ))

        # 1.1 Extract named entities
        entity_counter = Counter()
        for doc in spacy_docs:
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "PERSON", "GPE"]:
                    if 2 <= len(ent.text.split()) <= 5:
                        entity_counter[ent.text.lower()] += 1

        # Get recurring entities (appear in multiple texts)
        for entity, count in entity_counter.items():
            if count >= min_phrase_freq and len(entity) > 3:
                # Calculate frequency as percentage of author's texts
                freq = count / len(author_texts)
                results['entities'].append({
                    'phrase': entity,
                    'score': freq * 100,  # Higher weight for named entities
                    'type': 'entity'
                })

        # 1.2 Match custom patterns for greetings and sign-offs
        # Create pattern matchers
        matcher = Matcher(nlp.vocab)

        # Common greeting patterns
        greeting_patterns = [
            [{"LOWER": "hey"}, {"LOWER": "everyone"}],
            [{"LOWER": "hello"}, {"LOWER": "folks"}],
            [{"LOWER": "hi"}, {"LOWER": "guys"}],
            [{"LOWER": "what's"}, {"LOWER": "up"}],
            [{"LOWER": "welcome"}, {"LOWER": "back"}],
            [{"LOWER": "welcome"}, {"LOWER": "to"}, {"LOWER": "my"}],
            [{"LOWER": "welcome"}, {"LOWER": "to"}, {"LOWER": "another"}],
        ]

        # Common sign-off patterns
        signoff_patterns = [
            [{"LOWER": "thanks"}, {"LOWER": "for"}, {"LOWER": "watching"}],
            [{"LOWER": "see"}, {"LOWER": "you"}, {"LOWER": "next"}, {"LOWER": "time"}],
            [{"LOWER": "don't"}, {"LOWER": "forget"}, {"LOWER": "to"}, {"LOWER": "subscribe"}],
            [{"LOWER": "don't"}, {"LOWER": "forget"}, {"LOWER": "to"}, {"LOWER": "like"}],
            [{"LOWER": "until"}, {"LOWER": "next"}, {"LOWER": "time"}],
            [{"LOWER": "like"}, {"LOWER": "and"}, {"LOWER": "subscribe"}],
            [{"LOWER": "hit"}, {"LOWER": "that"}, {"LOWER": "like"}, {"LOWER": "button"}],
        ]

        # Add channel/personal branding patterns
        branding_patterns = [
            [{"LOWER": "check"}, {"LOWER": "out"}, {"LOWER": "my"}],
            [{"LOWER": "follow"}, {"LOWER": "me"}, {"LOWER": "on"}],
            [{"LOWER": "link"}, {"LOWER": "in"}, {"LOWER": "the"}, {"LOWER": "description"}],
            [{"LOWER": "join"}, {"LOWER": "my"}],
            [{"LOWER": "on"}, {"LOWER": "my"}, {"LOWER": "channel"}],
            [{"TEXT": {"REGEX": "@[a-zA-Z0-9_]+"}},]  # Match @username
        ]

        # Add patterns to matcher
        for i, pattern in enumerate(greeting_patterns):
            matcher.add(f"GREETING_{i}", [pattern])
        for i, pattern in enumerate(signoff_patterns):
            matcher.add(f"SIGNOFF_{i}", [pattern])
        for i, pattern in enumerate(branding_patterns):
            matcher.add(f"BRANDING_{i}", [pattern])

        # Apply matcher to documents
        pattern_counter = Counter()
        for doc in spacy_docs:
            matches = matcher(doc)
            for _, start, end in matches:
                pattern_text = doc[start:end].text.lower()
                pattern_counter[pattern_text] += 1

        # Extract recurring patterns
        for pattern, count in pattern_counter.items():
            if count >= min_phrase_freq:
                pattern_type = "unknown"
                if any(p in pattern for p in ["welcome", "hey", "hello", "hi", "what's up"]):
                    pattern_type = "greeting"
                elif any(p in pattern for p in ["thanks", "thank you", "see you", "next time", "subscribe"]):
                    pattern_type = "signoff"
                elif any(p in pattern for p in ["@", "channel", "follow me", "check out"]):
                    pattern_type = "branding"

                results['patterns'].append({
                    'phrase': pattern,
                    'score': (count / len(author_texts)) * 80,  # High weight for patterns
                    'type': pattern_type
                })

        # Skip verb phrase extraction for performance

    except Exception as e:
        warnings.warn(f"Error in entity recognition: {e}")

    ###############################
    # PART 2: SEMANTIC SIMILARITY - SKIP FOR PERFORMANCE
    ###############################
    # PART 3: POSITIONAL ANALYSIS
    ###############################
    # Simple approach to extract first/last sentence patterns
    first_sentences = []
    last_sentences = []

    for text in author_texts:
        sentences = sent_tokenize(text)
        if sentences:
            # Get first sentence words (up to 8)
            first_words = sentences[0].split()[:8]
            if len(first_words) >= 3:
                first_sentences.append(' '.join(first_words).lower())

            # Get last sentence words (up to 8)
            if len(sentences) > 1:
                last_words = sentences[-1].split()[-8:]
                if len(last_words) >= 3:
                    last_sentences.append(' '.join(last_words).lower())

    # Find common beginnings/endings
    first_counter = Counter(first_sentences)
    last_counter = Counter(last_sentences)

    for phrase, count in first_counter.items():
        if count >= min_phrase_freq:
            results['position'].append({
                'phrase': phrase,
                'score': (count / len(author_texts)) * 90,  # High weight for position
                'type': 'first_sentence'
            })

    for phrase, count in last_counter.items():
        if count >= min_phrase_freq:
            results['position'].append({
                'phrase': phrase,
                'score': (count / len(author_texts)) * 90,
                'type': 'last_sentence'
            })

    ###############################
    # PART 4: COMBINE RESULTS
    ###############################
    # Flatten all results into a single list
    all_results = []
    for _, items in results.items():
        all_results.extend(items)

    # Sort by score (descending)
    all_results.sort(key=lambda x: x['score'], reverse=True)

    # Remove duplicates or nearly identical phrases
    final_results = []
    seen_phrases = set()

    for result in all_results:
        phrase = result['phrase']
        phrase_key = ''.join(phrase.split()).lower()  # Normalize for comparison

        # Skip if we've seen this or very similar phrase
        if any(similar(phrase_key, seen) for seen in seen_phrases):
            continue

        final_results.append(result)
        seen_phrases.add(phrase_key)

        # Stop when we have enough phrases
        if len(final_results) >= max_phrases:
            break

    # Format final results
    formatted_results = []
    for result in final_results:
        # Add prefix for certain types
        phrase = result['phrase']
        if result['type'] == 'greeting_phrase' or result['type'] == 'first_sentence':
            formatted_results.append(f"greeting: {phrase}")
        elif result['type'] == 'signoff_phrase' or result['type'] == 'last_sentence':
            formatted_results.append(f"sign-off: {phrase}")
        elif result['type'] == 'entity' or result['type'] == 'branding':
            formatted_results.append(f"brand: {phrase}")
        else:
            formatted_results.append(phrase)

    return formatted_results[:max_phrases]

# Fallback method if advanced approach fails - optimized for speed
def get_uncommon_regular_words(texts, common_word_threshold=1000, stopwords_en: set | None = None):
    """
    Simple fallback method to identify uncommon but regularly used words by an author.
    """
    # Limit the amount of text we process
    sampled_texts = texts
    if len(texts) > 5:
        sampled_texts = texts[:5]

    # Join with space to split properly
    all_text = ' '.join(sampled_texts)

    # Limit text size for performance
    if len(all_text) > 20000:
        all_text = all_text[:20000]

    words = re.findall(r'\b\w+\b', all_text.lower())
    word_freq = Counter(words)

    # Filter common words faster
    stop_words = stopwords_en if stopwords_en is not None else STOPWORDS_EN or set(stopwords.words('english'))
    filtered_words = {
        w for w in word_freq
        if word_freq[w] < common_word_threshold and w not in stop_words and len(w) > 3
    }

    # Simplified frequency check
    regular_words = [w for w in filtered_words if word_freq[w] >= 2]

    return regular_words[:15]  # Return at most 15 words

###################################################
#            OPTIMIZED PROCESSING                #
###################################################

def analyze_post(post, author_common_phrases):
    """Process a single post with all analyses"""
    if post.get('is_repost') or not post.get('post_text'):
        # Skip analysis for reposts or empty text
        return post

    text = post['post_text']

    if PERFORMANCE_CONFIG['author_phrases_only']:
        # Only attach common phrases, skip other analyses
        author = post.get('author_name', 'unknown')
        post['common_phrases'] = author_common_phrases.get(author, [])
        return post

    # Run style analyses
    post['sentence_structure'] = _mod_sent_struct(text)
    post['vocabulary_usage'] = _mod_vocab(text)
    post['line_breaks'], post['avg_line_breaks'] = _mod_line_breaks(text)
    post['punctuation_usage'] = _mod_punct(text)
    if PERFORMANCE_CONFIG['use_parallel_processing']:
        post['divider_style'] = detect_divider_styles(text)
    else:
        post['divider_style'] = _mod_detect_dividers(text, CTX) if CTX else detect_divider_styles(text)
    # Use mp-safe bullet detection when multiprocessing to avoid pickling CTX
    if PERFORMANCE_CONFIG['use_parallel_processing']:
        post['bullet_styles'] = detect_bullet_styles(text)
    else:
        post['bullet_styles'] = _mod_detect_bullets(text, CTX) if CTX else detect_bullet_styles(text)

    # Skip expensive BERT analysis if disabled
    if PERFORMANCE_CONFIG['use_bert_for_topics']:
        post['topic_shifts'] = analyze_topic_transitions(text, tokenizer, model)
    else:
        post['topic_shifts'] = []

    # Narrative + sentiment (aligned to backup behavior)
    flow, pacing, arc = analyze_narrative_structure(text)
    post['flow'], post['pacing'], post['sentiment_arc'] = flow, pacing, arc
    post['profanity'] = _mod_profanity(text)

    # Attach the common_phrases for the author
    author = post.get('author_name', 'unknown')
    post['common_phrases'] = author_common_phrases.get(author, [])

    return post

def process_post_batch(posts, author_common_phrases):
    """Process a batch of posts and return the analyzed results"""
    if PERFORMANCE_CONFIG['use_parallel_processing']:
        # Use multiprocessing
        with mp.Pool(PERFORMANCE_CONFIG['max_workers']) as pool:
            func = partial(analyze_post, author_common_phrases=author_common_phrases)
            analyzed_posts = pool.map(func, posts)
    else:
        # Process posts sequentially
        analyzed_posts = []
        for post in tqdm(posts, desc="Analyzing posts"):
            analyzed_posts.append(analyze_post(post, author_common_phrases))

    return analyzed_posts

def main(input_file: str,
         output_file: str,
         temp_output_file: str,
         run_id=None,
         base_dir: str = "data/processed",
         seed: int | None = None):

    # Seed for determinism where applicable
    from utils.seed import set_global_seed
    set_global_seed(seed)

    # Prepare standardized outputs via centralized resolver
    from utils.io import resolve_io
    from utils.artifacts import ArtifactNames
    std_output_path = None
    std_temp_output_path = None
    std_outfile = None
    std_temp_outfile = None
    if run_id:
        # Accept outputs from 15-clean-context, 12-clean-opinions, or 11-extract-opinion in that order
        input_file, std_output_path, run_id = resolve_io(stage="17-writing-style", run_id=run_id, base_dir=base_dir, explicit_in=input_file, prior_stage=["15-clean-context", "12-clean-opinions", "11-extract-opinion"], std_name=ArtifactNames.STAGE17_STYLE)
        std_temp_output_path = os.path.join(base_dir, run_id, ArtifactNames.STAGE17_PARTIAL)

        # If up-to-date, skip work (signature on input + PERF config)
        from utils.version import STAGE_VERSION
        signature = compute_hash([input_file], config={"stage": 17, "perf": PERFORMANCE_CONFIG, "stage_version": STAGE_VERSION})
        manifest = read_manifest(run_id, base_dir)
        if should_skip(manifest, "17-writing-style", signature, [std_output_path, std_temp_output_path]):
            logger.info(f"Skipping 17-writing-style; up-to-date at {os.path.join(base_dir, run_id)}")
            return

        std_outfile = open(std_output_path, 'w')
        std_temp_outfile = open(std_temp_output_path, 'w')

    print(f"Processing {input_file} -> {output_file}")
    print(f"Temporary results will be saved to {temp_output_file}")
    if std_output_path:
        print(f"Also writing standardized artifacts to: {std_output_path} and {std_temp_output_path}")

    # Count total posts
    total_posts = 0
    with open(input_file, 'r') as infile:
        for _ in infile:
            total_posts += 1

    logger.info(f"Total posts to process: {total_posts}")

    # Sample if needed
    process_posts = total_posts
    if PERFORMANCE_CONFIG['sample_percent'] < 100:
        process_posts = int(total_posts * PERFORMANCE_CONFIG['sample_percent'] / 100)
        logger.info(f"Will process a {PERFORMANCE_CONFIG['sample_percent']}% sample: {process_posts} posts")

    # 1) First pass: Collect author texts for phrase analysis
    logger.info("First pass: Collecting texts by author...")
    author_texts = defaultdict(list)
    posts_processed = 0

    with open(input_file, 'r') as infile:
        for line in tqdm(infile, total=total_posts, desc="Collecting author texts"):
            posts_processed += 1

            # Apply sampling
            if PERFORMANCE_CONFIG['sample_percent'] < 100:
                if random.random() * 100 > PERFORMANCE_CONFIG['sample_percent']:
                    continue

            post = json.loads(line)
            if post.get('is_repost') or not post.get('post_text'):
                continue
            author = post.get('author_name', 'unknown')

            # Only keep up to N longest posts per author for the first pass
            texts = author_texts[author]
            pt = post['post_text']
            if len(texts) < PERFORMANCE_CONFIG['max_posts_per_author']:
                texts.append(pt)
            else:
                # Replace a shorter one if current is longer
                min_idx, min_len = min(((i, len(t)) for i, t in enumerate(texts)), key=lambda x: x[1])
                if len(pt) > min_len:
                    texts[min_idx] = pt

    # 2) Compute distinctive phrases per author
    logger.info(f"Analyzing distinctive phrases for {len(author_texts)} authors...")
    author_common_phrases = {}

    # Group authors by post count for efficiency
    authors_by_count = defaultdict(list)
    for author, texts in author_texts.items():
        authors_by_count[len(texts)].append(author)

    # Process authors, starting with those who have more posts
    for post_count in sorted(authors_by_count.keys(), reverse=True):
        authors = authors_by_count[post_count]
        logger.info(f"Processing {len(authors)} authors with {post_count} posts each")

        for author in tqdm(authors, desc=f"Authors with {post_count} posts"):
            texts = author_texts[author]

            if post_count >= PERFORMANCE_CONFIG['min_posts_for_hybrid']:
                try:
                    # Use hybrid approach for authors with enough posts
                    phrases = get_author_distinctive_phrases_hybrid(
                        texts,
                        all_authors_texts=None,  # Skip cross-author comparison for speed
                        max_phrases=15
                    )
                    author_common_phrases[author] = phrases
                except Exception as e:
                    print(f"Error analyzing phrases for {author}: {e}")
                    # Fallback to simpler approach if hybrid fails
                    author_common_phrases[author] = get_uncommon_regular_words(texts)
            else:
                # Use simpler approach for authors with few texts
                author_common_phrases[author] = get_uncommon_regular_words(texts)

    # Free up memory
    del author_texts

    # 3) Process posts in batches
    logger.info(f"Second pass: Processing posts in batches of {PERFORMANCE_CONFIG['batch_size']}...")

    posts_processed = 0
    current_batch = []

    # Open both output files
    with open(output_file, 'w') as outfile, open(temp_output_file, 'w') as temp_outfile, open(input_file, 'r') as infile:
        for line in tqdm(infile, total=total_posts, desc="Processing posts"):
            posts_processed += 1

            # Apply sampling
            if PERFORMANCE_CONFIG['sample_percent'] < 100:
                if random.random() * 100 > PERFORMANCE_CONFIG['sample_percent']:
                    continue

            try:
                post = json.loads(line)
                current_batch.append(post)

                # Process batch when it reaches the batch size
                if len(current_batch) >= PERFORMANCE_CONFIG['batch_size']:
                    processed_batch = process_post_batch(current_batch, author_common_phrases)

                    # Write results
                    for processed_post in processed_batch:
                        serialized = json.dumps(processed_post)
                        outfile.write(serialized + '\n')
                        temp_outfile.write(serialized + '\n')
                        if std_outfile:
                            std_outfile.write(serialized + '\n')
                        if std_temp_outfile:
                            std_temp_outfile.write(serialized + '\n')

                    # Clear batch
                    current_batch = []

                    # Report progress
                    elapsed = time.time() - start_time
                    posts_per_second = posts_processed / elapsed
                    estimated_total = elapsed * (total_posts / posts_processed)
                    remaining = estimated_total - elapsed

                    logger.info(
                        f"Processed {posts_processed}/{total_posts} posts "
                        f"({posts_processed/total_posts*100:.1f}%) - "
                        f"Speed: {posts_per_second:.1f} posts/sec - "
                        f"Elapsed: {elapsed/60:.1f} min - "
                        f"Remaining: {remaining/60:.1f} min"
                    )

            except Exception as e:
                logger.warning(f"Error processing line {posts_processed}: {e}")
                # Continue processing other posts

        # Process any remaining posts
        if current_batch:
            processed_batch = process_post_batch(current_batch, author_common_phrases)
            for processed_post in processed_batch:
                serialized = json.dumps(processed_post)
                outfile.write(serialized + '\n')
                temp_outfile.write(serialized + '\n')
                if std_outfile:
                    std_outfile.write(serialized + '\n')
                if std_temp_outfile:
                    std_temp_outfile.write(serialized + '\n')

    # Close standardized outputs if opened
    if std_outfile:
        std_outfile.close()
    if std_temp_outfile:
        std_temp_outfile.close()


    # Validate standardized JSONL before updating manifest
    try:
        from schemas import Stage17Record
        from utils.validation import validate_jsonl_records
        ok_std = True
        if std_output_path:
            ok_std = validate_jsonl_records(std_output_path, model_cls=Stage17Record, required_keys=["post_text"])  # just basic presence
        if not ok_std:
            logger.error("Stage17 standardized JSONL failed validation; skipping manifest update")
            return
    except Exception:
        pass

    # Report completion
    elapsed = time.time() - start_time
    logger.info(f"Classification completed in {elapsed/60:.2f} minutes.")
    logger.info(f"Processed {posts_processed} posts at a rate of {posts_processed/elapsed:.1f} posts/second.")
    logger.info(f"Results written to {output_file}")
    if std_output_path and run_id:
        manifest = read_manifest(run_id, base_dir)
        from utils.version import STAGE_VERSION
        signature = compute_hash([input_file], config={"stage": 17, "perf": PERFORMANCE_CONFIG, "stage_version": STAGE_VERSION})
        update_stage(
            run_id,
            base_dir,
            manifest,
            stage_name="17-writing-style",
            input_path=input_file,
            outputs=[p for p in [std_output_path, std_temp_output_path] if p],
            signature=signature,
            extra={"processed": posts_processed},
        )
        logger.info(f"Standardized results also written to {std_output_path} and {std_temp_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract writing-style features (Phase 1 compat CLI)")
    parser.add_argument("--input", dest="input_path", default=None)
    parser.add_argument("--output", dest="output_path", default="step-17-posts-with-writing-style.jsonl")
    parser.add_argument("--temp-output", dest="temp_output_path", default="step-17-partial-results.jsonl")
    add_standard_args(parser, include_seed=True)
    args = parser.parse_args()

    args = resolve_common_args(args, require_input_when_no_run_id=True)

    main(
        input_file=args.input_path,
        output_file=args.output_path,
        temp_output_file=args.temp_output_path,
        run_id=args.run_id,
        base_dir=args.base_dir,
        seed=args.seed,
    )