from __future__ import annotations

from collections import Counter
from typing import List
from nltk.tokenize import sent_tokenize

from .context import FeatureContext


def similar(phrase1: str, phrase2: str) -> bool:
    """Check if two phrases are similar based on character overlap/subset."""
    if phrase1 in phrase2 or phrase2 in phrase1:
        return True
    min_len = min(len(phrase1), len(phrase2))
    if min_len < 10:
        return False
    matches = sum(1 for a, b in zip(phrase1, phrase2) if a == b)
    match_ratio = matches / min_len
    return match_ratio > 0.8


def get_author_distinctive_phrases_hybrid(ctx: FeatureContext, author_texts: List[str], max_phrases: int = 15, min_phrase_freq: int = 2) -> List[str]:
    # Limit texts per author handled by caller; ensure sentence segmentation available
    if not author_texts:
        return []

    # Trim long texts to intro/outro zones
    processed_texts: List[str] = [t[:2000] + " " + t[-2000:] if len(t) > 5000 else t for t in author_texts]
    spacy_docs = list(ctx.nlp.pipe(processed_texts, n_process=1, batch_size=32))

    results = {"entities": [], "patterns": [], "position": []}

    # Entities
    entity_counter = Counter()
    for doc in spacy_docs:
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "PERSON", "GPE"] and 2 <= len(ent.text.split()) <= 5:
                entity_counter[ent.text.lower()] += 1
    for entity, count in entity_counter.items():
        if count >= min_phrase_freq and len(entity) > 3:
            freq = count / len(author_texts)
            results['entities'].append({'phrase': entity, 'score': freq * 100, 'type': 'entity'})

    # Pattern matching (greetings/sign-offs/branding)
    from spacy.matcher import Matcher
    matcher = Matcher(ctx.nlp.vocab)
    greeting_patterns = [
        [{"LOWER": "hey"}, {"LOWER": "everyone"}],
        [{"LOWER": "hello"}, {"LOWER": "folks"}],
        [{"LOWER": "hi"}, {"LOWER": "guys"}],
        [{"LOWER": "what's"}, {"LOWER": "up"}],
        [{"LOWER": "welcome"}, {"LOWER": "back"}],
        [{"LOWER": "welcome"}, {"LOWER": "to"}, {"LOWER": "my"}],
        [{"LOWER": "welcome"}, {"LOWER": "to"}, {"LOWER": "another"}],
    ]
    signoff_patterns = [
        [{"LOWER": "thanks"}, {"LOWER": "for"}, {"LOWER": "watching"}],
        [{"LOWER": "see"}, {"LOWER": "you"}, {"LOWER": "next"}, {"LOWER": "time"}],
        [{"LOWER": "don't"}, {"LOWER": "forget"}, {"LOWER": "to"}, {"LOWER": "subscribe"}],
        [{"LOWER": "don't"}, {"LOWER": "forget"}, {"LOWER": "to"}, {"LOWER": "like"}],
        [{"LOWER": "until"}, {"LOWER": "next"}, {"LOWER": "time"}],
        [{"LOWER": "like"}, {"LOWER": "and"}, {"LOWER": "subscribe"}],
        [{"LOWER": "hit"}, {"LOWER": "that"}, {"LOWER": "like"}, {"LOWER": "button"}],
    ]
    branding_patterns = [
        [{"LOWER": "check"}, {"LOWER": "out"}, {"LOWER": "my"}],
        [{"LOWER": "follow"}, {"LOWER": "me"}, {"LOWER": "on"}],
        [{"LOWER": "link"}, {"LOWER": "in"}, {"LOWER": "the"}, {"LOWER": "description"}],
        [{"LOWER": "join"}, {"LOWER": "my"}],
        [{"LOWER": "on"}, {"LOWER": "my"}, {"LOWER": "channel"}],
        [{"TEXT": {"REGEX": "@[a-zA-Z0-9_]+"}}],
    ]
    for i, p in enumerate(greeting_patterns): matcher.add(f"GREETING_{i}", [p])
    for i, p in enumerate(signoff_patterns): matcher.add(f"SIGNOFF_{i}", [p])
    for i, p in enumerate(branding_patterns): matcher.add(f"BRANDING_{i}", [p])

    pattern_counter = Counter()
    for doc in spacy_docs:
        matches = matcher(doc)
        for _, start, end in matches:
            pattern_text = doc[start:end].text.lower()
            pattern_counter[pattern_text] += 1
    for pattern, count in pattern_counter.items():
        if count >= min_phrase_freq:
            t = pattern
            pattern_type = "unknown"
            if any(p in t for p in ["welcome", "hey", "hello", "hi", "what's up"]):
                pattern_type = "greeting"
            elif any(p in t for p in ["thanks", "thank you", "see you", "next time", "subscribe"]):
                pattern_type = "signoff"
            elif any(p in t for p in ["@", "channel", "follow me", "check out"]):
                pattern_type = "branding"
            results['patterns'].append({'phrase': pattern, 'score': (count / len(author_texts)) * 80, 'type': pattern_type})

    # Positional patterns
    first_sentences, last_sentences = [], []
    for text in author_texts:
        sentences = sent_tokenize(text)
        if sentences:
            fw = sentences[0].split()[:8]
            if len(fw) >= 3: first_sentences.append(' '.join(fw).lower())
            if len(sentences) > 1:
                lw = sentences[-1].split()[-8:]
                if len(lw) >= 3: last_sentences.append(' '.join(lw).lower())
    for phrase, count in Counter(first_sentences).items():
        if count >= min_phrase_freq:
            results['position'].append({'phrase': phrase, 'score': (count / len(author_texts)) * 90, 'type': 'first_sentence'})
    for phrase, count in Counter(last_sentences).items():
        if count >= min_phrase_freq:
            results['position'].append({'phrase': phrase, 'score': (count / len(author_texts)) * 90, 'type': 'last_sentence'})

    # Combine and dedup
    all_results = []
    for _, items in results.items():
        all_results.extend(items)
    all_results.sort(key=lambda x: x['score'], reverse=True)
    final_results = []
    seen = set()
    for result in all_results:
        phrase = result['phrase']
        key = ''.join(phrase.split()).lower()
        if any(similar(key, s) for s in seen):
            continue
        final_results.append(result)
        seen.add(key)
        if len(final_results) >= max_phrases:
            break

    # Format output
    formatted = []
    for r in final_results:
        p = r['phrase']
        if r['type'] in ('greeting_phrase', 'first_sentence'):
            formatted.append(f"greeting: {p}")
        elif r['type'] in ('signoff_phrase', 'last_sentence'):
            formatted.append(f"sign-off: {p}")
        elif r['type'] in ('entity', 'branding'):
            formatted.append(f"brand: {p}")
        else:
            formatted.append(p)
    return formatted[:max_phrases]


def get_uncommon_regular_words(ctx: FeatureContext, texts: List[str], common_word_threshold: int = 1000) -> List[str]:
    sampled = texts[:5] if len(texts) > 5 else texts
    all_text = ' '.join(sampled)
    if len(all_text) > 20000:
        all_text = all_text[:20000]
    import re
    words = re.findall(r'\b\w+\b', all_text.lower())
    from collections import Counter
    word_freq = Counter(words)
    stop_words = ctx.stopwords_en
    filtered = {w for w in word_freq if word_freq[w] < common_word_threshold and w not in stop_words and len(w) > 3}
    regular_words = [w for w in filtered if word_freq[w] >= 2]
    return regular_words[:15]

