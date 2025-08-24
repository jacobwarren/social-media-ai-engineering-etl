import re
import json
import random
import os
import argparse
from datetime import datetime
# Compatibility bootstrap: allow running this file directly or as part of the package
if __package__ is None or __package__ == "":
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from pipe.utils.logging_setup import init_pipeline_logging
from pipe.utils.manifest import read_manifest, write_manifest, compute_hash, should_skip, update_stage, discover_input
from pipe.schemas import PromptRecord, validate_record
from pipe.utils.seed import set_global_seed
from pipe.utils.cli import add_standard_args, resolve_common_args

#######################################################
#          1. Utility Descriptions (Updated)          #
#######################################################

def describe_sentence_structure(sentence_lengths):
    """Describe sentence length patterns."""
    if not sentence_lengths:
        return "No sentences found."
    avg_length = sum(sentence_lengths) / len(sentence_lengths)
    if avg_length < 10:
        return "Short sentences, suggesting brevity and conciseness."
    elif avg_length > 20:
        return "Long and complex sentences, indicating a detailed and elaborate style."
    else:
        return "A mix of short and long sentences, showing a balanced style."

def describe_vocabulary_usage(vocab_count, post_text):
    """Describe how rich or narrow the vocabulary is, based on a ratio of unique words."""
    words = [word.lower() for word in post_text.split() if word.isalpha()]
    total_words = len(words)
    vocab_ratio = vocab_count / total_words if total_words > 0 else 0

    if vocab_ratio > 0.5:     # More than 50% unique
        return "A rich vocabulary, showcasing extensive language use and depth."
    elif vocab_ratio > 0.35:  # More than 35% unique
        return "A developed vocabulary, indicating a wide range of language and expression."
    elif vocab_ratio > 0.25:  # More than 25% unique
        return "A normal vocabulary, reflecting a balanced and versatile use of language."
    elif vocab_ratio > 0.15:  # More than 15% unique
        return "A conservative vocabulary, suggesting a focused and deliberate choice of words."
    else:
        return "A very narrow vocabulary, highlighting a specific and targeted use of language."

def describe_line_breaks(line_breaks, avg_breaks):
    """Interpret the line-break count and average usage."""
    if line_breaks == 0:
        return "No line breaks, indicating a continuous block of text."
    if line_breaks > 10:
        return "Frequent line breaks, contributing to an easy-to-read structure."
    elif avg_breaks < 0.2:
        return "Fewer line breaks, indicating a more compact writing style."
    else:
        return "A moderate number of line breaks, balancing readability and density."

def describe_punctuation(punctuation_usage, post_text):
    """Provide an overview of punctuation usage intensity."""
    desc = []
    total_length = len(post_text)

    # If the text is extremely short, skip
    if total_length < 1:
        return "No punctuation data."

    for punctuation, count in punctuation_usage.items():
        if count == 0:
            continue
        ratio = count / total_length
        # Example thresholds; adjust as desired
        if punctuation == ".":
            if ratio > 0.02:
                desc.append("Heavy use of periods.")
            elif ratio > 0.01:
                desc.append("Regular use of periods.")
        elif punctuation == ",":
            if ratio > 0.02:
                desc.append("Heavy use of commas.")
            elif ratio > 0.01:
                desc.append("Regular use of commas.")
        elif punctuation == "!":
            if ratio > 0.02:
                desc.append("Heavy use of exclamation marks.")
            elif ratio > 0.01:
                desc.append("Regular use of exclamation marks.")
        elif punctuation == "?":
            if ratio > 0.02:
                desc.append("Heavy use of question marks.")
            elif ratio > 0.01:
                desc.append("Regular use of question marks.")
        elif punctuation == ";":
            if ratio > 0.02:
                desc.append("Heavy use of semicolons.")
            elif ratio > 0.01:
                desc.append("Regular use of semicolons.")

    return ' '.join(desc) if desc else "Standard punctuation usage."

def describe_bullet_styles(bullet_style):
    """Interpret the bullet style from the new detection method."""
    # Some "illogical" bullet styles get replaced with a generic bullet
    illogical_bullet_styles = ['"', "'", '""', '#', '$', '%', '&', '(', ')',
                               ',', '.', '/', ':', ';', '<', '=', '>', '?', '@',
                               '[', '\\', ']', '^', '_', '`', '{', '|', '}']

    if bullet_style in illogical_bullet_styles:
        bullet_style = '•'

    if bullet_style is None:
        return "No specific bullet style, indicating a straightforward narrative style."
    elif bullet_style == "Differing Emojis":
        return "Uses varying emojis as bullet points, adding a casual and modern touch."
    elif bullet_style == "EmojiBullets":
        return "Uses multiple emojis as bullet points, bringing a fun, visually engaging style."
    elif bullet_style == "Mixed Bullet Styles":
        return "Multiple bullet styles detected, indicating a creative or varied structuring."
    else:
        # Could be "Numbered", "Lettered", "Emoji", or a single symbol like '-'
        return f"Uses {bullet_style} for bullet points, indicating a structured format."

def describe_topic_shifts(topic_shifts):
    """
    Now that we use a BERT-based approach returning a list of dicts like:
        [{"from_segment": 0, "to_segment": 1, "shift_score": 0.45}, ...]
    We look at the max shift_score to gauge how dynamic the shifts are.
    """
    if not topic_shifts:
        return "Consistent topic focus, highlighting a thorough exploration of a single subject."

    # Extract shift_score from each transition dict
    shift_scores = [t["shift_score"] for t in topic_shifts if "shift_score" in t]
    if not shift_scores:
        return "Consistent topic focus, highlighting a thorough exploration of a single subject."

    overall_max_shift = max(shift_scores)

    if overall_max_shift > 0.8:
        return "Dynamic topic shifts, showing a highly versatile and engaging writing style."
    elif overall_max_shift > 0.6:
        return "Regular topic shifts, reflecting a balanced and varied approach."
    elif overall_max_shift > 0.4:
        return "Moderate topic shifts, indicating a well-rounded but focused narrative."
    elif overall_max_shift > 0.2:
        return "Conservative topic shifts, suggesting a cautious approach to topic changes."
    else:
        return "Consistent topic focus, highlighting a deep and thorough exploration of subjects."

def describe_narrative_flow(flow):
    """
    The new flow can include:
      - "Introduction/Setup"
      - "Conflict/Resolution Point"
      - "Introduction/Development"
      - "Transition/Reflection"
    """
    if not flow:
        return "No discernible narrative flow detected."

    # For very short flows
    if len(flow) <= 4:
        descriptions = []
        for i, label in enumerate(flow):
            if label == "Introduction/Setup":
                text = "introducing the subject" if i == 0 else "setting up a new idea"
            elif label == "Introduction/Development":
                text = "introducing or expanding on key concepts"
            elif label == "Conflict/Resolution Point":
                text = "highlighting conflict or resolution"
            elif label == "Transition/Reflection":
                text = "shifting into reflection or transitioning between ideas"
            else:
                text = label  # fallback if new labels appear
            descriptions.append(text)

        # Join them more nicely
        combined = "; then ".join(descriptions)
        return f"The narrative flow includes: {combined}."
    else:
        return ("A complex narrative flow that progresses through multiple stages, "
                "indicating a dynamic and layered storytelling approach.")

def describe_pacing(pacing):
    """
    The new pacing can be:
      - "Fast", "Slow", "Variable", "Dynamic", "Moderate"
      - OR "Short/Not Enough Data" if < 3 sentences
    """
    if pacing == "Short/Not Enough Data":
        return "Pacing analysis is inconclusive due to the brevity of the text."
    return f"The pacing is described as '{pacing}', indicating the rhythm and speed of the narrative."

def describe_sentiment_arc(sentiment_arc):
    """
    The updated sentiment arc can be:
      - "Short/Not Enough Data for Arc"
      - "Upward Trend", "Downward Trend"
      - "Stable", "Complex/Variable"
    Also handle backup labels: "Positive", "Negative", "Neutral".
    """
    descriptions = {
        "Upward Trend": (
            "A steadily rising sentiment, indicating an increasingly positive or hopeful tone."
        ),
        "Downward Trend": (
            "A consistently declining sentiment, suggesting a shift toward negativity or seriousness."
        ),
        "Stable": (
            "A relatively stable sentiment, implying a consistent emotional tone throughout."
        ),
        "Complex/Variable": (
            "A multifaceted sentiment arc with multiple ups and downs, reflecting a nuanced emotional journey."
        ),
        "Short/Not Enough Data for Arc": (
            "Insufficient length to determine a clear sentiment progression."
        ),
    }
    backup_descriptions = {
        "Positive": "Overall positive tone, conveying optimism or encouragement.",
        "Negative": "Overall negative tone, conveying concern or seriousness.",
        "Neutral":  "Balanced tone without a strong emotional shift.",
    }
    return descriptions.get(
        sentiment_arc,
        backup_descriptions.get(
            sentiment_arc,
            "A diverse emotional range, showcasing a dynamic and unpredictable sentiment."
        )
    )

def describe_phrases(phrases):
    """List uncommon/regular phrases if available."""
    if not phrases:
        return "No particularly common or distinctive phrases identified."
    return ", ".join(phrases)

#######################################################
#    2. Markdown-Based Prompt Generation (Updated)    #
#######################################################

def generate_writing_style_summary(data, include_writing_style, common_phrases):
    """
    Produce a structured Markdown prompt describing the post context, key messages,
    and extracted writing-style features. This format is more LLM-friendly.
    """

    # 1) Map structure to a high-level command or creative direction
    structures = {
        "instructional":   "Create a LinkedIn post that **shares a step-by-step guide**",
        "reflective":      "Create a LinkedIn post that **reflects on an experience**",
        "inspirational":   "Create a LinkedIn post that **inspires and motivates**",
        "controversial":   "Create a LinkedIn post that **challenges popular opinions**",
        "insightful":      "Create a LinkedIn post that **offers keen observations**",
        "comparative":     "Create a LinkedIn post that **compares two or more items**",
        "announcement":    "Create a LinkedIn post that **announces something new**",
    }

    # 2) Start building the prompt
    structure_label = data.get('structure')
    base_command = structures.get(structure_label)

    # We assume data['topic'] or data['key_message'] might be missing if not fully validated
    topic = data.get('topic')
    key_msg = f"{data.get('opinion')} {data.get('context')}"

    # 3) Build a Markdown output
    # ----------------------------------------------------
    # HEAD
    # ----------------------------------------------------
    summary = []
    summary.append(f"# Request")
    summary.append(f"{base_command} **on the topic of**: `{topic}`\n")

    summary.append("### Key Message")
    summary.append(f"```\n{key_msg}\n```")

    # 4) Basic style constraints
    summary.append("### Writing Constraints")
    max_len = data.get('max_length')
    tone = data.get('tone')
    emoji_usage = data.get('emoji_usage')

    summary.append(f"- **Suggested Post Length**: {max_len}")
    summary.append(f"- **Emoji Usage**: {emoji_usage}")
    summary.append(f"- **Tone**: {tone}")

    # 5) Conditionally include the deeper writing style analysis
    if include_writing_style:
        summary.append("### Writing Style Features")

        # Sentence structure
        if data.get('sentence_structure') is not None:
            desc = describe_sentence_structure(data['sentence_structure'])
            summary.append(f"- **Sentence Structure**: {desc}")

        # Vocabulary
        if data.get('vocabulary_usage') is not None:
            vocab_desc = describe_vocabulary_usage(data['vocabulary_usage'], data.get('post_text', ''))
            summary.append(f"- **Vocabulary Usage**: {vocab_desc}")

        # Common phrases
        if len(common_phrases) > 0:
            phrase_desc = describe_phrases(common_phrases)
            summary.append(f"- **Common Phrases**: {phrase_desc}")

        # Divider style
        if data.get('divider_style') is not None:
            summary.append(f"- **Section Divider**: `{data['divider_style']}`")

        # Line Breaks
        if data.get('line_breaks') is not None and data.get('avg_line_breaks') is not None:
            line_desc = describe_line_breaks(data['line_breaks'], data['avg_line_breaks'])
            summary.append(f"- **Line Break Usage**: {line_desc}")

        # Punctuation
        if data.get('punctuation_usage') is not None:
            punctuation_desc = describe_punctuation(data['punctuation_usage'], data.get('post_text', ''))
            summary.append(f"- **Punctuation**: {punctuation_desc}")

        # Bullet Styles
        if data.get('bullet_styles') is not None:
            bullet_desc = describe_bullet_styles(data['bullet_styles'])
            summary.append(f"- **Bullet Styles**: {bullet_desc}")

        # Topic Shifts
        if data.get('topic_shifts') is not None:
            shift_desc = describe_topic_shifts(data['topic_shifts'])
            summary.append(f"- **Topic Shifts**: {shift_desc}")

        # Narrative Flow
        if data.get('flow') is not None:
            flow_desc = describe_narrative_flow(data['flow'])
            summary.append(f"- **Narrative Flow**: {flow_desc}")

        # Pacing
        if data.get('pacing') is not None:
            pacing_desc = describe_pacing(data['pacing'])
            summary.append(f"- **Pacing**: {pacing_desc}")

        # Sentiment Arc
        if data.get('sentiment_arc') is not None:
            arc_desc = describe_sentiment_arc(data['sentiment_arc'])
            summary.append(f"- **Sentiment Arc**: {arc_desc}")

        # Profanity
        if data.get('profanity') is not None:
            summary.append(f"- **Profanity Level**: {data['profanity']}")

    # 6) Join everything with new lines for a final Markdown block
    return "\n".join(summary)


#######################################################
#  3. Main Post-Processing / Prompt Injection Logic   #
#######################################################

def find_first_matching_term(structure):
    """Map user-provided structure text to a known label."""
    if structure is None:
        return None

    words = structure.lower().split()
    terms = [
        'instructional', 'inspirational', 'controversial',
        'insightful', 'comparative', 'reflective', 'announcement'
    ]
    for word in words:
        if word in terms:
            return word
    return None

count = 0

def process_posts(input_path,
                  output_path,
                  run_id=None,
                  base_dir="data/processed",
                  seed=None,
                  print_first_n=5):
    global count

    logger = init_pipeline_logging("phase2.prompts", run_id, "18-generate-prompts")

    # Resolve IO centrally (enforce standardization)
    from pipe.utils.io import resolve_io
    from pipe.utils.artifacts import ArtifactNames
    input_path, std_output_path, run_id = resolve_io(stage="18-prompts", run_id=run_id, base_dir=base_dir, explicit_in=input_path, prior_stage="17-writing-style", std_name=ArtifactNames.STAGE18_PROMPTS)
    logger.info(f"Resolved input: {input_path}; std_out: {std_output_path}")

    # Seeding for determinism of any randomness in this step
    set_global_seed(seed)

    # Skip if up-to-date
    std_outfile = None
    from utils.version import STAGE_VERSION
    signature = compute_hash([input_path], config={"stage": 18, "seed": seed, "stage_version": STAGE_VERSION})
    manifest = read_manifest(run_id, base_dir)
    if should_skip(manifest, "18-prompts", signature, [std_output_path]):
        logger.info(f"Skipping 18-prompts; up-to-date at {std_output_path}")
        return

    std_outfile = open(std_output_path, 'w')

    processed = 0
    skipped = 0

    # Open the input file and standardized output only
    with open(input_path, 'r') as infile:
        for line in infile:
            try:
                # Parse each line as a single post
                post = json.loads(line)

                # Initialize prompt field
                post['prompt'] = None

                # Extract common phrases directly from the post
                common_phrases = post.get('common_phrases', [])

                # Ensure mandatory fields exist
                if (not post.get('post_text')
                        or post.get('structure') is None
                        or post.get('topic') is None
                        or post.get('opinion') is None
                        or post.get('context') is None
                        or post.get('tone') is None):
                    # Write the post without a prompt and continue
                    serialized = json.dumps(post)
                    std_outfile.write(serialized + '\n')
                    skipped += 1
                    continue

                # Map the structure to a known label
                structure = find_first_matching_term(post['structure'])
                if structure is None:
                    # If none found, use generic structure
                    post['structure'] = 'instructional'
                else:
                    post['structure'] = structure

                # Generate the prompt with full writing style detail
                include_writing_style = True
                prompt_text = generate_writing_style_summary(post, include_writing_style, common_phrases)
                post['prompt'] = prompt_text

                # Validate (if pydantic present)
                try:
                    post = validate_record(PromptRecord, post)
                except Exception:
                    pass

                # (Optional) Print it out for debugging
                if count < print_first_n:
                    logger.info(f"PROMPT {count+1}:\n{post['prompt']}")

                count += 1
                processed += 1

                # Write the updated post to standardized output only
                serialized = json.dumps(post)
                std_outfile.write(serialized + '\n')

            except json.JSONDecodeError:
                logger.warning("Invalid JSON line; skipping")
                continue
            except Exception as e:
                logger.error(f"Error processing line: {str(e)}")
    # Validate standardized JSONL before updating manifest
    try:
        from pipe.schemas import Stage18Record
        from pipe.utils.validation import validate_jsonl_records
        ok_std = True
        if std_output_path:
            ok_std = validate_jsonl_records(std_output_path, model_cls=Stage18Record, required_keys=["prompt"])  # ensure 'prompt' exists
        if not ok_std:
            logger.error("Stage18 standardized JSONL failed validation; skipping manifest update")
            return
    except Exception:
        pass

    if std_outfile:
        std_outfile.close()
    if std_output_path and run_id:
        manifest = read_manifest(run_id, base_dir)
        from utils.version import STAGE_VERSION
        signature = compute_hash([input_path], config={"stage": 18, "seed": seed, "stage_version": STAGE_VERSION})
        update_stage(
            run_id,
            base_dir,
            manifest,
            stage_name="18-prompts",
            input_path=input_path,
            outputs=[std_output_path],
            signature=signature,
            extra={"processed": processed, "skipped": skipped},
        )
        logger.info(f"Standardized output written to: {std_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prompts from writing-style-enriched posts (run-id manifest mode only)")
    parser.add_argument("--input", dest="input_path", default=None)
    add_standard_args(parser, include_seed=True)
    parser.add_argument("--print-first-n", dest="print_first_n", type=int, default=5)
    args = parser.parse_args()

    # Require run-id; no ad-hoc output path support
    args = resolve_common_args(args, require_input_when_no_run_id=True)

    print(f"Processing {args.input_path or '[manifest]'} -> [standardized]")
    if args.run_id:
        print(f"Also writing standardized artifact under {args.base_dir}/{args.run_id}")

    process_posts(
        input_path=args.input_path,
        output_path=None,
        run_id=args.run_id,
        base_dir=args.base_dir,
        seed=args.seed,
        print_first_n=args.print_first_n,
    )

    print(f"Total posts with prompts: {count}")