import json
import os
import random
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
import pandas as pd

# Compatibility bootstrap: allow running this file directly or as part of the package
if __package__ is None or __package__ == "":
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from utils.logging_setup import init_pipeline_logging
from utils.seed import set_global_seed
from utils.manifest import read_manifest, discover_input, compute_hash, should_skip, update_stage
from utils.run_id import get_last_run_id
from utils.version import STAGE_VERSION

# Configure logging
logger = init_pipeline_logging("phase2.tone_ablation", None, "10-tone-micro-ablation")

# Defaults (overridable via CLI)
RUN_ID = None
BASE_DIR = "data/processed"
INPUT_FILE = None  # resolved from manifest when run-id provided
SAMPLE_PERCENTAGE = 0.20  # Test 20% of the dataset
MODEL_NAME = "Qwen/Qwen3-32B"
MIN_SAMPLE_SIZE = 5
MAX_SAMPLE_SIZE = 20

def create_prompt(post, include_tone=True):
    """
    Create prompt with or without tone feature
    All other features remain constant between conditions
    """
    # Base prompt components that stay the same in both conditions
    structure = post.get('structure', 'insightful')
    post_text = post.get('post_text', '')
    emoji_usage = post.get('emoji_usage', 'medium')
    topic = post.get('topic', 'Professional Development')
    
    # First line is the same for both conditions
    first_line = f"Create a LinkedIn post on the topic of: `{topic}`"
    
    # Only difference is whether we include specific tone or use default tone
    if include_tone and 'tone' in post:
        tone = post.get('tone')
    else:
        # Default neutral tone when tone feature is ablated
        tone = "professional, engaging"
        
    # Build the prompt with all features identical except for tone
    content = f"""{first_line}

### Key Message

{post_text}

### Writing Constraints
- **Suggested Post Length**: Between 750 and 1,500 characters long
- **Emoji Usage**: {emoji_usage}
- **Tone**: {tone}
- **Structure**: {structure}
"""
    # Wrap in chat template for Qwen-style chat models
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

def generate_posts(llm, posts, include_tone=True, llm_batch=8, temperature=0.7, max_tokens=1000, top_p=0.9):
    """
    Generate LinkedIn posts using the LLM in batches.
    """
    logger.info(f"Generating posts with tone={'included' if include_tone else 'excluded'}...")
    prompts = [create_prompt(post, include_tone) for post in posts]

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )

    generated_texts: List[str] = []

    for i in range(0, len(prompts), llm_batch):
        batch_prompts = prompts[i:i+llm_batch]
        try:
            outputs = llm.generate(batch_prompts, sampling_params)
            batch_texts = [output.outputs[0].text if (output and output.outputs) else "" for output in outputs]
            generated_texts.extend(batch_texts)
        except Exception as e:
            logger.error(f"Error generating batch {i//llm_batch}: {e}")
            generated_texts.extend(["" for _ in range(len(batch_prompts))])

    return generated_texts


def compute_embeddings(texts, model, embed_batch=64):
    """
    Compute embeddings for a list of texts with batch inference.
    """
    logger.info("Computing embeddings...")
    valid_texts = [text for text in texts if text.strip()]
    if not valid_texts:
        logger.error("No valid texts to embed!")
        return np.array([])

    parts = []
    for i in range(0, len(valid_texts), embed_batch):
        parts.append(model.encode(valid_texts[i:i+embed_batch]))
    return np.vstack(parts) if parts else np.array([])

def compute_embedding_similarities(with_tone_embeddings, without_tone_embeddings):
    """
    Compute cosine similarities between embeddings
    """
    logger.info("Computing similarities between embeddings...")
    similarities = []
    
    for i in range(min(len(with_tone_embeddings), len(without_tone_embeddings))):
        sim = cosine_similarity(
            with_tone_embeddings[i].reshape(1, -1),
            without_tone_embeddings[i].reshape(1, -1)
        )[0][0]
        similarities.append(sim)
    
    return similarities

def analyze_differences(with_tone_texts, without_tone_texts, posts):
    """
    Analyze differences between texts generated with and without tone
    """
    logger.info("Analyzing text differences...")
    results = []
    
    for i, (with_t, without_t, post) in enumerate(zip(with_tone_texts, without_tone_texts, posts)):
        # Skip empty generations
        if not with_t.strip() or not without_t.strip():
            continue
            
        # Basic text statistics
        with_t_length = len(with_t)
        without_t_length = len(without_t)
        length_diff = abs(with_t_length - without_t_length)
        length_diff_pct = length_diff / max(with_t_length, without_t_length) * 100
        
        # Calculate number of paragraphs
        with_t_paragraphs = len([p for p in with_t.split('\n\n') if p.strip()])
        without_t_paragraphs = len([p for p in without_t.split('\n\n') if p.strip()])
        paragraph_diff = abs(with_t_paragraphs - without_t_paragraphs)
        
        # Count bullet points (simplified approach)
        with_t_bullets = with_t.count('- ')
        without_t_bullets = without_t.count('- ')
        bullet_diff = abs(with_t_bullets - without_t_bullets)
        
        # Get tone metadata
        tone = post.get('tone', 'professional, engaging')
        
        results.append({
            'index': i,
            'tone': tone,
            'with_tone_length': with_t_length,
            'without_tone_length': without_t_length,
            'length_diff_pct': length_diff_pct,
            'paragraph_diff': paragraph_diff,
            'bullet_diff': bullet_diff,
            'with_tone_text': with_t[:200] + "..." if len(with_t) > 200 else with_t,
            'without_tone_text': without_t[:200] + "..." if len(without_t) > 200 else without_t
        })
    
    return pd.DataFrame(results)

def categorize_tone(tone_string):
    """
    Categorize tones into broader groups for visualization
    Since tones can be multiple values, we'll assign the primary category
    based on the first tone that matches
    """
    # Convert to lowercase for case-insensitive matching
    tone_lower = tone_string.lower()
    
    # Define tone categories and their keywords
    categories = {
        'Professional/Formal': ['professional', 'formal', 'authoritative', 'scholarly', 'serious', 
                               'no-nonsense', 'trustworthy', 'reliable', 'stable', 'capable'],
        'Friendly/Casual': ['friendly', 'casual', 'conversational', 'comfortable', 'charming', 
                           'endearing', 'inviting', 'natural', 'caring'],
        'Energetic/Exciting': ['energetic', 'exciting', 'bold', 'adventurous', 'intense', 
                              'lively', 'stimulating', 'upbeat', 'strong', 'powerful', 'dramatic'],
        'Creative/Artistic': ['creative', 'artistic', 'whimsical', 'quirky', 'playful', 
                             'colorful', 'unique', 'eccentric', 'unconventional'],
        'Informative/Educational': ['informative', 'detailed', 'thoughtful', 'smart', 'scholarly'],
        'Positive/Inspiring': ['inspiring', 'cheerful', 'bright', 'delightful', 'optimistic', 
                              'fabulous', 'engaging', 'persuasive', 'fun']
    }
    
    # Check each tone in the comma-separated string
    for tone in tone_lower.split(','):
        tone = tone.strip()
        for category, keywords in categories.items():
            if any(keyword in tone for keyword in keywords):
                return category
    
    # If no match found, return Other
    return 'Other'

def visualize_results(results_df, similarities):
    """
    Visualize the results of the ablation test for tones
    """
    logger.info("Creating visualizations...")
    # Clean up dataframe, remove rows without similarities
    valid_indices = list(range(min(len(similarities), len(results_df))))
    if len(valid_indices) < len(results_df):
        results_df = results_df.iloc[valid_indices].reset_index(drop=True)
    
    # Add similarities to the DataFrame
    if len(similarities) > 0:
        results_df['embedding_similarity'] = similarities[:len(results_df)]
    else:
        results_df['embedding_similarity'] = 0
    
    # Add tone categories
    results_df['tone_category'] = results_df['tone'].apply(categorize_tone)
    
    # Create figure
    plt.figure(figsize=(16, 14))
    
    # Plot 1: Overall similarity distribution
    plt.subplot(2, 2, 1)
    sns.histplot(results_df['embedding_similarity'], bins=15, kde=True)
    plt.axvline(results_df['embedding_similarity'].mean(), color='r', linestyle='--', 
                label=f'Mean: {results_df["embedding_similarity"].mean():.3f}')
    plt.title('Distribution of Semantic Similarities\nWith vs Without Tone', fontsize=14)
    plt.xlabel('Cosine Similarity (higher = more similar)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend()
    
    # Plot 2: Impact (1-similarity) by tone category
    plt.subplot(2, 2, 2)
    tone_impact = results_df.groupby('tone_category')['embedding_similarity'].agg(['mean', 'std', 'count']).reset_index()
    tone_impact['impact'] = 1 - tone_impact['mean']
    tone_impact = tone_impact.sort_values('impact', ascending=False)
    
    # Add hue column for color coding
    tone_impact['tone_category_hue'] = tone_impact['tone_category']
    
    # Plot with error bars
    sns.barplot(x='tone_category', y='impact', hue='tone_category_hue', 
                data=tone_impact, palette='viridis', legend=False, ax=plt.gca())
    
    # Add error bars
    for i, row in tone_impact.iterrows():
        plt.errorbar(i, row['impact'], yerr=row['std'], color='black', capsize=5, alpha=0.7)
    
    plt.title('Tone Impact by Category\n(Higher = More Change)', fontsize=14)
    plt.xlabel('Tone Category', fontsize=12)
    plt.ylabel('Impact (1-Similarity)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    
    # Plot 3: Length differences by tone categories
    plt.subplot(2, 2, 3)
    tone_length = results_df.groupby('tone_category')['length_diff_pct'].mean().reset_index()
    tone_length = tone_length.sort_values('length_diff_pct', ascending=False)
    
    # Add hue column
    tone_length['tone_category_hue'] = tone_length['tone_category']
    
    sns.barplot(x='tone_category', y='length_diff_pct', hue='tone_category_hue', 
                data=tone_length, palette='magma', legend=False)
    plt.title('Text Length Difference by Tone Category', fontsize=14)
    plt.xlabel('Tone Category', fontsize=12)
    plt.ylabel('Mean Length Difference (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Plot 4: Feature impact summary
    plt.subplot(2, 2, 4)
    
    # Calculate overall impact metrics
    mean_similarity = results_df['embedding_similarity'].mean()
    mean_length_diff = results_df['length_diff_pct'].mean()
    mean_paragraph_diff = results_df['paragraph_diff'].mean()
    mean_bullet_diff = results_df['bullet_diff'].mean()
    
    # Normalize metrics for comparison
    impact_metrics = {
        'Semantic Difference': 1 - mean_similarity,
        'Length Difference': mean_length_diff / 100,  # Convert percentage to 0-1 scale
        'Paragraph Structure': mean_paragraph_diff / 5,  # Normalize assuming max diff of ~5 paragraphs
        'Bullet Points': mean_bullet_diff / 10  # Normalize assuming max diff of ~10 bullets
    }
    
    # Create summary bar chart
    impact_df = pd.DataFrame({
        'Metric': list(impact_metrics.keys()),
        'Impact': list(impact_metrics.values())
    })
    
    # Add hue column
    impact_df['Metric_hue'] = impact_df['Metric']
    
    sns.barplot(x='Impact', y='Metric', hue='Metric_hue', 
                data=impact_df, palette='plasma', legend=False)
    plt.title('Overall Impact Metrics\n(Higher = More Change)', fontsize=14)
    plt.xlabel('Normalized Impact (0-1 scale)', fontsize=12)
    plt.xlim(0, 1.0)
    
    # Add value annotations
    for i, v in enumerate(impact_df['Impact']):
        plt.text(v + 0.02, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig('tone_ablation_results.png', dpi=300)
    logger.info("Saved visualization to tone_ablation_results.png")
    
    # Create and save correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_columns = ['embedding_similarity', 'length_diff_pct', 'paragraph_diff', 'bullet_diff']
    corr_matrix = results_df[corr_columns].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Between Different Impact Metrics', fontsize=14)
    plt.tight_layout()
    plt.savefig('tone_ablation_correlations.png', dpi=300)
    logger.info("Saved correlation matrix to tone_ablation_correlations.png")

def _resolve_input(run_id: str | None, base_dir: str, explicit_input: str | None) -> str:
    if run_id:
        if run_id == "latest":
            rid = get_last_run_id(base_dir)
            if not rid:
                raise ValueError("No .last_run_id found; run previous stages first or set --run-id")
            run_id = rid
        manifest = read_manifest(run_id, base_dir)
        discovered = discover_input(manifest, "09-extract-tone") or discover_input(manifest, "07-clean-topics")
        if discovered and os.path.exists(discovered):
            return discovered
        return os.path.join(base_dir, run_id, "09-tone.jsonl")
    if not explicit_input:
        raise ValueError("Provide --input when --run-id is not used")
    return explicit_input


def run_tone_ablation_test(run_id: str | None,
                           base_dir: str,
                           input_path: str | None,
                           sample_pct: float,
                           model_name: str,
                           seed: int | None,
                           llm_batch: int,
                           embed_batch: int):
    logger.info("Starting micro ablation test for tones…")

    set_global_seed(seed)

    # Resolve input and report outputs
    in_path = _resolve_input(run_id, base_dir, input_path)
    reports_dir = os.path.join("reports", run_id or "adhoc", "tone-ablation")
    os.makedirs(reports_dir, exist_ok=True)

    sig = compute_hash([in_path], {"stage": 10, "sample_pct": sample_pct, "model": model_name, "llm_batch": llm_batch, "embed_batch": embed_batch, "stage_version": STAGE_VERSION})
    if run_id:
        manifest = read_manifest(run_id, base_dir)
        out_paths = [
            os.path.join(reports_dir, "tone_ablation_results.png"),
            os.path.join(reports_dir, "tone_ablation_correlations.png"),
            os.path.join(reports_dir, "tone_ablation_details.txt"),
        ]
        if should_skip(manifest, "10-tone-ablation", sig, out_paths):
            logger.info(f"Skipping 10-tone-ablation; up-to-date in {reports_dir}")
            return

    # Streaming: reservoir sample of eligible posts
    eligible = 0
    sample_posts: List[dict] = []
    with open(in_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                post = json.loads(line)
            except Exception:
                continue
            if not post.get('post_text') or not post.get('tone'):
                continue
            eligible += 1
            desired = max(MIN_SAMPLE_SIZE, min(MAX_SAMPLE_SIZE, int(eligible * sample_pct)))
            if len(sample_posts) < desired:
                sample_posts.append(post)
            else:
                j = random.randint(0, eligible - 1)
                if j < desired:
                    sample_posts[j] = post

    logger.info(f"Eligible posts: {eligible}, sampled: {len(sample_posts)}")
    if not sample_posts:
        logger.warning("No samples selected; aborting.")
        return

    # Initialize models
    logger.info(f"Loading LLM: {model_name}")
    try:
        llm = LLM(model=model_name, max_model_len=8192)
    except Exception as e:
        logger.error(f"Error loading LLM: {e}")
        return

    logger.info("Loading embedding model…")
    try:
        embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        return

    # Generate with and without tone
    with_tone_texts = generate_posts(llm, sample_posts, include_tone=True, llm_batch=llm_batch)
    without_tone_texts = generate_posts(llm, sample_posts, include_tone=False, llm_batch=llm_batch)

    valid_with_texts = [t for t in with_tone_texts if t.strip()]
    valid_without_texts = [t for t in without_tone_texts if t.strip()]
    if not valid_with_texts or not valid_without_texts:
        logger.error("No valid texts for embedding!")
        return

    with_tone_embeddings = compute_embeddings(valid_with_texts, embedding_model, embed_batch=embed_batch)
    without_tone_embeddings = compute_embeddings(valid_without_texts, embedding_model, embed_batch=embed_batch)

    similarities = []
    if len(with_tone_embeddings) > 0 and len(without_tone_embeddings) > 0:
        similarities = compute_embedding_similarities(with_tone_embeddings, without_tone_embeddings)

    # Analyze and visualize
    results_df = analyze_differences(with_tone_texts, without_tone_texts, sample_posts)
    if len(results_df) == 0:
        logger.error("No valid results to analyze!")
        return

    visualize_results(results_df, similarities)

    mean_similarity = float(np.mean(similarities)) if similarities else 0.0
    overall_impact = 1 - mean_similarity

    # Save detailed results and plots under reports
    details_path = os.path.join(reports_dir, 'tone_ablation_details.txt')
    with open(details_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TONE MICRO ABLATION TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"- Posts analyzed: {len(results_df)}\n")
        f.write(f"- Overall tone impact: {overall_impact:.4f} (0=none, 1=complete change)\n")
        f.write(f"- Mean similarity between versions: {mean_similarity:.4f}\n")
        f.write(f"- Mean text length difference: {results_df['length_diff_pct'].mean():.2f}%\n\n")

        results_df['tone_category'] = results_df['tone'].apply(categorize_tone)
        f.write("Impact by tone category:\n")
        tone_impact_df = results_df.groupby('tone_category')['embedding_similarity'].agg(['mean', 'std', 'count']).reset_index()
        tone_impact_df['impact'] = 1 - tone_impact_df['mean']
        tone_impact_df = tone_impact_df.sort_values('impact', ascending=False)
        for _, row in tone_impact_df.iterrows():
            f.write(f"- {row['tone_category']}: {row['impact']:.4f} (samples: {row['count']})\n")

        f.write("\n\n" + "=" * 80 + "\n")
        f.write("EXAMPLE COMPARISONS\n")
        f.write("=" * 80 + "\n\n")
        if 'embedding_similarity' in results_df.columns and len(results_df) > 0:
            most_different = results_df.sort_values('embedding_similarity').head(5)
            for _, row in most_different.iterrows():
                f.write(f"Example (tone: {row['tone']}, similarity: {row['embedding_similarity']:.4f}):\n\n")
                with_text = with_tone_texts[row['index']] if row['index'] < len(with_tone_texts) else "N/A"
                without_text = without_tone_texts[row['index']] if row['index'] < len(without_tone_texts) else "N/A"
                f.write("WITH TONE:\n" + with_text + "\n\n")
                f.write("WITHOUT TONE:\n" + without_text + "\n\n")
                f.write("-" * 80 + "\n\n")

    # Save figures to reports
    plt.savefig(os.path.join(reports_dir, 'tone_ablation_results.png'), dpi=300)
    plt.close('all')
    plt.savefig(os.path.join(reports_dir, 'tone_ablation_correlations.png'), dpi=300)

    logger.info(f"Saved tone ablation artifacts to {reports_dir}")

    if run_id:
        manifest = read_manifest(run_id, base_dir)
        update_stage(run_id, base_dir, manifest, "10-tone-ablation", in_path, [
            os.path.join(reports_dir, 'tone_ablation_results.png'),
            os.path.join(reports_dir, 'tone_ablation_correlations.png'),
            details_path,
        ], sig, extra={"sampled": len(sample_posts), "eligible": eligible, "mean_similarity": mean_similarity})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tone micro-ablation using 09-tone output")
    parser.add_argument("--run-id", dest="run_id", default=None, help="Use 'latest' to use most recent run")
    parser.add_argument("--base-dir", dest="base_dir", default="data/processed")
    parser.add_argument("--input", dest="input_path", default=None, help="Only used when --run-id is not provided")
    parser.add_argument("--sample", dest="sample_pct", type=float, default=SAMPLE_PERCENTAGE)
    parser.add_argument("--model", dest="model_name", default=MODEL_NAME)
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("--llm-batch", dest="llm_batch", type=int, default=8)
    parser.add_argument("--embed-batch", dest="embed_batch", type=int, default=64)
    args = parser.parse_args()

    try:
        run_tone_ablation_test(
            run_id=args.run_id,
            base_dir=args.base_dir,
            input_path=args.input_path,
            sample_pct=args.sample_pct,
            model_name=args.model_name,
            seed=args.seed,
            llm_batch=args.llm_batch,
            embed_batch=args.embed_batch,
        )
    except Exception as e:
        logger.error(f"Error in tone micro ablation test: {e}", exc_info=True)
        print(f"Test failed with error: {e}")