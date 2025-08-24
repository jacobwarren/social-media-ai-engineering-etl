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

from pipe.utils.logging_setup import init_pipeline_logging
from pipe.utils.seed import set_global_seed
from pipe.utils.manifest import read_manifest, discover_input, compute_hash, should_skip, update_stage
from pipe.utils.run_id import get_last_run_id
from pipe.utils.version import STAGE_VERSION

# Configure logging
logger = init_pipeline_logging("phase2.opinion_ablation", None, "13-opinion-micro-ablation")

# Defaults (overridable via CLI)
RUN_ID = None
BASE_DIR = "data/processed"
INPUT_FILE = None  # resolved from manifest when run-id provided
SAMPLE_PERCENTAGE = 0.20  # Test 20% of the dataset
MODEL_NAME = "RekaAI/reka-flash-3"
MIN_SAMPLE_SIZE = 5
MAX_SAMPLE_SIZE = 20


def create_prompt(post, include_opinion=True):
    """
    Create prompt with or without opinion feature
    All other features remain constant between conditions
    """
    # Base prompt components that stay the same in both conditions
    structure = post.get('structure', 'insightful')
    post_text = post.get('post_text', '')
    emoji_usage = post.get('emoji_usage', 'medium')
    tone = post.get('tone', 'professional, engaging')

    # Only difference is whether we include specific opinion or use generic wording
    if include_opinion and 'opinion' in post:
        opinion = post.get('opinion')
        first_line = f"Create a LinkedIn post on the opinion of: `{opinion}`"
    else:
        # Generic opening without specific opinion
        first_line = f"Create a LinkedIn post"

    # Build the prompt with all features identical except for opinion
    prompt = f"""{first_line}

### Key Message

{post_text}
### Writing Constraints
- **Suggested Post Length**: Between 750 and 1,500 characters long
- **Emoji Usage**: {emoji_usage}
- **Tone**: {tone}
- **Structure**: {structure}
"""

    return prompt

def generate_posts(llm, posts, include_opinion=True, llm_batch=8, temperature=0.7, max_tokens=1000, top_p=0.9):
    """
    Generate LinkedIn posts using the LLM in batches.
    """
    logger.info(f"Generating posts with opinion={'included' if include_opinion else 'excluded'}...")
    prompts = [create_prompt(post, include_opinion) for post in posts]

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

def compute_embedding_similarities(with_opinion_embeddings, without_opinion_embeddings):
    """
    Compute cosine similarities between embeddings
    """
    logger.info("Computing similarities between embeddings...")
    similarities = []

    for i in range(min(len(with_opinion_embeddings), len(without_opinion_embeddings))):
        sim = cosine_similarity(
            with_opinion_embeddings[i].reshape(1, -1),
            without_opinion_embeddings[i].reshape(1, -1)
        )[0][0]
        similarities.append(sim)

    return similarities

def analyze_differences(with_opinion_texts, without_opinion_texts, posts):
    """
    Analyze differences between texts generated with and without opinion
    """
    logger.info("Analyzing text differences...")
    results = []

    for i, (with_t, without_t, post) in enumerate(zip(with_opinion_texts, without_opinion_texts, posts)):
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

        # Get opinion metadata
        opinion = post.get('opinion', 'unknown')

        results.append({
            'index': i,
            'opinion': opinion,
            'with_opinion_length': with_t_length,
            'without_opinion_length': without_t_length,
            'length_diff_pct': length_diff_pct,
            'paragraph_diff': paragraph_diff,
            'bullet_diff': bullet_diff,
            'with_opinion_text': with_t[:200] + "..." if len(with_t) > 200 else with_t,
            'without_opinion_text': without_t[:200] + "..." if len(without_t) > 200 else without_t
        })

    return pd.DataFrame(results)

def categorize_opinion(opinion):
    """
    Categorize opinions into broader groups for visualization
    """
    opinion_lower = opinion.lower()
    if any(kw in opinion_lower for kw in ['marketing', 'sales', 'branding', 'advertising']):
        return 'Marketing & Sales'
    elif any(kw in opinion_lower for kw in ['tech', 'ai', 'software', 'data', 'programming', 'digital']):
        return 'Technology & AI'
    elif any(kw in opinion_lower for kw in ['leadership', 'management', 'career', 'professional']):
        return 'Leadership & Career'
    elif any(kw in opinion_lower for kw in ['health', 'wellness', 'fitness', 'mental']):
        return 'Health & Wellness'
    elif any(kw in opinion_lower for kw in ['finance', 'investing', 'money', 'business']):
        return 'Business & Finance'
    else:
        return 'Other'

def visualize_results(results_df, similarities):
    """
    Visualize the results of the ablation test for opinions
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

    # Add opinion categories
    results_df['opinion_category'] = results_df['opinion'].apply(categorize_opinion)

    # Create figure
    plt.figure(figsize=(16, 14))

    # Plot 1: Overall similarity distribution
    plt.subplot(2, 2, 1)
    sns.histplot(results_df['embedding_similarity'], bins=15, kde=True)
    plt.axvline(results_df['embedding_similarity'].mean(), color='r', linestyle='--',
                label=f'Mean: {results_df["embedding_similarity"].mean():.3f}')
    plt.title('Distribution of Semantic Similarities\nWith vs Without Opinion', fontsize=14)
    plt.xlabel('Cosine Similarity (higher = more similar)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend()

    # Plot 2: Impact (1-similarity) by opinion category
    plt.subplot(2, 2, 2)
    opinion_impact = results_df.groupby('opinion_category')['embedding_similarity'].agg(['mean', 'std', 'count']).reset_index()
    opinion_impact['impact'] = 1 - opinion_impact['mean']
    opinion_impact = opinion_impact.sort_values('impact', ascending=False)

    # Add hue column for color coding
    opinion_impact['opinion_category_hue'] = opinion_impact['opinion_category']

    # Plot with error bars
    sns.barplot(x='opinion_category', y='impact', hue='opinion_category_hue',
                data=opinion_impact, palette='viridis', legend=False, ax=plt.gca())

    # Add error bars
    for i, row in opinion_impact.iterrows():
        plt.errorbar(i, row['impact'], yerr=row['std'], color='black', capsize=5, alpha=0.7)

    plt.title('Opinion Impact by Category\n(Higher = More Change)', fontsize=14)
    plt.xlabel('Opinion Category', fontsize=12)
    plt.ylabel('Impact (1-Similarity)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)

    # Plot 3: Length differences by opinion categories
    plt.subplot(2, 2, 3)
    opinion_length = results_df.groupby('opinion_category')['length_diff_pct'].mean().reset_index()
    opinion_length = opinion_length.sort_values('length_diff_pct', ascending=False)

    # Add hue column
    opinion_length['opinion_category_hue'] = opinion_length['opinion_category']

    sns.barplot(x='opinion_category', y='length_diff_pct', hue='opinion_category_hue',
                data=opinion_length, palette='magma', legend=False)
    plt.title('Text Length Difference by Opinion Category', fontsize=14)
    plt.xlabel('Opinion Category', fontsize=12)
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
    plt.savefig('opinion_ablation_results.png', dpi=300)
    logger.info("Saved visualization to opinion_ablation_results.png")

    # Create and save correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_columns = ['embedding_similarity', 'length_diff_pct', 'paragraph_diff', 'bullet_diff']
    corr_matrix = results_df[corr_columns].corr()

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Between Different Impact Metrics', fontsize=14)
    plt.tight_layout()
    plt.savefig('opinion_ablation_correlations.png', dpi=300)
    logger.info("Saved correlation matrix to opinion_ablation_correlations.png")

def _resolve_input(run_id: str | None, base_dir: str, explicit_input: str | None) -> str:
    if run_id:
        if run_id == "latest":
            rid = get_last_run_id(base_dir)
            if not rid:
                raise ValueError("No .last_run_id found; run previous stages first or set --run-id")
            run_id = rid
        manifest = read_manifest(run_id, base_dir)
        discovered = discover_input(manifest, "12-clean-opinions") or discover_input(manifest, "11-extract-opinion")
        if discovered and os.path.exists(discovered):
            return discovered
        return os.path.join(base_dir, run_id, "12-clean-opinion.jsonl")
    if not explicit_input:
        raise ValueError("Provide --input when --run-id is not used")
    return explicit_input


def run_opinion_ablation_test(run_id: str | None,
                              base_dir: str,
                              input_path: str | None,
                              sample_pct: float,
                              model_name: str,
                              seed: int | None,
                              llm_batch: int,
                              embed_batch: int):
    logger.info("Starting micro ablation test for opinions…")

    set_global_seed(seed)

    # Resolve input and report outputs
    in_path = _resolve_input(run_id, base_dir, input_path)
    reports_dir = os.path.join("reports", run_id or "adhoc", "opinion-ablation")
    os.makedirs(reports_dir, exist_ok=True)

    sig = compute_hash([in_path], {"stage": 13, "sample_pct": sample_pct, "model": model_name, "llm_batch": llm_batch, "embed_batch": embed_batch, "stage_version": STAGE_VERSION})
    if run_id:
        manifest = read_manifest(run_id, base_dir)
        out_paths = [
            os.path.join(reports_dir, "opinion_ablation_results.png"),
            os.path.join(reports_dir, "opinion_ablation_correlations.png"),
            os.path.join(reports_dir, "opinion_ablation_details.txt"),
        ]
        if should_skip(manifest, "13-opinion-ablation", sig, out_paths):
            logger.info(f"Skipping 13-opinion-ablation; up-to-date in {reports_dir}")
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
            if not post.get('post_text') or not post.get('opinion'):
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
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        return

    # Generate with and without opinion
    with_opinion_texts = generate_posts(llm, sample_posts, include_opinion=True, llm_batch=llm_batch)
    without_opinion_texts = generate_posts(llm, sample_posts, include_opinion=False, llm_batch=llm_batch)

    valid_with_texts = [t for t in with_opinion_texts if t.strip()]
    valid_without_texts = [t for t in without_opinion_texts if t.strip()]
    if not valid_with_texts or not valid_without_texts:
        logger.error("No valid texts for embedding!")
        return

    with_opinion_embeddings = compute_embeddings(valid_with_texts, embedding_model, embed_batch=embed_batch)
    without_opinion_embeddings = compute_embeddings(valid_without_texts, embedding_model, embed_batch=embed_batch)

    similarities = []
    if len(with_opinion_embeddings) > 0 and len(without_opinion_embeddings) > 0:
        similarities = compute_embedding_similarities(with_opinion_embeddings, without_opinion_embeddings)

    # Analyze and visualize
    results_df = analyze_differences(with_opinion_texts, without_opinion_texts, sample_posts)
    if len(results_df) == 0:
        logger.error("No valid results to analyze!")
        return

    visualize_results(results_df, similarities)

    mean_similarity = float(np.mean(similarities)) if similarities else 0.0
    overall_impact = 1 - mean_similarity

    # Save detailed results and plots under reports
    details_path = os.path.join(reports_dir, 'opinion_ablation_details.txt')
    with open(details_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("OPINION MICRO ABLATION TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"- Posts analyzed: {len(results_df)}\n")
        f.write(f"- Overall opinion impact: {overall_impact:.4f} (0=none, 1=complete change)\n")
        f.write(f"- Mean similarity between versions: {mean_similarity:.4f}\n")
        f.write(f"- Mean text length difference: {results_df['length_diff_pct'].mean():.2f}%\n\n")

        results_df['opinion_category'] = results_df['opinion'].apply(categorize_opinion)
        f.write("Impact by opinion category:\n")
        opinion_impact_df = results_df.groupby('opinion_category')['embedding_similarity'].agg(['mean', 'std', 'count']).reset_index()
        opinion_impact_df['impact'] = 1 - opinion_impact_df['mean']
        opinion_impact_df = opinion_impact_df.sort_values('impact', ascending=False)
        for _, row in opinion_impact_df.iterrows():
            f.write(f"- {row['opinion_category']}: {row['impact']:.4f} (samples: {row['count']})\n")

        f.write("\n\n" + "=" * 80 + "\n")
        f.write("EXAMPLE COMPARISONS\n")
        f.write("=" * 80 + "\n\n")
        if 'embedding_similarity' in results_df.columns and len(results_df) > 0:
            most_different = results_df.sort_values('embedding_similarity').head(5)
            for _, row in most_different.iterrows():
                f.write(f"Example (opinion: {row['opinion']}, similarity: {row['embedding_similarity']:.4f}):\n\n")
                with_text = with_opinion_texts[row['index']] if row['index'] < len(with_opinion_texts) else "N/A"
                without_text = without_opinion_texts[row['index']] if row['index'] < len(without_opinion_texts) else "N/A"
                f.write("WITH OPINION:\n" + with_text + "\n\n")
                f.write("WITHOUT OPINION:\n" + without_text + "\n\n")
                f.write("-" * 80 + "\n\n")

    # Save figures to reports
    plt.savefig(os.path.join(reports_dir, 'opinion_ablation_results.png'), dpi=300)
    plt.close('all')
    plt.savefig(os.path.join(reports_dir, 'opinion_ablation_correlations.png'), dpi=300)

    logger.info(f"Saved opinion ablation artifacts to {reports_dir}")

    if run_id:
        manifest = read_manifest(run_id, base_dir)
        update_stage(run_id, base_dir, manifest, "13-opinion-ablation", in_path, [
            os.path.join(reports_dir, 'opinion_ablation_results.png'),
            os.path.join(reports_dir, 'opinion_ablation_correlations.png'),
            details_path,
        ], sig, extra={"sampled": len(sample_posts), "eligible": eligible, "mean_similarity": mean_similarity})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Opinion micro-ablation using 12-clean-opinion output")
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
        run_opinion_ablation_test(
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
        logger.error(f"Error in opinion micro ablation test: {e}", exc_info=True)
        print(f"Test failed with error: {e}")