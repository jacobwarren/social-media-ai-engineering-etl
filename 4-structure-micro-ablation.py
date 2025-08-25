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
from utils.version import STAGE_VERSION

# Configure logging
logger = init_pipeline_logging("phase2.ablation", None, "04-structure-micro-ablation")

# Defaults (can be overridden via CLI)
INPUT_FILE = None  # resolved from manifest when --run-id provided
BASE_DIR = "data/processed"
RUN_ID = None
SAMPLE_PERCENTAGE = 0.20  # Test 20% of the dataset
MODEL_NAME = "Qwen/Qwen3-32B"
MIN_SAMPLE_SIZE = 5
MAX_SAMPLE_SIZE = 20

def create_prompt(post, include_structure=True):
    """
    Create prompt with or without structure feature
    """
    # Base prompt always includes topic
    topic = post.get('topic', 'Professional Development')
    # Try to extract topic from post_text if not available
    if topic == 'Professional Development' and 'post_text' in post:
        # Simple heuristic: first sentence might contain the topic
        first_sentence = post['post_text'].split('.')[0]
        if len(first_sentence) < 100:  # Reasonable topic length
            topic = first_sentence
    
    base_content = f"""Create a LinkedIn post on the topic of: `{topic}`

### Key Message
```
{post.get('post_text', '')}
```
### Writing Constraints
- **Suggested Post Length**: Between 750 and 1,500 characters long
- **Emoji Usage**: {post.get('emoji_usage', 'medium')}
- **Tone**: {post.get('tone', 'professional, engaging')}
"""

    # Add structure if requested
    if include_structure and 'structure' in post:
        structure = post['structure']
        if structure == 'instructional':
            style_note = "that provides step-by-step advice"
        elif structure == 'inspirational':
            style_note = "that inspires and motivates"
        elif structure == 'analytical':
            style_note = "that analyzes data or trends"
        elif structure == 'controversial':
            style_note = "that challenges conventional wisdom"
        elif structure == 'insightful':
            style_note = "that offers unique insights"
        elif structure == 'comparative':
            style_note = "that compares different approaches"
        elif structure == 'reflective':
            style_note = "that reflects on experiences"
        elif structure == 'evolutionary':
            style_note = "that shows contrast between past and present"
        elif structure == 'announcement':
            style_note = "that announces something exciting"
        else:
            style_note = ""
            
        # Update first line with structure if we have a style note
        if style_note:
            first_line = f"Create a LinkedIn post {style_note} on the topic of: `{topic}`"
            base_content = base_content.replace(base_content.split('\n')[0], first_line)

    # Wrap in chat template for Qwen-style chat models
    return f"<|im_start|>user\n{base_content}<|im_end|>\n<|im_start|>assistant\n"

def generate_posts(llm, posts, include_structure=True, llm_batch=8, temperature=0.7, max_tokens=1000, top_p=0.9):
    """
    Generate LinkedIn posts using the LLM in batches.
    """
    logger.info(f"Generating posts with structure={'included' if include_structure else 'excluded'}...")
    prompts = [create_prompt(post, include_structure) for post in posts]

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

def compute_embedding_similarities(with_structure_embeddings, without_structure_embeddings):
    """
    Compute cosine similarities between embeddings
    """
    logger.info("Computing similarities between embeddings...")
    similarities = []
    
    for i in range(min(len(with_structure_embeddings), len(without_structure_embeddings))):
        sim = cosine_similarity(
            with_structure_embeddings[i].reshape(1, -1),
            without_structure_embeddings[i].reshape(1, -1)
        )[0][0]
        similarities.append(sim)
    
    return similarities

def analyze_differences(with_structure_texts, without_structure_texts, posts):
    """
    Analyze differences between texts generated with and without structure
    """
    logger.info("Analyzing text differences...")
    results = []
    
    for i, (with_s, without_s, post) in enumerate(zip(with_structure_texts, without_structure_texts, posts)):
        # Skip empty generations
        if not with_s.strip() or not without_s.strip():
            continue
            
        # Basic text statistics
        with_s_length = len(with_s)
        without_s_length = len(without_s)
        length_diff = abs(with_s_length - without_s_length)
        length_diff_pct = length_diff / max(with_s_length, without_s_length) * 100
        
        # Calculate number of paragraphs
        with_s_paragraphs = len([p for p in with_s.split('\n\n') if p.strip()])
        without_s_paragraphs = len([p for p in without_s.split('\n\n') if p.strip()])
        paragraph_diff = abs(with_s_paragraphs - without_s_paragraphs)
        
        # Count bullet points (simplified approach)
        with_s_bullets = with_s.count('- ')
        without_s_bullets = without_s.count('- ')
        bullet_diff = abs(with_s_bullets - without_s_bullets)
        
        # Get structure type
        structure_type = post.get('structure', 'unknown')
        
        results.append({
            'index': i,
            'structure': structure_type,
            'with_structure_length': with_s_length,
            'without_structure_length': without_s_length,
            'length_diff_pct': length_diff_pct,
            'paragraph_diff': paragraph_diff,
            'bullet_diff': bullet_diff,
            'with_structure_text': with_s[:200] + "..." if len(with_s) > 200 else with_s,
            'without_structure_text': without_s[:200] + "..." if len(without_s) > 200 else without_s
        })
    
    return pd.DataFrame(results)

def visualize_results(results_df, similarities):
    """
    Visualize the results of the ablation test
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
    
    # Create figure
    plt.figure(figsize=(16, 14))
    
    # Plot 1: Overall similarity distribution
    plt.subplot(2, 2, 1)
    sns.histplot(results_df['embedding_similarity'], bins=15, kde=True)
    plt.axvline(results_df['embedding_similarity'].mean(), color='r', linestyle='--', 
                label=f'Mean: {results_df["embedding_similarity"].mean():.3f}')
    plt.title('Distribution of Semantic Similarities\nWith vs Without Structure', fontsize=14)
    plt.xlabel('Cosine Similarity (higher = more similar)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend()
    
    # Plot 2: Similarity by structure type
    plt.subplot(2, 2, 2)
    structure_impact_df = results_df.groupby('structure')['embedding_similarity'].agg(['mean', 'std', 'count']).reset_index()
    structure_impact_df = structure_impact_df.sort_values('mean')
    
    # Create hue column for color coding
    structure_impact_df['structure_hue'] = structure_impact_df['structure']
    
    # Plot with error bars
    sns.barplot(x='structure', y='mean', hue='structure_hue', data=structure_impact_df, 
                palette='viridis', legend=False, ax=plt.gca())
    
    # Add error bars
    for i, row in structure_impact_df.iterrows():
        plt.errorbar(i, row['mean'], yerr=row['std'], color='black', capsize=5, alpha=0.7)
    
    plt.title('Impact of Structure by Type\n(Lower Similarity = Bigger Impact)', fontsize=14)
    plt.xlabel('Structure Type', fontsize=12)
    plt.ylabel('Mean Similarity', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    
    # Plot 3: Length differences by structure type
    plt.subplot(2, 2, 3)
    structure_length = results_df.groupby('structure')['length_diff_pct'].mean().reset_index()
    structure_length = structure_length.sort_values('length_diff_pct', ascending=False)
    
    # Add hue column
    structure_length['structure_hue'] = structure_length['structure']
    
    sns.barplot(x='structure', y='length_diff_pct', hue='structure_hue', 
                data=structure_length, palette='magma', legend=False)
    plt.title('Text Length Difference by Structure Type', fontsize=14)
    plt.xlabel('Structure Type', fontsize=12)
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
    plt.savefig('micro_ablation_results.png', dpi=300)
    logger.info("Saved visualization to micro_ablation_results.png")
    
    # Create and save correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_columns = ['embedding_similarity', 'length_diff_pct', 'paragraph_diff', 'bullet_diff']
    corr_matrix = results_df[corr_columns].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Between Different Impact Metrics', fontsize=14)
    plt.tight_layout()
    plt.savefig('micro_ablation_correlations.png', dpi=300)
    logger.info("Saved correlation matrix to micro_ablation_correlations.png")

def resolve_input_path() -> str:
    global INPUT_FILE, RUN_ID, BASE_DIR
    if RUN_ID:
        from utils.run_id import get_last_run_id
        if RUN_ID == "latest":
            rid = get_last_run_id(BASE_DIR)
            if not rid:
                raise ValueError("No .last_run_id found; run previous stages first or set --run-id")
            RUN_ID = rid
        # Prefer manifest if available
        manifest = read_manifest(RUN_ID, BASE_DIR)
        discovered = discover_input(manifest, "03-structures")
        if discovered and os.path.exists(discovered):
            return discovered
        # Fallback to standard path
        return os.path.join(BASE_DIR, RUN_ID, "03-structures.jsonl")
    # No run id: require explicit input
    if not INPUT_FILE:
        raise ValueError("Provide --input when --run-id is not used")
    return INPUT_FILE


def run_ablation_test(run_id: str | None,
                      base_dir: str,
                      input_path: str | None,
                      sample_pct: float,
                      model_name: str,
                      seed: int | None,
                      llm_batch: int,
                      embed_batch: int) -> None:
    logger.info("Starting micro ablation test...")

    global RUN_ID, BASE_DIR, INPUT_FILE, SAMPLE_PERCENTAGE, MODEL_NAME
    RUN_ID, BASE_DIR, INPUT_FILE = run_id, base_dir, input_path
    SAMPLE_PERCENTAGE, MODEL_NAME = sample_pct, model_name

    set_global_seed(seed)

    # Resolve input
    input_path = resolve_input_path()

    # Report directory and manifest skip
    reports_dir = os.path.join("reports", RUN_ID or "adhoc", "ablation")
    os.makedirs(reports_dir, exist_ok=True)

    sig = compute_hash([input_path], {"stage": 4, "sample_pct": sample_pct, "model": model_name, "llm_batch": llm_batch, "embed_batch": embed_batch, "stage_version": STAGE_VERSION})
    if RUN_ID:
        manifest = read_manifest(RUN_ID, BASE_DIR)
        out_paths = [
            os.path.join(reports_dir, "micro_ablation_results.png"),
            os.path.join(reports_dir, "micro_ablation_correlations.png"),
            os.path.join(reports_dir, "micro_ablation_details.txt"),
        ]
        if should_skip(manifest, "04-structure-ablation", sig, out_paths):
            logger.info(f"Skipping 04-structure-ablation; up-to-date in {reports_dir}")
            return

    # Streaming sample selection (reservoir)
    eligible = 0
    k = 0
    sample_posts: List[dict] = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                post = json.loads(line)
            except Exception:
                continue
            if not post.get('post_text') or 'structure' not in post:
                continue
            eligible += 1
            # decide sample size after first pass? use running k via desired = int(eligible*sample_pct)
            desired = max(MIN_SAMPLE_SIZE, min(MAX_SAMPLE_SIZE, int(eligible * sample_pct)))
            if len(sample_posts) < desired:
                sample_posts.append(post)
            else:
                # Reservoir replacement
                j = random.randint(0, eligible - 1)
                if j < desired:
                    sample_posts[j] = post
            k = desired

    logger.info(f"Eligible posts: {eligible}, sampled: {len(sample_posts)} (target {k})")

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

    logger.info("Loading embedding model...")
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        return

    # Generate posts with and without structure
    with_structure_texts = generate_posts(llm, sample_posts, include_structure=True, llm_batch=llm_batch)
    without_structure_texts = generate_posts(llm, sample_posts, include_structure=False, llm_batch=llm_batch)

    # Compute embeddings if we have text
    valid_with_texts = [t for t in with_structure_texts if t.strip()]
    valid_without_texts = [t for t in without_structure_texts if t.strip()]

    if not valid_with_texts or not valid_without_texts:
        logger.error("No valid texts for embedding!")
        return

    with_structure_embeddings = compute_embeddings(valid_with_texts, embedding_model, embed_batch=embed_batch)
    without_structure_embeddings = compute_embeddings(valid_without_texts, embedding_model, embed_batch=embed_batch)

    similarities = []
    if len(with_structure_embeddings) > 0 and len(without_structure_embeddings) > 0:
        similarities = compute_embedding_similarities(with_structure_embeddings, without_structure_embeddings)

    # Analyze and visualize
    logger.info("Analyzing differences...")
    results_df = analyze_differences(with_structure_texts, without_structure_texts, sample_posts)

    if len(results_df) == 0:
        logger.error("No valid results to analyze!")
        return

    logger.info("Visualizing results...")
    visualize_results(results_df, similarities)

    # Calculate and display overall impact
    mean_similarity = float(np.mean(similarities)) if similarities else 0.0
    overall_impact = 1 - mean_similarity

    # Save detailed results and plots into reports_dir
    details_path = os.path.join(reports_dir, 'micro_ablation_details.txt')
    with open(details_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MICRO ABLATION TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Test summary:\n")
        f.write(f"- Posts analyzed: {len(results_df)}\n")
        f.write(f"- Overall structure impact: {overall_impact:.4f} (0=none, 1=complete change)\n")
        f.write(f"- Mean similarity between versions: {mean_similarity:.4f}\n")
        f.write(f"- Mean text length difference: {results_df['length_diff_pct'].mean():.2f}%\n\n")

        f.write("Impact by structure type:\n")
        structure_impact_df = results_df.groupby('structure')['embedding_similarity'].agg(['mean', 'std', 'count']).reset_index()
        structure_impact_df['impact'] = 1 - structure_impact_df['mean']
        structure_impact_df = structure_impact_df.sort_values('impact', ascending=False)

        for i, row in structure_impact_df.iterrows():
            f.write(f"- {row['structure']}: {row['impact']:.4f} (samples: {row['count']})\n")

        f.write("\n\n" + "=" * 80 + "\n")
        f.write("EXAMPLE COMPARISONS\n")
        f.write("=" * 80 + "\n\n")

        if 'embedding_similarity' in results_df.columns and len(results_df) > 0:
            most_different = results_df.sort_values('embedding_similarity').head(5)
            for i, row in most_different.iterrows():
                f.write(f"Example {i+1} (structure: {row['structure']}, similarity: {row['embedding_similarity']:.4f}):\n\n")
                f.write("WITH STRUCTURE:\n")
                with_text = with_structure_texts[row['index']] if row['index'] < len(with_structure_texts) else "N/A"
                f.write(with_text)
                f.write("\n\n")
                f.write("WITHOUT STRUCTURE:\n")
                without_text = without_structure_texts[row['index']] if row['index'] < len(without_structure_texts) else "N/A"
                f.write(without_text)
                f.write("\n\n")
                f.write("-" * 80 + "\n\n")

    # Move plots to reports_dir by saving there
    plt.figure()
    # visualize_results already saves plots in current version; we re-run saving under reports_dir
    # Recreate figures via visualize_results
    visualize_results(results_df, similarities)
    plt.savefig(os.path.join(reports_dir, 'micro_ablation_results.png'), dpi=300)
    plt.close('all')

    # Heatmap saved by visualize_results; save again under reports_dir
    # To ensure both images exist even if internals change
    # The function itself saves to filenames; we keep this as redundancy

    logger.info(f"Saved ablation artifacts to {reports_dir}")

    # Update manifest
    if RUN_ID:
        manifest = read_manifest(RUN_ID, BASE_DIR)
        update_stage(RUN_ID, BASE_DIR, manifest, "04-structure-ablation", input_path, [
            os.path.join(reports_dir, 'micro_ablation_results.png'),
            os.path.join(reports_dir, 'micro_ablation_correlations.png'),
            details_path,
        ], sig, extra={"sampled": len(sample_posts), "eligible": eligible, "mean_similarity": mean_similarity})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Structure micro-ablation using 03-structures output")
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
        run_ablation_test(
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
        logger.error(f"Error in micro ablation test: {e}", exc_info=True)
        print(f"Test failed with error: {e}")