import json
import os
import random
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Compatibility bootstrap: allow running this file directly or as part of the package
if __package__ is None or __package__ == "":
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from pipe.utils.logging_setup import init_pipeline_logging
from pipe.utils.seed import set_global_seed
from pipe.utils.manifest import read_manifest, discover_input, compute_hash, should_skip, update_stage
from pipe.utils.run_id import get_last_run_id

# Configure logging
logger = init_pipeline_logging("phase2.scenario_tests", None, "19-scenario-tests")

# Defaults (overridable via CLI)
RUN_ID = None
BASE_DIR = "data/processed"
INPUT_FILE = None  # resolved from manifest when run-id provided
SAMPLE_PERCENTAGE = 0.20
MODEL_NAME = "RekaAI/reka-flash-3"
MIN_SAMPLE_SIZE = 5
MAX_SAMPLE_SIZE = 20

# Will be set in main
REPORTS_DIR = "scenario_test_results"

# Define feature categories for testing
FEATURES_TO_TEST = {
    'structure': {
        'instructional': "that provides step-by-step advice",
        'inspirational': "that inspires and motivates",
        'analytical': "that analyzes data or trends",
        'controversial': "that challenges conventional wisdom",
        'insightful': "that offers unique insights",
        'comparative': "that compares different approaches",
        'reflective': "that reflects on experiences",
        'announcement': "that announces something exciting"
    },
    'tone': {
        'professional': "professional, formal",
        'casual': "casual, conversational",
        'inspiring': "inspiring, motivational",
        'witty': "witty, humorous",
        'thoughtful': "thoughtful, reflective",
        'authoritative': "authoritative, expert",
        'friendly': "friendly, approachable"
    },
    'emoji_usage': {
        'high': "high",
        'medium': "medium",
        'low': "low",
        'none': "none"
    },
    'pacing': {
        'Fast': "Fast",
        'Slow': "Slow",
        'Moderate': "Moderate",
        'Dynamic': "Dynamic",
        'Variable': "Variable"
    },
    'sentiment_arc': {
        'Upward Trend': "Upward Trend",
        'Downward Trend': "Downward Trend",
        'Stable': "Stable",
        'Complex/Variable': "Complex/Variable"
    },
    'narrative_flow': {
        'Introduction/Setup': "A narrative that starts by introducing the subject",
        'Conflict/Resolution': "A narrative that focuses on highlighting a problem and its solution",
        'Introduction/Development': "A narrative that introduces and expands on key concepts",
        'Transition/Reflection': "A narrative that shifts into reflection or transitions between ideas"
    }
}

def modify_prompt(prompt, feature, original_value, new_value):
    """
    Modify a specific feature in the given prompt.
    Returns the modified prompt text.
    """
    if prompt is None:
        return None

    # Split the prompt into lines for easier editing
    lines = prompt.split('\n')
    modified_lines = []

    # Feature-specific modifications
    if feature == 'structure':
        # Handle structure change
        for i, line in enumerate(lines):
            if i == 0 and "Create a LinkedIn post" in line:
                # Replace the structure description in the first line
                original_desc = FEATURES_TO_TEST['structure'].get(original_value, "")
                new_desc = FEATURES_TO_TEST['structure'].get(new_value, "")

                if original_desc and new_desc:
                    modified_line = line.replace(original_desc, new_desc)
                    modified_lines.append(modified_line)
                else:
                    # If we don't find the exact structure, do our best to replace
                    for desc in FEATURES_TO_TEST['structure'].values():
                        if desc in line:
                            modified_line = line.replace(desc, new_desc)
                            modified_lines.append(modified_line)
                            break
                    else:
                        modified_lines.append(line)  # No replacement found
            else:
                modified_lines.append(line)

    elif feature == 'tone':
        # Handle tone change
        for line in lines:
            if "**Tone**:" in line:
                modified_line = f"- **Tone**: {new_value}"
                modified_lines.append(modified_line)
            else:
                modified_lines.append(line)

    elif feature == 'emoji_usage':
        # Handle emoji usage change
        for line in lines:
            if "**Emoji Usage**:" in line:
                modified_line = f"- **Emoji Usage**: {new_value}"
                modified_lines.append(modified_line)
            else:
                modified_lines.append(line)

    elif feature == 'pacing':
        # Handle pacing change
        for line in lines:
            if "**Pacing**:" in line:
                modified_line = f"- **Pacing**: {new_value}"
                modified_lines.append(modified_line)
            else:
                modified_lines.append(line)

    elif feature == 'sentiment_arc':
        # Handle sentiment arc change
        for line in lines:
            if "**Sentiment Arc**:" in line:
                modified_line = f"- **Sentiment Arc**: {new_value}"
                modified_lines.append(modified_line)
            else:
                modified_lines.append(line)

    elif feature == 'narrative_flow':
        # Handle narrative flow change
        for line in lines:
            if "**Narrative Flow**:" in line:
                modified_line = f"- **Narrative Flow**: {new_value}"
                modified_lines.append(modified_line)
            else:
                modified_lines.append(line)

def _resolve_input(run_id: str | None, base_dir: str, explicit_input: str | None) -> Tuple[str, str | None]:
    reports_dir = os.path.join("reports", run_id or "adhoc", "scenario-tests")
    os.makedirs(reports_dir, exist_ok=True)

    if run_id:
        if run_id == "latest":
            rid = get_last_run_id(base_dir)
            if not rid:
                raise ValueError("No .last_run_id found; run previous stages first or set --run-id")
            run_id = rid
        manifest = read_manifest(run_id, base_dir)
        discovered = discover_input(manifest, "18-prompts") or discover_input(manifest, "17-writing-style")
        in_path = discovered or explicit_input
        if not in_path:
            raise ValueError("No input found: provide --input or ensure manifest contains 18-prompts output")
        return in_path, reports_dir
    else:
        if not explicit_input:
            raise ValueError("Provide --input when --run-id is not used")
        return explicit_input, reports_dir

                modified_lines.append(modified_line)
            else:
                modified_lines.append(line)

    # If no modifications were made or unknown feature, return original prompt
    if not modified_lines:
        return prompt

    # Join modified lines back into a single prompt
    return '\n'.join(modified_lines)

def run_scenario_test(llm, embedding_model, post, feature, original_value, new_value):
    """
    Run a single scenario test, comparing original feature value vs new value
    """
    # Get original prompt
    original_prompt = post.get('prompt', '')
    if not original_prompt:
        return {'success': False, 'error': 'No prompt found in post'}

    # Create modified prompt with new feature value
    modified_prompt = modify_prompt(original_prompt, feature, original_value, new_value)
    if not modified_prompt or modified_prompt == original_prompt:
        return {'success': False, 'error': 'Failed to modify prompt'}

    # Generate content with both prompts
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=1000,
        top_p=0.9,
    )

    try:



def reservoir_sample(path: str, fraction: float, min_n: int, max_n: int, predicate=None):
    eligible = 0
    sample = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                post = json.loads(line)
            except Exception:
                continue
            if predicate and not predicate(post):
                continue
            eligible += 1
            desired = max(min_n, min(max_n, int(eligible * fraction)))
            if len(sample) < desired:
                sample.append(post)
            else:
                j = random.randint(0, eligible - 1)
                if j < desired:
                    sample[j] = post
    return sample, eligible

        top_p=0.9,
    )

    try:
        outputs = llm.generate([original_prompt, modified_prompt], sampling_params)
        original_text = outputs[0].outputs[0].text
        modified_text = outputs[1].outputs[0].text

        # Compute embeddings and similarity
        embeddings = embedding_model.encode([original_text, modified_text])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        # Compute text stats
        original_length = len(original_text)
        modified_length = len(modified_text)
        length_diff_pct = abs(original_length - modified_length) / max(original_length, modified_length) * 100

        return {
            'feature': feature,
            'original_value': original_value,
            'new_value': new_value,
            'embedding_similarity': similarity,
            'length_diff_pct': length_diff_pct,
            'original_text': original_text,
            'modified_text': modified_text,
            'original_length': original_length,
            'modified_length': modified_length,
            'original_prompt': original_prompt,
            'modified_prompt': modified_prompt,
            'success': True
        }
    except Exception as e:
        logger.error(f"Error generating content: {e}")
        return {'success': False, 'error': str(e)}

def run_tests():
    """
    Main function to run scenario-based tests for multiple features
    """
    # Create output directory
    os.makedirs('scenario_test_results', exist_ok=True)

    # Load posts with all features
    posts = []
    try:
        with open(INPUT_FILE, 'r') as f:
            for line in f:
                try:
                    posts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        logger.error(f"Input file {INPUT_FILE} not found!")
        return

    logger.info(f"Loaded {len(posts)} posts from {INPUT_FILE}")

    # Filter out posts without prompt
    valid_posts = [post for post in posts if post.get('prompt')]
    logger.info(f"Found {len(valid_posts)} posts with prompts")

    # Sample posts for testing
    sample_size = max(MIN_SAMPLE_SIZE, min(MAX_SAMPLE_SIZE, int(len(valid_posts) * SAMPLE_PERCENTAGE)))
    sample_posts = random.sample(valid_posts, sample_size) if len(valid_posts) > sample_size else valid_posts

    logger.info(f"Running scenario tests with {len(sample_posts)} posts...")

    # Initialize models
    logger.info("Loading LLM...")
    llm = LLM(model=MODEL_NAME, max_model_len=8192)

    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Define which features to test for each post
    features_to_test = list(FEATURES_TO_TEST.keys())[:3]  # Start with top 3 features
    logger.info(f"Testing features: {features_to_test}")

    # Distribute feature tests among posts
    test_assignments = []
    for i, post in enumerate(sample_posts):
        feature_idx = i % len(features_to_test)
        feature = features_to_test[feature_idx]

        # Get current value for this feature
        current_value = None
        if feature == 'structure':
            current_value = post.get('structure')
        elif feature == 'tone':
            current_value = post.get('tone')
        elif feature == 'emoji_usage':
            current_value = post.get('emoji_usage')
        elif feature == 'pacing':
            current_value = post.get('pacing')
        elif feature == 'sentiment_arc':
            current_value = post.get('sentiment_arc')
        elif feature == 'narrative_flow':
            flow = post.get('flow', [])
            if flow and len(flow) > 0:
                current_value = flow[0]  # Use first flow element

        # Skip if feature value not found
        if not current_value:
            continue

        # Get possible values to test against
        possible_values = list(FEATURES_TO_TEST[feature].keys())
        alternate_values = [v for v in possible_values if v != current_value]

        if not alternate_values:
            continue

        # Choose a random alternate value
        alternate_value = random.choice(alternate_values)

        test_assignments.append({
            'post': post,
            'feature': feature,
            'current_value': current_value,
            'alternate_value': alternate_value
        })

    # Run the assigned tests
    results = []
    for assignment in tqdm(test_assignments, desc="Running scenario tests"):
        result = run_scenario_test(
            llm,
            embedding_model,
            assignment['post'],
            assignment['feature'],
            assignment['current_value'],
            assignment['alternate_value']
        )

        if result['success']:
            results.append(result)

    if not results:
        logger.error("No successful tests completed!")
        return

    # Convert results to DataFrame
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ['original_text', 'modified_text', 'original_prompt', 'modified_prompt']}
        for r in results
    ])

    # Group results by feature for analysis
    feature_results = {}
    for feature in features_to_test:
        feature_data = results_df[results_df['feature'] == feature]
        if len(feature_data) > 0:
            feature_results[feature] = feature_data

    # Create visualizations for each feature
    create_feature_visualizations(feature_results)

    # Create overall impact summary
    create_impact_summary(results_df)

    # Save detailed examples to text files
    save_detailed_examples(results)

    logger.info("Scenario tests completed successfully!")

def create_feature_visualizations(feature_results):
    """
    Create visualizations for each feature showing the impact of transitions
    """
    for feature, data in feature_results.items():
        plt.figure(figsize=(15, 10))

        # Create a transition label for each row
        data['transition'] = data.apply(
            lambda x: f"{x['original_value']} → {x['new_value']}", axis=1
        )

        # Group by transition and calculate mean similarity
        transition_impact = data.groupby('transition')['embedding_similarity'].agg(
            ['mean', 'std', 'count']
        ).reset_index()

        # Sort by impact (lower similarity = higher impact)
        transition_impact = transition_impact.sort_values('mean')

        # Plot with colored bars based on impact level
        plt.figure(figsize=(12, 8))
        bars = plt.barh(
            transition_impact['transition'],
            1 - transition_impact['mean'],  # Convert to impact score
            xerr=transition_impact['std'],
            color=plt.cm.viridis(np.linspace(0, 1, len(transition_impact)))
        )

        # Add count annotations
        for i, (_, row) in enumerate(transition_impact.iterrows()):
            plt.text(
                1 - row['mean'] + 0.02,
                i,
                f"n={row['count']}",
                va='center'
            )

        plt.title(f'Impact of {feature.capitalize()} Transitions', fontsize=16)
        plt.xlabel('Impact Score (higher = more change)', fontsize=14)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.xlim(0, 1)

        plt.tight_layout()
        plt.savefig(f'scenario_test_results/{feature}_transitions.png', dpi=300)
        plt.close()

        logger.info(f"Created visualization for {feature}")

def create_impact_summary(results_df):
    """
    Create summary visualization showing impact of each feature
    """
    # Calculate average impact by feature
    feature_impact = results_df.groupby('feature').agg({
        'embedding_similarity': ['mean', 'std', 'count'],
        'length_diff_pct': ['mean', 'std']
    }).reset_index()

    # Flatten multi-level columns
    feature_impact.columns = [
        '_'.join(col).strip('_') for col in feature_impact.columns.values
    ]

    # Calculate impact score (1 - similarity)
    feature_impact['impact_score'] = 1 - feature_impact['embedding_similarity_mean']

    # Sort by impact score
    feature_impact = feature_impact.sort_values('impact_score', ascending=False)

    # Create visualization
    plt.figure(figsize=(12, 8))

    # Create bar chart with error bars
    bars = plt.bar(
        feature_impact['feature'],
        feature_impact['impact_score'],
        yerr=feature_impact['embedding_similarity_std'],
        color=plt.cm.viridis(np.linspace(0, 1, len(feature_impact)))
    )

    # Add count annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        count = feature_impact.iloc[i]['embedding_similarity_count']
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.02,
            f"n={count}",
            ha='center'
        )

    plt.title('Feature Impact Comparison', fontsize=16)
    plt.ylabel('Impact Score (higher = more change)', fontsize=14)
    plt.xlabel('Feature', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig('scenario_test_results/feature_impact_comparison.png', dpi=300)
    plt.close()

    # Create a second visualization showing length differences
    plt.figure(figsize=(12, 8))
    bars = plt.bar(
        feature_impact['feature'],
        feature_impact['length_diff_pct_mean'],
        yerr=feature_impact['length_diff_pct_std'],
        color=plt.cm.plasma(np.linspace(0, 1, len(feature_impact)))
    )

    plt.title('Effect on Content Length by Feature', fontsize=16)
    plt.ylabel('Average Length Difference (%)', fontsize=14)
    plt.xlabel('Feature', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig('scenario_test_results/length_differences_by_feature.png', dpi=300)
    plt.close()

    # Save numerical results to CSV
    feature_impact.to_csv('scenario_test_results/feature_impact_summary.csv', index=False)

    logger.info("Created feature impact summary visualizations")

def save_detailed_examples(results):
    """
    Save detailed examples of the most impactful transitions for each feature
    """
    # Create a nested dictionary to organize results by feature
    feature_examples = defaultdict(list)

    # Add results to appropriate feature
    for result in results:
        feature = result['feature']
        feature_examples[feature].append(result)

    # For each feature, sort examples by impact and save the most significant ones
    for feature, examples in feature_examples.items():
        # Sort by similarity (lower = higher impact)
        sorted_examples = sorted(examples, key=lambda x: x['embedding_similarity'])

        # Take top examples (most different)
        top_examples = sorted_examples[:min(3, len(sorted_examples))]

        # Write to file
        with open(f'scenario_test_results/{feature}_examples.txt', 'w') as f:
            f.write(f"=== Most Impactful {feature.capitalize()} Transitions ===\n\n")

            for i, example in enumerate(top_examples):
                f.write(f"Example {i+1}: {example['original_value']} → {example['new_value']}\n")
                f.write(f"Similarity: {example['embedding_similarity']:.4f}, Impact: {1-example['embedding_similarity']:.4f}\n")
                f.write(f"Length Difference: {example['length_diff_pct']:.2f}%\n\n")

                f.write("ORIGINAL PROMPT:\n")
                f.write(f"{example['original_prompt']}\n\n")

                f.write("MODIFIED PROMPT:\n")
                f.write(f"{example['modified_prompt']}\n\n")

                f.write("ORIGINAL OUTPUT:\n")
                f.write(f"{example['original_text']}\n\n")

                f.write("MODIFIED OUTPUT:\n")
                f.write(f"{example['modified_text']}\n\n")

                f.write("="*80 + "\n\n")

    # Create summary file (written under current working dir, which main() sets to reports_dir)
    with open('scenario_test_results/scenario_test_summary.txt', 'w') as f:
        f.write("=== SCENARIO-BASED FEATURE TESTS SUMMARY ===\n\n")



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scenario-based prompt micro tests")
    parser.add_argument("--run-id", dest="run_id", default=None, help="Use 'latest' to use most recent run")
    parser.add_argument("--base-dir", dest="base_dir", default="data/processed")
    parser.add_argument("--input", dest="input_path", default=None, help="Only used when --run-id is not provided")
    parser.add_argument("--sample", dest="sample_pct", type=float, default=SAMPLE_PERCENTAGE)
    parser.add_argument("--model", dest="model_name", default=MODEL_NAME)
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("--llm-batch", dest="llm_batch", type=int, default=8)
    parser.add_argument("--embed-batch", dest="embed_batch", type=int, default=64)
    args = parser.parse_args()

    set_global_seed(args.seed)

    in_path, reports_dir = _resolve_input(args.run_id, args.base_dir, args.input_path)

    # Idempotent skip based on signature
    sig = compute_hash([in_path], {"stage": 19, "sample_pct": args.sample_pct, "model": args.model_name, "llm_batch": args.llm_batch, "embed_batch": args.embed_batch})
    if args.run_id:
        manifest = read_manifest(args.run_id, args.base_dir)
        out_paths = [
            os.path.join(reports_dir, "feature_impact_comparison.png"),
            os.path.join(reports_dir, "length_differences_by_feature.png"),
            os.path.join(reports_dir, "feature_impact_summary.csv"),
            os.path.join(reports_dir, "scenario_test_summary.txt"),
        ]
        if should_skip(manifest, "19-scenario-tests", sig, out_paths):
            logger.info(f"Skipping 19-scenario-tests; up-to-date in {reports_dir}")
            return

    # Sample posts with prompts only
    def has_prompt(post):
        return bool(post.get('prompt'))

    sample_posts, eligible = reservoir_sample(in_path, args.sample_pct, MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE, predicate=has_prompt)

    logger.info(f"Eligible posts: {eligible}, sampled: {len(sample_posts)}")
    if not sample_posts:
        logger.error("No samples; aborting")
        return

    # Initialize models
    logger.info(f"Loading LLM: {args.model_name}")
    llm = LLM(model=args.model_name, max_model_len=8192)
    logger.info("Loading embedding model…")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Select features to test
    features_to_test = list(FEATURES_TO_TEST.keys())[:3]

    # Distribute tests
    test_assignments = []
    for i, post in enumerate(sample_posts):
        feature_idx = i % len(features_to_test)
        feature = features_to_test[feature_idx]
        current_value = None
        if feature == 'structure':
            current_value = post.get('structure')
        elif feature == 'tone':
            current_value = post.get('tone')
        elif feature == 'emoji_usage':
            current_value = post.get('emoji_usage')
        elif feature == 'pacing':
            current_value = post.get('pacing')
        elif feature == 'sentiment_arc':
            current_value = post.get('sentiment_arc')
        elif feature == 'narrative_flow':
            flow = post.get('flow', [])
            if flow:
                current_value = flow[0]
        if not current_value:
            continue
        possible_values = list(FEATURES_TO_TEST[feature].keys())
        alternate_values = [v for v in possible_values if v != current_value]
        if not alternate_values:
            continue
        alternate_value = random.choice(alternate_values)
        test_assignments.append({
            'post': post,
            'feature': feature,
            'current_value': current_value,
            'alternate_value': alternate_value,
        })

    # Run
    results = []
    for assignment in tqdm(test_assignments, desc="Running scenario tests"):
        result = run_scenario_test(
            llm,
            embedding_model,
            assignment['post'],
            assignment['feature'],
            assignment['current_value'],
            assignment['alternate_value'],
        )
        if result['success']:
            results.append(result)

    if not results:
        logger.error("No successful tests completed!")
        return

    # Save visuals and summaries into reports_dir
    cwd = os.getcwd()
    os.makedirs(reports_dir, exist_ok=True)
    os.chdir(reports_dir)
    try:
        results_df = pd.DataFrame([
            {k: v for k, v in r.items() if k not in ['original_text', 'modified_text', 'original_prompt', 'modified_prompt']}
            for r in results
        ])
        feature_results = {}
        for feat in features_to_test:
            fd = results_df[results_df['feature'] == feat]
            if len(fd) > 0:
                feature_results[feat] = fd
        create_feature_visualizations(feature_results)
        create_impact_summary(results_df)
        save_detailed_examples(results)
    finally:
        os.chdir(cwd)

    logger.info("Scenario tests completed successfully!")

    # Update manifest
    if args.run_id:
        manifest = read_manifest(args.run_id, args.base_dir)
        update_stage(args.run_id, args.base_dir, manifest, "19-scenario-tests", in_path, [
            os.path.join(reports_dir, "feature_impact_comparison.png"),
            os.path.join(reports_dir, "length_differences_by_feature.png"),
            os.path.join(reports_dir, "feature_impact_summary.csv"),
            os.path.join(reports_dir, "scenario_test_summary.txt"),
        ], sig, extra={"sampled": len(sample_posts), "eligible": eligible, "assignments": len(test_assignments)})


if __name__ == "__main__":
    main()

        # Group results by feature
        feature_stats = {}
        for result in results:
            feature = result['feature']
            if feature not in feature_stats:
                feature_stats[feature] = {
                    'count': 0,
                    'similarities': [],
                    'length_diffs': []
                }

            feature_stats[feature]['count'] += 1
            feature_stats[feature]['similarities'].append(result['embedding_similarity'])
            feature_stats[feature]['length_diffs'].append(result['length_diff_pct'])

        # Calculate stats for each feature
        for feature, stats in feature_stats.items():
            avg_similarity = np.mean(stats['similarities'])
            avg_length_diff = np.mean(stats['length_diffs'])

            f.write(f"{feature.capitalize()}:\n")
            f.write(f"  Tests conducted: {stats['count']}\n")
            f.write(f"  Average impact: {1-avg_similarity:.4f} (0=none, 1=complete change)\n")
            f.write(f"  Average length difference: {avg_length_diff:.2f}%\n")

            # Get most impactful transitions
            transitions = []
            for result in results:
                if result['feature'] == feature:
                    transitions.append((
                        f"{result['original_value']} → {result['new_value']}",
                        result['embedding_similarity']
                    ))

            # Sort and show top transitions
            transitions.sort(key=lambda x: x[1])
            f.write("  Most impactful transitions:\n")
            for transition, similarity in transitions[:3]:
                f.write(f"    - {transition}: {1-similarity:.4f}\n")

            f.write("\n")

        # Overall summary
        all_similarities = [r['embedding_similarity'] for r in results]
        all_length_diffs = [r['length_diff_pct'] for r in results]

        f.write("Overall Results:\n")
        f.write(f"  Total tests conducted: {len(results)}\n")
        f.write(f"  Average impact across all features: {1-np.mean(all_similarities):.4f}\n")
        f.write(f"  Average length difference: {np.mean(all_length_diffs):.2f}%\n")

        # Feature ranking
        feature_ranking = []
        for feature, stats in feature_stats.items():
            avg_similarity = np.mean(stats['similarities'])
            feature_ranking.append((feature, 1-avg_similarity))

        feature_ranking.sort(key=lambda x: x[1], reverse=True)
        f.write("\nFeatures Ranked by Impact:\n")
        for i, (feature, impact) in enumerate(feature_ranking):
            f.write(f"  {i+1}. {feature}: {impact:.4f}\n")

    logger.info("Saved detailed examples and summary")

if __name__ == "__main__":
    try:
        run_tests()
    except Exception as e:
        logger.error(f"Error in scenario-based tests: {e}", exc_info=True)
        print(f"Tests failed with error: {e}")