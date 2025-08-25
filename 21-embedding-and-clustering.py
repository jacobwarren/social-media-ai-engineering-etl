import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import umap
from tqdm import tqdm
from collections import Counter, defaultdict
import matplotlib.patches as mpatches
import logging
import os
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, Normalize
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple

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
logger = init_pipeline_logging("phase2.embedding_clustering", None, "21-embedding-clustering")

# Defaults
RUN_ID = None
BASE_DIR = "data/processed"
INPUT_FILE = None  # Resolved through manifest
FEATURE_TO_ANALYZE = 'structure'  # Default feature to analyze
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DEFAULT_DIM_REDUCTION = 'umap'
MAX_CLUSTERS = 12

# Feature groups for combined embedding creation
FEATURE_GROUPS = {
    'basic': ['structure', 'tone', 'emoji_usage'],
    'narrative': ['flow', 'pacing', 'sentiment_arc'],
    'formatting': ['line_breaks', 'avg_line_breaks', 'bullet_styles', 'punctuation_usage'],
    'content': ['vocabulary_usage', 'topic_shifts', 'profanity']
}

def _resolve_input(run_id: str | None, base_dir: str, explicit_input: str | None) -> Tuple[str, str]:
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
        reports_dir = os.path.join("reports", run_id or "adhoc", "embedding-clustering")
        os.makedirs(reports_dir, exist_ok=True)
        return in_path, reports_dir
    else:
        if not explicit_input:
            raise ValueError("Provide --input when --run-id is not used")
        reports_dir = os.path.join("reports", "adhoc", "embedding-clustering")
        os.makedirs(reports_dir, exist_ok=True)
        return explicit_input, reports_dir


def load_data(input_path: str):
    """Load data from JSONL file"""
    logger.info(f"Loading data from {input_path}")
    posts = []
    try:
        with open(input_path, 'r') as f:
            for line in f:
                try:
                    posts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        logger.error(f"Input file {input_path} not found!")
        return None

    df = pd.DataFrame(posts)
    logger.info(f"Loaded {len(df)} posts")

    df = df[df['post_text'].notna() & (df['post_text'] != '')]
    logger.info(f"After filtering empty posts: {len(df)} posts remaining")

    for feature in ['structure', 'tone', 'emoji_usage']:
        if feature in df.columns:
            df[feature] = df[feature].astype(str)

    return df

def extract_feature_vectors(df, feature_groups=None):
    """
    Extract numerical feature vectors from the posts
    Returns standardized numerical features and categorical mappings
    """
    if feature_groups is None:
        feature_groups = FEATURE_GROUPS
    
    logger.info("Extracting feature vectors")
    
    # 1. Prepare numerical features
    numerical_features = {}
    
    # Line breaks
    if 'line_breaks' in df.columns:
        numerical_features['line_breaks'] = df['line_breaks'].fillna(0)
    
    if 'avg_line_breaks' in df.columns:
        numerical_features['avg_line_breaks'] = df['avg_line_breaks'].fillna(0)
    
    # Vocabulary
    if 'vocabulary_usage' in df.columns:
        numerical_features['vocabulary_usage'] = df['vocabulary_usage'].fillna(0)
    
    # Text length
    numerical_features['text_length'] = df['post_text'].apply(len)
    
    # Word count
    numerical_features['word_count'] = df['post_text'].apply(lambda x: len(str(x).split()))
    
    # Topic shifts
    if 'topic_shifts' in df.columns:
        numerical_features['topic_shifts_count'] = df['topic_shifts'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        
        # Average shift score
        numerical_features['avg_shift_score'] = df['topic_shifts'].apply(
            lambda x: np.mean([shift.get('shift_score', 0) for shift in x]) 
            if isinstance(x, list) and len(x) > 0 else 0
        )
    
    # Extract punctuation counts if available
    if 'punctuation_usage' in df.columns:
        for punct in ['.', ',', '!', '?', ';', ':']:
            numerical_features[f'punct_{punct}'] = df['punctuation_usage'].apply(
                lambda x: x.get(punct, 0) if isinstance(x, dict) else 0
            )
    
    # Convert to DataFrame
    numerical_df = pd.DataFrame(numerical_features, index=df.index)
    
    # 2. Prepare categorical mappings
    categorical_mappings = {}
    
    # Structure
    if 'structure' in df.columns:
        structures = sorted(df['structure'].unique())
        categorical_mappings['structure'] = {s: i for i, s in enumerate(structures)}
    
    # Tone
    if 'tone' in df.columns:
        # Extract primary tone (first word)
        df['primary_tone'] = df['tone'].apply(
            lambda x: str(x).split(',')[0].strip() if isinstance(x, str) else 'unknown'
        )
        tones = sorted(df['primary_tone'].unique())
        categorical_mappings['tone'] = {t: i for i, t in enumerate(tones)}
    
    # Emoji usage
    if 'emoji_usage' in df.columns:
        emoji_levels = sorted(df['emoji_usage'].unique())
        categorical_mappings['emoji_usage'] = {e: i for i, e in enumerate(emoji_levels)}
    
    # Pacing
    if 'pacing' in df.columns:
        pacing_types = sorted(df['pacing'].unique())
        categorical_mappings['pacing'] = {p: i for i, p in enumerate(pacing_types)}
    
    # Sentiment arc
    if 'sentiment_arc' in df.columns:
        arc_types = sorted(df['sentiment_arc'].unique())
        categorical_mappings['sentiment_arc'] = {a: i for i, a in enumerate(arc_types)}
    
    # Flow/narrative (taking first element)
    if 'flow' in df.columns:
        # Extract first flow element
        df['first_flow'] = df['flow'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'unknown'
        )
        flow_types = sorted(df['first_flow'].unique())
        categorical_mappings['flow'] = {f: i for i, f in enumerate(flow_types)}
    
    # Bullet styles
    if 'bullet_styles' in df.columns:
        bullet_types = sorted(df['bullet_styles'].unique())
        categorical_mappings['bullet_styles'] = {b: i for i, b in enumerate(bullet_types)}
    
    # Profanity
    if 'profanity' in df.columns:
        profanity_levels = sorted(df['profanity'].unique())
        categorical_mappings['profanity'] = {p: i for i, p in enumerate(profanity_levels)}
    
    # 3. Convert categorical features to one-hot vectors
    categorical_vectors = {}
    
    for feature, mapping in categorical_mappings.items():
        # Get the actual data column name
        data_col = feature
        if feature == 'flow':
            data_col = 'first_flow'
        elif feature == 'tone':
            data_col = 'primary_tone'
        
        # Skip if column not in dataframe
        if data_col not in df.columns:
            continue
            
        # Create one-hot encoder
        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(df[[data_col]])
        
        # Create column names
        categories = list(mapping.keys())
        column_names = [f"{feature}_{cat}" for cat in categories]
        
        # Create DataFrame
        encoded_df = pd.DataFrame(encoded, index=df.index, columns=column_names)
        
        # Add to categorical vectors
        for col in encoded_df.columns:
            categorical_vectors[col] = encoded_df[col]
    
    # 4. Standardize numerical features
    scaler = StandardScaler()
    numerical_scaled = scaler.fit_transform(numerical_df)
    numerical_scaled_df = pd.DataFrame(
        numerical_scaled, index=df.index, columns=numerical_df.columns
    )
    
    # 5. Combine all features into a single DataFrame
    all_features_df = pd.concat([numerical_scaled_df, pd.DataFrame(categorical_vectors, index=df.index)], axis=1)
    
    logger.info(f"Extracted {all_features_df.shape[1]} feature dimensions")
    
    return all_features_df, categorical_mappings, numerical_df.columns.tolist()

def generate_embeddings(df):
    """Generate text embeddings and combined embeddings with feature vectors"""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    logger.info("Generating text embeddings...")
    text_embeddings = model.encode(df['post_text'].tolist(), show_progress_bar=True)
    logger.info(f"Generated text embeddings with shape: {text_embeddings.shape}")
    
    # Extract feature vectors
    feature_vectors, categorical_mappings, numerical_features = extract_feature_vectors(df)
    
    # Create combined embeddings
    logger.info("Creating combined embeddings (text + features)...")
    combined_embeddings = np.concatenate([
        text_embeddings, 
        feature_vectors.values
    ], axis=1)
    
    logger.info(f"Combined embeddings shape: {combined_embeddings.shape}")
    
    return text_embeddings, combined_embeddings, feature_vectors, categorical_mappings, numerical_features

def reduce_dimensions(embeddings, method=DEFAULT_DIM_REDUCTION, n_components=2):
    """Reduce dimensionality of embeddings for visualization"""
    logger.info(f"Reducing dimensions using {method}...")
    
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
        method_name = "Principal Component Analysis"
    elif method == 'tsne':
        perplexity = min(30, len(embeddings) - 1)
        reducer = TSNE(n_components=n_components, random_state=42, 
                      perplexity=perplexity, init='pca')
        method_name = f"t-SNE (perplexity={perplexity})"
    elif method == 'umap':
        # Use more robust settings for large datasets
        n_neighbors = min(15, len(embeddings) - 1)
        min_dist = 0.1
        reducer = umap.UMAP(n_components=n_components, random_state=42,
                          n_neighbors=n_neighbors,
                          min_dist=min_dist,
                          metric='cosine')  # Use cosine for text embeddings
        method_name = f"UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})"
    else:
        logger.error(f"Unknown dimensionality reduction method: {method}")
        return None, None
    
    try:
        reduced_embeddings = reducer.fit_transform(embeddings)
        logger.info(f"Reduced embeddings to {n_components} dimensions")
        return reduced_embeddings, method_name
    except Exception as e:
        logger.error(f"Error during dimensionality reduction: {e}")
        # Fall back to PCA if the requested method fails
        if method != 'pca':
            logger.info("Falling back to PCA...")
            reducer = PCA(n_components=n_components, random_state=42)
            reduced_embeddings = reducer.fit_transform(embeddings)
            return reduced_embeddings, "Principal Component Analysis (fallback)"
        else:
            raise

def optimize_clusters(embeddings, min_clusters=2, max_clusters=None):
    """Find optimal number of clusters using silhouette score"""
    logger.info("Finding optimal number of clusters...")
    
    if max_clusters is None:
        max_clusters = min(MAX_CLUSTERS, len(embeddings) - 1)
    
    silhouette_scores = []
    inertia_values = []
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        # Calculate inertia (within-cluster sum of squares)
        inertia_values.append(kmeans.inertia_)
        
        logger.info(f"  Clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.4f}, Inertia: {kmeans.inertia_:.2f}")
    
    # Find optimal number of clusters based on silhouette score
    optimal_clusters = np.argmax(silhouette_scores) + min_clusters
    logger.info(f"Optimal number of clusters (by silhouette score): {optimal_clusters}")
    
    # Create plot to show cluster optimization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot silhouette scores
    cluster_range = list(range(min_clusters, max_clusters + 1))
    ax1.plot(cluster_range, silhouette_scores, 'bo-')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Silhouette Score vs. Number of Clusters')
    ax1.axvline(x=optimal_clusters, color='r', linestyle='--')
    ax1.grid(True)
    
    # Plot inertia (elbow method)
    ax2.plot(cluster_range, inertia_values, 'go-')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Inertia (Within-Cluster Sum of Squares)')
    ax2.set_title('Elbow Method for Optimal k')
    ax2.grid(True)
    
    # Find elbow point using rate of change
    inertia_changes = np.diff(inertia_values)
    rate_of_change = np.diff(inertia_changes)
    elbow_cluster_idx = np.argmin(rate_of_change) if len(rate_of_change) > 0 else 0
    elbow_cluster = elbow_cluster_idx + min_clusters + 2  # +2 because we did two diffs
    
    if min_clusters <= elbow_cluster <= max_clusters:
        ax2.axvline(x=elbow_cluster, color='r', linestyle='--')
        logger.info(f"Optimal number of clusters (by elbow method): {elbow_cluster}")
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('embedding_results', exist_ok=True)
    
    plt.savefig('embedding_results/cluster_optimization.png', dpi=300)
    logger.info("Saved cluster optimization plot to embedding_results/cluster_optimization.png")
    
    # Return silhouette-based optimal clusters
    return optimal_clusters

def cluster_embeddings(embeddings, n_clusters):
    """Cluster embeddings using KMeans"""
    logger.info(f"Clustering embeddings into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    logger.info(f"Clustered {len(embeddings)} embeddings into {n_clusters} clusters")
    
    # Get cluster centers for later analysis
    cluster_centers = kmeans.cluster_centers_
    
    return cluster_labels, cluster_centers

def analyze_clusters(df, cluster_labels, feature_to_analyze=None):
    """Analyze the distribution of features across clusters"""
    if feature_to_analyze is None:
        feature_to_analyze = FEATURE_TO_ANALYZE
    
    logger.info(f"Analyzing clusters for feature: {feature_to_analyze}")
    
    # Add cluster labels to DataFrame
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels
    
    # Prepare storage for cluster analysis
    cluster_analysis = {}
    
    # For each cluster, analyze feature distribution
    for cluster_id in range(max(cluster_labels) + 1):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        
        # Store cluster size
        cluster_size = len(cluster_data)
        
        # Skip empty clusters
        if cluster_size == 0:
            continue
        
        # Analysis will store feature distributions and examples
        analysis = {'size': cluster_size}
        
        # Analyze main feature distribution
        if feature_to_analyze in cluster_data.columns:
            # Count occurrences of each feature value
            value_counts = cluster_data[feature_to_analyze].value_counts(normalize=True)
            analysis['feature_distribution'] = value_counts.to_dict()
            
            # Find dominant feature (most common value)
            if not value_counts.empty:
                dominant_feature = value_counts.index[0]
                dominant_proportion = value_counts.iloc[0]
                analysis['dominant_feature'] = dominant_feature
                analysis['dominant_proportion'] = dominant_proportion
        
        # Analyze other key categorical features
        for feature in ['tone', 'emoji_usage', 'pacing', 'sentiment_arc', 'profanity']:
            if feature in cluster_data.columns:
                value_counts = cluster_data[feature].value_counts(normalize=True)
                if not value_counts.empty:
                    analysis[f'{feature}_distribution'] = value_counts.to_dict()
        
        # Analyze narrative flow (if available)
        if 'flow' in cluster_data.columns:
            # Count first flow element
            flow_values = []
            for flow_list in cluster_data['flow']:
                if isinstance(flow_list, list) and len(flow_list) > 0:
                    flow_values.append(flow_list[0])
            
            if flow_values:
                flow_counts = pd.Series(flow_values).value_counts(normalize=True)
                analysis['flow_distribution'] = flow_counts.to_dict()
        
        # Extract sample posts from this cluster
        sample_posts = cluster_data.head(3)['post_text'].tolist()
        analysis['examples'] = sample_posts
        
        # Store for this cluster
        cluster_analysis[cluster_id] = analysis
    
    return cluster_analysis

def analyze_feature_distributions_across_clusters(df, cluster_labels):
    """
    Analyze how different features are distributed across clusters
    Returns dict mapping features to their purity metrics
    """
    logger.info("Analyzing feature distributions across clusters")
    
    feature_analysis = {}
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels
    
    # Features to analyze
    features_to_analyze = [
        'structure', 'tone', 'emoji_usage', 'pacing', 'sentiment_arc', 'profanity'
    ]
    
    # For each feature, analyze distribution across clusters
    for feature in features_to_analyze:
        if feature not in df.columns:
            continue
            
        feature_analysis[feature] = {}
        
        # For each unique value of this feature
        for feature_value in df[feature].unique():
            if pd.isna(feature_value):
                continue
                
            # Find posts with this feature value
            feature_posts = df_with_clusters[df_with_clusters[feature] == feature_value]
            
            # Skip if no posts with this feature value
            if len(feature_posts) == 0:
                continue
                
            # Count occurrences in each cluster
            cluster_counts = feature_posts['cluster'].value_counts()
            
            # Calculate purity (% of posts with this feature in the dominant cluster)
            dominant_cluster = cluster_counts.index[0] if not cluster_counts.empty else None
            total_posts = len(feature_posts)
            
            if dominant_cluster is not None and total_posts > 0:
                dominant_count = cluster_counts.iloc[0]
                purity = dominant_count / total_posts
                
                # Store results
                feature_analysis[feature][feature_value] = {
                    'total_posts': total_posts,
                    'dominant_cluster': dominant_cluster,
                    'dominant_count': dominant_count,
                    'purity': purity,
                    'cluster_distribution': cluster_counts.to_dict()
                }
    
    return feature_analysis

def visualize_feature_distributions(cluster_analysis, feature_to_analyze=None):
    """Create visualizations of feature distributions within clusters"""
    if feature_to_analyze is None:
        feature_to_analyze = FEATURE_TO_ANALYZE
    
    logger.info(f"Creating feature distribution visualizations for {feature_to_analyze}")
    
    # Ensure output directory exists
    os.makedirs('embedding_results', exist_ok=True)
    
    # Extract feature distributions from cluster analysis
    feature_values = set()
    clusters = sorted(cluster_analysis.keys())
    
    # Find all unique feature values
    for cluster_id, analysis in cluster_analysis.items():
        if 'feature_distribution' in analysis:
            feature_values.update(analysis['feature_distribution'].keys())
    
    feature_values = sorted(feature_values)
    
    # Create matrix of proportions
    data = np.zeros((len(feature_values), len(clusters)))
    
    for i, feature_val in enumerate(feature_values):
        for j, cluster_id in enumerate(clusters):
            # Get proportion of this feature value in this cluster
            analysis = cluster_analysis[cluster_id]
            if 'feature_distribution' in analysis:
                data[i, j] = analysis['feature_distribution'].get(feature_val, 0)
    
    # Create colormap with distinct colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(feature_values)))
    
    # Create stacked bar chart
    plt.figure(figsize=(15, 10))
    
    bottoms = np.zeros(len(clusters))
    for i, feature_val in enumerate(feature_values):
        plt.bar(
            clusters, data[i], bottom=bottoms, 
            label=feature_val, color=colors[i], 
            edgecolor='white', width=0.8
        )
        bottoms += data[i]
    
    plt.title(f'Distribution of {feature_to_analyze} Values by Cluster', fontsize=16)
    plt.xlabel('Cluster ID', fontsize=14)
    plt.ylabel('Proportion', fontsize=14)
    plt.xticks(clusters)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend(title=feature_to_analyze, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'embedding_results/feature_distribution_by_cluster.png', dpi=300)
    plt.close()
    
    # Create dominant feature bar chart
    plt.figure(figsize=(15, 10))
    
    # Prepare data for dominant feature chart
    dominant_features = {}
    for cluster_id, analysis in cluster_analysis.items():
        if 'dominant_feature' in analysis and 'dominant_proportion' in analysis:
            dominant_features[cluster_id] = (
                analysis['dominant_feature'],
                analysis['dominant_proportion']
            )
    
    # Create DataFrame
    dominant_df = pd.DataFrame({
        'Cluster': list(dominant_features.keys()),
        'Dominant Feature': [x[0] for x in dominant_features.values()],
        'Proportion': [x[1] for x in dominant_features.values()]
    })
    
    # Sort by cluster
    dominant_df = dominant_df.sort_values('Cluster')
    
    # Create color map for dominant features
    unique_dominant = dominant_df['Dominant Feature'].unique()
    feature_to_idx = {f: i for i, f in enumerate(feature_values)}
    color_map = {val: colors[feature_to_idx.get(val, 0)] for val in unique_dominant}
    
    # Create bars
    bars = plt.bar(
        dominant_df['Cluster'], 
        dominant_df['Proportion'],
        color=[color_map.get(feature, 'gray') for feature in dominant_df['Dominant Feature']]
    )
    
    # Add cluster size annotations
    for cluster_id in dominant_df['Cluster']:
        cluster_size = cluster_analysis[cluster_id]['size']
        plt.annotate(
            f"n={cluster_size}", 
            xy=(cluster_id, 0.05),
            ha='center',
            va='bottom',
            fontsize=10,
            color='black'
        )
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=12
        )
    
    # Create legend
    patches = [mpatches.Patch(color=color_map.get(feature, 'gray'), label=feature) 
              for feature in unique_dominant]
    plt.legend(handles=patches, title=feature_to_analyze, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.title(f'Dominant {feature_to_analyze} in Each Cluster', fontsize=16)
    plt.xlabel('Cluster ID', fontsize=14)
    plt.ylabel('Proportion', fontsize=14)
    plt.xticks(dominant_df['Cluster'])
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'embedding_results/dominant_features_by_cluster.png', dpi=300)
    plt.close()
    
    logger.info("Saved feature distribution visualizations")

def visualize_embeddings_by_feature(reduced_embeddings, df, feature, cluster_labels=None, method_name=None):
    """
    Create visualization of embeddings colored by feature and cluster
    """
    logger.info(f"Creating embedding visualization for feature: {feature}")
    
    # Ensure output directory exists
    os.makedirs('embedding_results', exist_ok=True)
    
    # Skip if feature not in DataFrame
    if feature not in df.columns:
        logger.warning(f"Feature '{feature}' not found in DataFrame")
        return
    
    # Clean feature values (ensure strings and handle NaN)
    feature_values = df[feature].fillna('unknown').astype(str).tolist()
    
    # Count occurrences of each value
    value_counts = Counter(feature_values)
    
    # Get unique values
    unique_values = sorted(value_counts.keys())
    
    # Create colormap for feature values
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_values)))
    color_dict = {val: colors[i] for i, val in enumerate(unique_values)}
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot 1: Colored by feature
    for i, val in enumerate(unique_values):
        mask = np.array([f == val for f in feature_values])
        if np.any(mask):  # Only plot if we have examples
            ax1.scatter(
                reduced_embeddings[mask, 0], 
                reduced_embeddings[mask, 1],
                c=np.array([color_dict[val]]),
                label=f"{val} ({value_counts[val]})",
                alpha=0.7,
                edgecolors='none',
                s=50
            )
    
    # Add labels and title
    dim_reduction_name = method_name if method_name else DEFAULT_DIM_REDUCTION.upper()
    ax1.set_title(f'Post Embeddings by {feature.capitalize()}', fontsize=14)
    ax1.legend(title=feature.capitalize(), loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_xlabel(f'{dim_reduction_name} Dimension 1', fontsize=12)
    ax1.set_ylabel(f'{dim_reduction_name} Dimension 2', fontsize=12)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Colored by cluster with feature labels
    if cluster_labels is not None:
        # Create a colormap for clusters
        n_clusters = len(set(cluster_labels))
        cluster_colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
        
        # Plot each point
        scatter = ax2.scatter(
            reduced_embeddings[:, 0], 
            reduced_embeddings[:, 1],
            c=cluster_labels,
            cmap=ListedColormap(cluster_colors),
            alpha=0.7,
            edgecolors='none',
            s=50
        )
        
        # Add cluster annotation
        for cluster_id in range(n_clusters):
            # Get center of cluster points
            cluster_points = reduced_embeddings[cluster_labels == cluster_id]
            if len(cluster_points) == 0:
                continue
                
            center_x = np.mean(cluster_points[:, 0])
            center_y = np.mean(cluster_points[:, 1])
            
            # Get dominant feature for this cluster
            cluster_features = [feature_values[i] for i, c in enumerate(cluster_labels) if c == cluster_id]
            counter = Counter(cluster_features)
            dominant_feature, count = counter.most_common(1)[0]
            proportion = count / len(cluster_features)
            
            # Add annotation with dominant feature and percentage
            ax2.annotate(
                f"{cluster_id}: {dominant_feature}\n({proportion:.0%})",
                (center_x, center_y),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
                ha='center', va='center'
            )
        
        # Add labels and title
        ax2.set_title(f'Post Embeddings by Cluster (n_clusters={n_clusters})', fontsize=14)
        ax2.set_xlabel(f'{dim_reduction_name} Dimension 1', fontsize=12)
        ax2.set_ylabel(f'{dim_reduction_name} Dimension 2', fontsize=12)
        ax2.grid(alpha=0.3)
        
        # Add colorbar for clusters
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Cluster ID', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'embedding_results/embeddings_by_{feature}.png', dpi=300)
    plt.close()
    
    logger.info(f"Saved embedding visualization for {feature}")

def create_feature_correlation_heatmap(df, cluster_labels):
    """
    Create a correlation heatmap between features and clusters
    """
    logger.info("Creating feature-cluster correlation matrix")
    
    # Ensure output directory exists
    os.makedirs('embedding_results', exist_ok=True)
    
    # Features to analyze
    categorical_features = ['structure', 'tone', 'emoji_usage', 'pacing', 'sentiment_arc', 'profanity']
    
    # Filter to available features
    available_features = [f for f in categorical_features if f in df.columns]
    
    if not available_features:
        logger.warning("No categorical features found for correlation analysis")
        return
    
    # Create correlation DataFrame
    correlation_df = pd.DataFrame()
    
    # Add cluster dummy variables
    for cluster_id in range(max(cluster_labels) + 1):
        correlation_df[f'cluster_{cluster_id}'] = (cluster_labels == cluster_id).astype(int)
    
    # Add feature dummy variables
    for feature in available_features:
        # Get dummy variables for this feature
        dummies = pd.get_dummies(df[feature], prefix=feature)
        # Add to correlation DataFrame
        correlation_df = pd.concat([correlation_df, dummies], axis=1)
    
    # Calculate correlation matrix
    corr_matrix = correlation_df.corr()
    
    # Extract only correlations between features and clusters
    feature_cols = [col for col in corr_matrix.columns if not col.startswith('cluster_')]
    cluster_cols = [col for col in corr_matrix.columns if col.startswith('cluster_')]
    
    # Get correlation submatrix
    submatrix = corr_matrix.loc[feature_cols, cluster_cols]
    
    # Create heatmap
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        submatrix, 
        annot=True, 
        cmap='RdBu_r',
        center=0,
        fmt='.2f',
        linewidths=.5
    )
    
    plt.title('Correlation between Features and Clusters', fontsize=16)
    plt.tight_layout()
    plt.savefig('embedding_results/feature_cluster_correlation.png', dpi=300)
    plt.close()
    
    logger.info("Saved feature-cluster correlation matrix")

def calculate_feature_separability(df, reduced_embeddings, feature):
    """
    Calculate how well a feature separates in the embedding space
    Returns a separability score based on within-class vs between-class distances
    """
    logger.info(f"Calculating separability for feature: {feature}")
    
    # Skip if feature not in DataFrame
    if feature not in df.columns:
        logger.warning(f"Feature '{feature}' not found in DataFrame")
        return None
    
    # Get feature values
    feature_values = df[feature].fillna('unknown').astype(str).tolist()
    
    # Get unique values
    unique_values = sorted(set(feature_values))
    
    # Skip if only one unique value
    if len(unique_values) <= 1:
        logger.warning(f"Feature '{feature}' has only one unique value")
        return None
    
    # Calculate centroid for each feature value
    centroids = {}
    for val in unique_values:
        points = reduced_embeddings[[f == val for f in feature_values]]
        if len(points) > 0:
            centroids[val] = np.mean(points, axis=0)
    
    # Calculate average within-class distance
    within_distances = []
    for val in unique_values:
        points = reduced_embeddings[[f == val for f in feature_values]]
        if len(points) > 0 and val in centroids:
            distances = np.sqrt(np.sum((points - centroids[val])**2, axis=1))
            within_distances.extend(distances)
    
    avg_within = np.mean(within_distances) if within_distances else 0
    
    # Calculate average between-class distance
    between_distances = []
    for i, val1 in enumerate(unique_values):
        if val1 not in centroids:
            continue
        for j, val2 in enumerate(unique_values[i+1:], i+1):
            if val2 not in centroids:
                continue
            distance = np.sqrt(np.sum((centroids[val1] - centroids[val2])**2))
            between_distances.append(distance)
    
    avg_between = np.mean(between_distances) if between_distances else 0
    
    # Calculate separability (ratio of between to within)
    if avg_within > 0:
        separability = avg_between / avg_within
    else:
        separability = 0
    
    return {
        'feature': feature,
        'unique_values': len(unique_values),
        'avg_within_distance': avg_within,
        'avg_between_distance': avg_between,
        'separability': separability
    }

def run_analysis(run_id: str | None,
                base_dir: str,
                input_path: str | None,
                feature_to_analyze: str,
                seed: int | None):
    """Main function to run embedding analysis"""
    set_global_seed(seed)
    logger.info("Starting embedding analysis")

    # Resolve input and reports dir
    in_path, reports_dir = _resolve_input(run_id, base_dir, input_path)

    # Idempotent skip via signature
    sig = compute_hash([in_path], {"stage": 21, "feature": feature_to_analyze, "stage_version": STAGE_VERSION})
    if run_id:
        manifest = read_manifest(run_id, base_dir)
        out_dir = os.path.join(reports_dir, 'embedding_results')
        out_paths = [
            os.path.join(out_dir, 'cluster_optimization.png'),
            os.path.join(out_dir, 'feature_distribution_by_cluster.png'),
            os.path.join(out_dir, f'embeddings_by_{feature_to_analyze}.png'),
            os.path.join(out_dir, 'dominant_features_by_cluster.png'),
            os.path.join(out_dir, 'feature_cluster_correlation.png'),
            os.path.join(out_dir, 'feature_separability.png'),
            os.path.join(out_dir, 'embedding_analysis_results.txt'),
        ]
        if should_skip(manifest, "21-embedding-clustering", sig, out_paths):
            logger.info(f"Skipping 21-embedding-clustering; up-to-date in {reports_dir}")
            return

    # Ensure outputs in reports dir
    out_dir = os.path.join(reports_dir, 'embedding_results')
    os.makedirs(out_dir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(reports_dir)
    try:
        # Load data
        df = load_data(in_path)
        if df is None:
            return

        # Generate text and combined embeddings
        text_embeddings, combined_embeddings, feature_vectors, category_mappings, numerical_features = generate_embeddings(df)

        # Find optimal number of clusters
        n_clusters = optimize_clusters(combined_embeddings)

        # Cluster embeddings
        cluster_labels, cluster_centers = cluster_embeddings(combined_embeddings, n_clusters)

        # Analyze clusters for main feature
        global FEATURE_TO_ANALYZE
        FEATURE_TO_ANALYZE = feature_to_analyze
        cluster_analysis = analyze_clusters(df, cluster_labels, FEATURE_TO_ANALYZE)

        # Analyze how features distribute across clusters
        feature_distributions = analyze_feature_distributions_across_clusters(df, cluster_labels)

        # Reduce dimensions and render
        for method in ['umap', 'tsne', 'pca']:
            try:
                reduced_embeddings, method_name = reduce_dimensions(combined_embeddings, method=method)
                if reduced_embeddings is not None:
                    visualize_embeddings_by_feature(reduced_embeddings, df, FEATURE_TO_ANALYZE, cluster_labels, method_name)
                    for feature in ['tone', 'emoji_usage', 'pacing', 'sentiment_arc']:
                        if feature in df.columns:
                            visualize_embeddings_by_feature(reduced_embeddings, df, feature, cluster_labels, method_name)
                    if method == DEFAULT_DIM_REDUCTION:
                        visualize_feature_distributions(cluster_analysis, FEATURE_TO_ANALYZE)
                        create_feature_correlation_heatmap(df, cluster_labels)
                        separability_results = []
                        for feature in ['structure', 'tone', 'emoji_usage', 'pacing', 'sentiment_arc', 'profanity']:
                            if feature in df.columns:
                                result = calculate_feature_separability(df, reduced_embeddings, feature)
                                if result:
                                    separability_results.append(result)
                        if separability_results:
                            separability_df = pd.DataFrame(separability_results).sort_values('separability', ascending=False)
                            plt.figure(figsize=(12, 8))
                            norm = Normalize(separability_df['separability'].min(), separability_df['separability'].max())
                            colors = plt.cm.viridis(norm(separability_df['separability']))
                            bars = plt.bar(separability_df['feature'], separability_df['separability'], color=colors)
                            for bar in bars:
                                height = bar.get_height()
                                plt.text(bar.get_x() + bar.get_width()/2., height + 0.05, f'{height:.2f}', ha='center', va='bottom', fontsize=12)
                            plt.title('Feature Separability in Embedding Space', fontsize=16)
                            plt.ylabel('Separability Score (higher = better)', fontsize=14)
                            plt.xlabel('Feature', fontsize=14)
                            plt.ylim(0, max(separability_df['separability']) * 1.2)
                            plt.grid(axis='y', linestyle='--', alpha=0.7)
                            plt.tight_layout()
                            plt.savefig('embedding_results/feature_separability.png', dpi=300)
                            plt.close()
                    break
            except Exception as e:
                logger.error(f"Error with {method} dimensionality reduction: {e}")
                continue

        with open('embedding_results/embedding_analysis_results.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"EMBEDDING ANALYSIS RESULTS FOR {FEATURE_TO_ANALYZE.upper()}\n")
            f.write("=" * 80 + "\n\n")
            f.write("DATASET SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"\nDistribution of {FEATURE_TO_ANALYZE} values:\n")
            if FEATURE_TO_ANALYZE in df.columns:
                value_counts = df[FEATURE_TO_ANALYZE].value_counts()
                for value, count in value_counts.items():
                    f.write(f"  - {value}: {count} ({count/len(df):.1%})\n")
            else:
                f.write(f"  Feature {FEATURE_TO_ANALYZE} not found in dataset.\n")
            f.write("\n\nEMBEDDING AND CLUSTERING DETAILS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Embedding model: {EMBEDDING_MODEL}\n")
            f.write(f"Text embedding dimensions: {text_embeddings.shape[1]}\n")
            f.write(f"Combined embedding dimensions: {combined_embeddings.shape[1]}\n")
            f.write(f"Dimensionality reduction: {DEFAULT_DIM_REDUCTION}\n")
            f.write(f"Number of clusters: {n_clusters}\n")
            f.write("\n\nCLUSTER ANALYSIS\n")
            f.write("-" * 80 + "\n")
            cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
            for cluster_id in sorted(cluster_analysis.keys()):
                analysis = cluster_analysis[cluster_id]
                f.write(f"\nCluster {cluster_id} (size: {analysis['size']}, {analysis['size']/len(df):.1%} of data):\n")
                if 'feature_distribution' in analysis:
                    f.write(f"  {FEATURE_TO_ANALYZE} distribution:\n")
                    for feature_val, proportion in sorted(analysis['feature_distribution'].items(), key=lambda x: x[1], reverse=True):
                        feature_count = int(proportion * analysis['size'])
                        f.write(f"    - {feature_val}: {proportion:.2f} ({feature_count} posts)\n")
                for feature in ['tone', 'emoji_usage', 'pacing', 'sentiment_arc', 'flow']:
                    dist_key = f'{feature}_distribution'
                    if dist_key in analysis:
                        f.write(f"\n  {feature.capitalize()} distribution:\n")
                        for val, prop in sorted(analysis[dist_key].items(), key=lambda x: x[1], reverse=True)[:3]:
                            f.write(f"    - {val}: {prop:.2f}\n")
                f.write("\n  Example posts:\n")
                for i, example in enumerate(analysis['examples'][:2]):
                    truncated = example[:200] + "..." if len(example) > 200 else example
                    truncated = truncated.replace('\n', ' ')
                    f.write(f"    Example {i+1}: {truncated}\n")
            f.write("\n\nFEATURE PURITY ANALYSIS\n")
            f.write("-" * 80 + "\n")
            if FEATURE_TO_ANALYZE in feature_distributions:
                feature_data = feature_distributions[FEATURE_TO_ANALYZE]
                sorted_features = sorted(feature_data.items(), key=lambda x: x[1]['purity'] if 'purity' in x[1] else 0, reverse=True)
                for feature_val, analysis in sorted_features:
                    f.write(f"\n{feature_val}:\n")
                    f.write(f"  Total posts: {analysis['total_posts']}\n")
                    f.write(f"  Dominant cluster: {analysis['dominant_cluster']} ")
                    f.write(f"({analysis['dominant_count']} posts, {analysis['purity']:.1%})\n")
                    f.write("  Cluster distribution:\n")
                    for cluster, count in sorted(analysis['cluster_distribution'].items(), key=lambda x: x[1], reverse=True)[:3]:
                        pct = count / analysis['total_posts']
                        f.write(f"    - Cluster {cluster}: {count} posts ({pct:.1%})\n")
            if 'separability_results' in locals() and separability_results:
                f.write("\n\nFEATURE SEPARABILITY IN EMBEDDING SPACE\n")
                f.write("-" * 80 + "\n")
                sorted_results = sorted(separability_results, key=lambda x: x['separability'], reverse=True)
                for result in sorted_results:
                    f.write(f"{result['feature']}:\n")
                    f.write(f"  Unique values: {result['unique_values']}\n")
                    f.write(f"  Avg within-class distance: {result['avg_within_distance']:.4f}\n")
                    f.write(f"  Avg between-class distance: {result['avg_between_distance']:.4f}\n")
                    f.write(f"  Separability score: {result['separability']:.4f}\n\n")
    finally:
        os.chdir(cwd)

    if run_id:
        manifest = read_manifest(run_id, base_dir)
        out_dir = os.path.join(reports_dir, 'embedding_results')
        update_stage(run_id, base_dir, manifest, "21-embedding-clustering", in_path, [
            os.path.join(out_dir, 'cluster_optimization.png'),
            os.path.join(out_dir, 'feature_distribution_by_cluster.png'),
            os.path.join(out_dir, f'embeddings_by_{feature_to_analyze}.png'),
            os.path.join(out_dir, 'dominant_features_by_cluster.png'),
            os.path.join(out_dir, 'feature_cluster_correlation.png'),
            os.path.join(out_dir, 'feature_separability.png'),
            os.path.join(out_dir, 'embedding_analysis_results.txt'),
        ], sig, extra={"rows": len(df), "n_clusters": int(n_clusters)})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Embedding and clustering analysis")
    parser.add_argument("--run-id", dest="run_id", default=None, help="Use 'latest' to use most recent run")
    parser.add_argument("--base-dir", dest="base_dir", default="data/processed")
    parser.add_argument("--input", dest="input_path", default=None, help="Only used when --run-id is not provided")
    parser.add_argument("--feature", dest="feature", default=FEATURE_TO_ANALYZE)
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    args = parser.parse_args()

    try:
        run_analysis(
            run_id=args.run_id,
            base_dir=args.base_dir,
            input_path=args.input_path,
            feature_to_analyze=args.feature,
            seed=args.seed,
        )
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        print(f"Analysis failed with error: {e}")