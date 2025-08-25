import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import logging
import os
from collections import defaultdict
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
logger = init_pipeline_logging("phase2.permutation_correlation", None, "20-permutation-correlation")

# Defaults
RUN_ID = None
BASE_DIR = "data/processed"
INPUT_FILE = None  # Resolved via manifest when run-id provided


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
        reports_dir = os.path.join("reports", run_id or "adhoc", "feature-importance")
        os.makedirs(reports_dir, exist_ok=True)
        return in_path, reports_dir
    else:
        if not explicit_input:
            raise ValueError("Provide --input when --run-id is not used")
        reports_dir = os.path.join("reports", "adhoc", "feature-importance")
        os.makedirs(reports_dir, exist_ok=True)
        return explicit_input, reports_dir


def load_and_add_synthetic_metrics(file_path):
    """
    Load data and add synthetic engagement metrics for demonstration if needed
    """
    logger.info(f"Loading data from {file_path}")
    posts = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    posts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        logger.error(f"Input file {file_path} not found!")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(posts)
    logger.info(f"Loaded {len(df)} posts")

    # Check for engagement metrics
    if not any(col in df.columns for col in ['likes', 'comments', 'shares', 'views', 'tier', 'engagement_score']):
        logger.info("No engagement metrics found. Adding synthetic metrics for demonstration.")

        # Add structure-based scores (assuming some structures perform better)
        structure_scores = {
            'instructional': 0.8,
            'inspirational': 0.9,
            'analytical': 0.7,
            'controversial': 0.85,
            'insightful': 0.75,
            'comparative': 0.6,
            'reflective': 0.65,
            'announcement': 0.8
        }

        # Add tone-based scores
        tone_scores = {
            'professional': 0.75,
            'casual': 0.65,
            'inspiring': 0.85,
            'witty': 0.8,
            'thoughtful': 0.7,
            'authoritative': 0.75,
            'friendly': 0.7
        }

        # Add emoji usage scores
        emoji_scores = {
            'high': 0.8,
            'medium': 0.7,
            'low': 0.6,
            'none': 0.5
        }

        # Length score (longer isn't always better, sweet spot around 1000 chars)
        df['text_length'] = df['post_text'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        df['length_score'] = df['text_length'].apply(
            lambda x: 1 - min(abs(x - 1000) / 1000, 0.5)  # Penalize deviation from 1000, max 50% penalty
        )

        # Structure score
        df['structure_score'] = df['structure'].map(
            lambda x: structure_scores.get(x, 0.5)  # Default 0.5 for unknown structures
        )

        # Tone score
        df['tone_score'] = df['tone'].apply(
            lambda x: tone_scores.get(x.split(',')[0].strip().lower(), 0.6) if isinstance(x, str) else 0.6
        )

        # Emoji usage score
        df['emoji_score'] = df['emoji_usage'].map(
            lambda x: emoji_scores.get(x, 0.6) if isinstance(x, str) else 0.6
        )

        # Add some randomness
        np.random.seed(42)  # For reproducibility
        df['random_factor'] = np.random.normal(1, 0.2, size=len(df))  # Mean 1, std 0.2

        # Calculate synthetic engagement
        df['engagement_score'] = (
            df['structure_score'] *
            df['tone_score'] *
            df['emoji_score'] *
            df['length_score'] *
            df['random_factor']
        )

        # Normalize to 0-1
        min_score = df['engagement_score'].min()
        max_score = df['engagement_score'].max()
        df['engagement_score'] = (df['engagement_score'] - min_score) / (max_score - min_score)

        logger.info("Added synthetic engagement metrics")

    return df

def extract_features(df):
    """
    Extract and prepare all available features for modeling
    """
    logger.info("Extracting and preparing features")

    features = {}

    # 1. Basic text features
    if 'post_text' in df.columns:
        features['text_length'] = df['post_text'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        features['word_count'] = df['post_text'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
        features['avg_word_length'] = df['post_text'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if isinstance(x, str) and x.split() else 0
        )

    # 2. Structural features
    if 'structure' in df.columns:
        features['structure'] = df['structure']

    # 3. Style features
    if 'tone' in df.columns:
        features['tone'] = df['tone']

    if 'emoji_usage' in df.columns:
        features['emoji_usage'] = df['emoji_usage']

    # 4. Advanced features
    if 'vocabulary_usage' in df.columns:
        features['vocabulary_usage'] = df['vocabulary_usage']

    if 'line_breaks' in df.columns:
        features['line_breaks'] = df['line_breaks']

    if 'avg_line_breaks' in df.columns:
        features['avg_line_breaks'] = df['avg_line_breaks']

    # 5. Bullet features
    if 'bullet_styles' in df.columns:
        features['bullet_styles'] = df['bullet_styles']

    # 6. Narrative features
    if 'flow' in df.columns:
        # Extract the first flow element as a feature
        features['narrative_flow'] = df['flow'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
        )

    if 'pacing' in df.columns:
        features['pacing'] = df['pacing']

    if 'sentiment_arc' in df.columns:
        features['sentiment_arc'] = df['sentiment_arc']

    # 7. Content features
    if 'profanity' in df.columns:
        features['profanity'] = df['profanity']

    # 8. Punctuation features
    if 'punctuation_usage' in df.columns:
        # Extract counts for common punctuation
        for punct in ['.', ',', '!', '?', ';', ':']:
            features[f'punct_{punct}'] = df['punctuation_usage'].apply(
                lambda x: x.get(punct, 0) if isinstance(x, dict) else 0
            )

    # 9. Topic shifts
    if 'topic_shifts' in df.columns:
        features['topic_shift_count'] = df['topic_shifts'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

        # Calculate average shift score
        features['avg_shift_score'] = df['topic_shifts'].apply(
            lambda x: np.mean([shift.get('shift_score', 0) for shift in x])
            if isinstance(x, list) and len(x) > 0 else 0
        )

    # Convert to DataFrame
    feature_df = pd.DataFrame(features)
    logger.info(f"Extracted {len(feature_df.columns)} features")

    return feature_df

def encode_categorical_features(feature_df):
    """
    One-hot encode categorical features and scale numerical features
    """
    logger.info("Encoding categorical features")

    # Identify categorical features
    categorical_features = [
        col for col in feature_df.columns
        if feature_df[col].dtype == 'object' or col in [
            'structure', 'tone', 'emoji_usage', 'bullet_styles',
            'narrative_flow', 'pacing', 'sentiment_arc', 'profanity'
        ]
    ]

    # Identify numerical features
    numerical_features = [
        col for col in feature_df.columns
        if col not in categorical_features
    ]

    logger.info(f"Categorical features: {', '.join(categorical_features)}")
    logger.info(f"Numerical features: {', '.join(numerical_features)}")

    # Create empty DataFrames for encoded features
    encoded_categorical = pd.DataFrame(index=feature_df.index)
    encoded_numerical = pd.DataFrame(index=feature_df.index)

    # One-hot encode each categorical feature
    for feature in categorical_features:
        # Skip if feature is missing in most rows
        if feature_df[feature].isna().mean() > 0.8:
            logger.info(f"Skipping {feature} due to >80% missing values")
            continue

        # Replace NaN with 'unknown'
        feature_df[feature] = feature_df[feature].fillna('unknown')

        # One-hot encode
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(feature_df[[feature]])

        # Create column names
        categories = encoder.categories_[0]
        column_names = [f"{feature}_{cat}" for cat in categories]

        # Add to encoded features
        encoded_df = pd.DataFrame(encoded, index=feature_df.index, columns=column_names)
        encoded_categorical = pd.concat([encoded_categorical, encoded_df], axis=1)

    # Scale numerical features
    scaler = StandardScaler()
    if numerical_features:
        # Fill NaN with 0
        for feature in numerical_features:
            feature_df[feature] = feature_df[feature].fillna(0)

        # Scale features
        scaled = scaler.fit_transform(feature_df[numerical_features])
        encoded_numerical = pd.DataFrame(
            scaled,
            index=feature_df.index,
            columns=numerical_features
        )

    # Combine encoded features
    encoded_df = pd.concat([encoded_categorical, encoded_numerical], axis=1)
    logger.info(f"Encoded features expanded to {len(encoded_df.columns)} columns")

    return encoded_df, numerical_features, list(encoded_categorical.columns)

def train_model(X, y):
    """
    Train a Random Forest model and return evaluation metrics
    """
    logger.info("Training Random Forest model")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Model trained with MSE: {mse:.4f}, R²: {r2:.4f}")

    return model, X_train, X_test, y_train, y_test

def calculate_feature_importance(model, X, feature_names):
    """
    Calculate and visualize feature importance from the model
    """
    logger.info("Calculating model-based feature importance")

    # Get feature importance from the model
    importances = model.feature_importances_

    # Create DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)

    return importance_df

def calculate_permutation_importance(model, X, y, feature_names):
    """
    Calculate permutation importance
    """
    logger.info("Calculating permutation importance (this may take a moment)")

    # Calculate permutation importance
    result = permutation_importance(
        model, X, y, n_repeats=5, random_state=42, n_jobs=-1
    )

    # Create DataFrame for visualization
    perm_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': result.importances_mean,
        'Std': result.importances_std
    })

    # Sort by importance
    perm_importance_df = perm_importance_df.sort_values('Importance', ascending=False)

    return perm_importance_df

def calculate_feature_correlations(df, encoded_features, target='engagement_score'):
    """
    Calculate correlations between features and target
    """
    logger.info("Calculating feature correlations")

    # Get all features plus target
    all_cols = list(encoded_features) + [target]
    correlation_df = df[all_cols].copy()

    # Calculate correlations with target
    correlations = correlation_df.corr()[target].drop(target)

    # Create DataFrame for visualization
    correlation_df = pd.DataFrame({
        'Feature': correlations.index,
        'Correlation': correlations.values
    })

    # Sort by absolute correlation
    correlation_df['Abs_Correlation'] = correlation_df['Correlation'].abs()
    correlation_df = correlation_df.sort_values('Abs_Correlation', ascending=False)

    return correlation_df.drop('Abs_Correlation', axis=1)

def analyze_feature_interactions(model, X, y, feature_names, top_n=5):
    """
    Analyze interactions between top features
    """
    logger.info("Analyzing feature interactions")

    # Get top features from model importance
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-top_n:]
    top_features = [feature_names[i] for i in top_indices]

    # Calculate correlation matrix for top features
    top_data = pd.DataFrame(X[:, top_indices], columns=top_features)
    top_data['target'] = y

    # Calculate correlation matrix
    corr_matrix = top_data.corr()

    return corr_matrix, top_features

def map_feature_categories(feature_names):
    """
    Map features to their categories for better visualization
    """
    category_mapping = {}

    # Define patterns for each category
    patterns = {
        'Structure': ['structure_'],
        'Tone': ['tone_'],
        'Emoji Usage': ['emoji_'],
        'Text Stats': ['text_length', 'word_count', 'avg_word_'],
        'Line Breaks': ['line_breaks'],
        'Punctuation': ['punct_'],
        'Bullet Styles': ['bullet_'],
        'Narrative': ['narrative_', 'flow_'],
        'Pacing': ['pacing_'],
        'Sentiment': ['sentiment_'],
        'Topic Shifts': ['topic_shift_'],
        'Vocabulary': ['vocabulary_']
    }

    # Map each feature to a category
    for feature in feature_names:
        for category, patterns_list in patterns.items():
            if any(pattern in feature for pattern in patterns_list):
                category_mapping[feature] = category
                break
        else:
            category_mapping[feature] = 'Other'

    return category_mapping

def visualize_importance_results(importance_df, perm_importance_df, correlation_df,
                               feature_categories, interaction_matrix=None, top_features=None):
    """
    Create comprehensive visualizations for feature importance analysis
    """
    os.makedirs('importance_results', exist_ok=True)

    # Use consistent color mapping for feature categories
    unique_categories = sorted(set(feature_categories.values()))
    color_map = dict(zip(unique_categories,
                         sns.color_palette('viridis', len(unique_categories))))

    # Add category and color to DataFrames
    for df in [importance_df, perm_importance_df, correlation_df]:
        df['Category'] = df['Feature'].map(lambda x: feature_categories.get(x, 'Other'))
        df['Color'] = df['Category'].map(color_map)

    # 1. Model Feature Importance
    plt.figure(figsize=(14, 10))
    # Take top 20 features
    top_n = min(20, len(importance_df))
    top_importance = importance_df.head(top_n)

    # Create horizontal bar chart
    bars = plt.barh(
        np.arange(len(top_importance)),
        top_importance['Importance'],
        color=top_importance['Color']
    )

    # Add feature names as y-tick labels
    plt.yticks(np.arange(len(top_importance)), top_importance['Feature'])

    # Add category legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=cat,
              markerfacecolor=color_map[cat], markersize=10)
        for cat in unique_categories if cat in top_importance['Category'].values
    ]
    plt.legend(handles=legend_elements, loc='lower right', title='Feature Category')

    plt.title('Top Features by Model Importance', fontsize=16)
    plt.xlabel('Importance Score', fontsize=14)
    plt.tight_layout()
    plt.savefig('importance_results/model_importance.png', dpi=300)
    plt.close()

    # 2. Permutation Importance
    plt.figure(figsize=(14, 10))
    # Take top 20 features
    top_n = min(20, len(perm_importance_df))
    top_perm = perm_importance_df.head(top_n)

    # Create horizontal bar chart with error bars
    bars = plt.barh(
        np.arange(len(top_perm)),
        top_perm['Importance'],
        xerr=top_perm['Std'],
        color=top_perm['Color']
    )

    # Add feature names as y-tick labels
    plt.yticks(np.arange(len(top_perm)), top_perm['Feature'])

    # Add category legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=cat,
              markerfacecolor=color_map[cat], markersize=10)
        for cat in unique_categories if cat in top_perm['Category'].values
    ]
    plt.legend(handles=legend_elements, loc='lower right', title='Feature Category')

    plt.title('Top Features by Permutation Importance', fontsize=16)
    plt.xlabel('Mean Accuracy Decrease When Feature Is Shuffled', fontsize=14)
    plt.tight_layout()
    plt.savefig('importance_results/permutation_importance.png', dpi=300)
    plt.close()

    # 3. Feature Correlations
    plt.figure(figsize=(14, 10))
    # Take top 20 features
    top_n = min(20, len(correlation_df))
    top_corr = correlation_df.head(top_n)

    # Create horizontal bar chart
    bars = plt.barh(
        np.arange(len(top_corr)),
        top_corr['Correlation'],
        color=[plt.cm.RdBu_r(0.5 + 0.5 * x / max(0.001, abs(top_corr['Correlation'].max())))
               for x in top_corr['Correlation']]
    )

    # Add feature names as y-tick labels
    plt.yticks(np.arange(len(top_corr)), top_corr['Feature'])

    # Add grid line at x=0
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    plt.title('Top Features by Correlation with Engagement', fontsize=16)
    plt.xlabel('Correlation Coefficient', fontsize=14)
    plt.tight_layout()
    plt.savefig('importance_results/feature_correlations.png', dpi=300)
    plt.close()

    # 4. Consolidated Importance Chart (comparing methods)
    # Merge top features from each method
    top_features_set = set()
    for df in [importance_df.head(10), perm_importance_df.head(10), correlation_df.head(10)]:
        top_features_set.update(df['Feature'])

    consolidated_df = pd.DataFrame({'Feature': list(top_features_set)})

    # Add importance scores from each method
    consolidated_df = consolidated_df.merge(
        importance_df[['Feature', 'Importance']],
        on='Feature', how='left'
    ).rename(columns={'Importance': 'Model_Importance'})

    consolidated_df = consolidated_df.merge(
        perm_importance_df[['Feature', 'Importance']],
        on='Feature', how='left'
    ).rename(columns={'Importance': 'Permutation_Importance'})

    consolidated_df = consolidated_df.merge(
        correlation_df[['Feature', 'Correlation']],
        on='Feature', how='left'
    )

    # Normalize each column to 0-1 scale
    for col in ['Model_Importance', 'Permutation_Importance', 'Correlation']:
        if col in consolidated_df.columns:
            if col == 'Correlation':
                consolidated_df[col] = consolidated_df[col].abs()
            max_val = consolidated_df[col].max()
            if max_val > 0:
                consolidated_df[col] = consolidated_df[col] / max_val

    # Calculate mean importance
    methods = [col for col in ['Model_Importance', 'Permutation_Importance', 'Correlation']
               if col in consolidated_df.columns]

    consolidated_df['Mean_Importance'] = consolidated_df[methods].mean(axis=1)
    consolidated_df = consolidated_df.sort_values('Mean_Importance', ascending=False)

    # Add categories
    consolidated_df['Category'] = consolidated_df['Feature'].map(
        lambda x: feature_categories.get(x, 'Other')
    )

    # Create grouped bar chart
    plt.figure(figsize=(14, 10))

    # Plot top 15 features
    top_n = min(15, len(consolidated_df))
    plot_df = consolidated_df.head(top_n).copy()

    # Reshape data for grouped bar chart
    plot_data = []
    for _, row in plot_df.iterrows():
        for method in methods:
            plot_data.append({
                'Feature': row['Feature'],
                'Method': method.replace('_', ' '),
                'Importance': row[method],
                'Category': row['Category']
            })

    plot_df = pd.DataFrame(plot_data)

    # Create grouped bar chart
    chart = sns.catplot(
        data=plot_df,
        kind='bar',
        x='Feature',
        y='Importance',
        hue='Method',
        palette='viridis',
        height=8,
        aspect=1.5
    )

    # Customize chart
    chart.set_xticklabels(rotation=45, ha='right')
    chart.fig.suptitle('Feature Importance Comparison Across Methods', fontsize=16)
    chart.set(xlabel='Feature', ylabel='Normalized Importance')

    # Add text labels for the feature categories
    for i, feature in enumerate(plot_df['Feature'].unique()):
        category = plot_df.loc[plot_df['Feature'] == feature, 'Category'].iloc[0]
        plt.text(i, -0.05, category, ha='center', fontsize=9, rotation=45)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('importance_results/consolidated_importance.png', dpi=300)
    plt.close()

    # 5. Feature Interaction Matrix (if available)
    if interaction_matrix is not None and top_features is not None:
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            interaction_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f'
        )
        plt.title('Feature Interaction Matrix (Correlation)', fontsize=16)
        plt.tight_layout()
        plt.savefig('importance_results/feature_interactions.png', dpi=300)
        plt.close()

    # 6. Importance by Feature Category
    # Group and aggregate by category
    category_importance = defaultdict(list)

    for _, row in importance_df.iterrows():
        category = feature_categories.get(row['Feature'], 'Other')
        category_importance[category].append(row['Importance'])

    category_means = {
        cat: np.mean(values) for cat, values in category_importance.items() if values
    }

    category_df = pd.DataFrame({
        'Category': list(category_means.keys()),
        'Importance': list(category_means.values())
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    bars = plt.bar(
        category_df['Category'],
        category_df['Importance'],
        color=[color_map.get(cat, 'gray') for cat in category_df['Category']]
    )

    plt.title('Feature Importance by Category', fontsize=16)
    plt.ylabel('Average Importance', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig('importance_results/category_importance.png', dpi=300)
    plt.close()

    logger.info("Saved visualizations to importance_results directory")

def run_analysis(run_id: str | None,
                base_dir: str,
                input_path: str | None,
                seed: int | None):
    """
    Main function to run the feature importance analysis
    """
    set_global_seed(seed)
    logger.info("Starting feature importance analysis")

    # Resolve input and reports dir
    in_path, reports_dir = _resolve_input(run_id, base_dir, input_path)

    # Idempotent skip via signature
    sig = compute_hash([in_path], {"stage": 20, "stage_version": STAGE_VERSION})
    if run_id:
        manifest = read_manifest(run_id, base_dir)
        out_paths = [
            os.path.join(reports_dir, 'importance_results', 'model_importance.png'),
            os.path.join(reports_dir, 'importance_results', 'permutation_importance.png'),
            os.path.join(reports_dir, 'importance_results', 'feature_correlations.png'),
            os.path.join(reports_dir, 'importance_results', 'consolidated_importance.png'),
            os.path.join(reports_dir, 'importance_results', 'feature_interactions.png'),
            os.path.join(reports_dir, 'importance_results', 'category_importance.png'),
            os.path.join(reports_dir, 'importance_results', 'feature_importance_details.txt'),
        ]
        if should_skip(manifest, "20-permutation-correlation", sig, out_paths):
            logger.info(f"Skipping 20-permutation-correlation; up-to-date in {reports_dir}")
            return

    # Ensure outputs in reports dir
    os.makedirs(os.path.join(reports_dir, 'importance_results'), exist_ok=True)
    cwd = os.getcwd()
    os.chdir(reports_dir)
    try:
        # Load and preprocess data
        df = load_and_add_synthetic_metrics(in_path)
        if df is None:
            return

        # Extract features
        feature_df = extract_features(df)

        # Add engagement score to feature DataFrame for analysis
        feature_df['engagement_score'] = df['engagement_score']

        # Encode categorical features
        encoded_features, numerical_features, categorical_features = encode_categorical_features(feature_df)

        # Prepare data for modeling
        X = encoded_features.drop('engagement_score', axis=1).values
        y = encoded_features['engagement_score'].values
        feature_names = encoded_features.drop('engagement_score', axis=1).columns.tolist()

        # Skip if no features or targets
        if len(feature_names) == 0 or len(y) == 0:
            logger.error("No features or target available for analysis")
            return

        # Train model
        model, X_train, X_test, y_train, y_test = train_model(X, y)

        # Calculate feature importance
        importance_df = calculate_feature_importance(model, X, feature_names)

        # Calculate permutation importance
        perm_importance_df = calculate_permutation_importance(model, X_test, y_test, feature_names)

        # Calculate feature correlations
        correlation_df = calculate_feature_correlations(encoded_features, feature_names)

        # Analyze feature interactions
        interaction_matrix, top_features = analyze_feature_interactions(model, X, y, feature_names)

        # Map features to categories
        feature_categories = map_feature_categories(feature_names)

        # Visualize results
        visualize_importance_results(
            importance_df,
            perm_importance_df,
            correlation_df,
            feature_categories,
            interaction_matrix,
            top_features
        )

        # Save detailed results to text file
        with open('importance_results/feature_importance_details.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FEATURE IMPORTANCE ANALYSIS RESULTS\n")
            f.write("=" * 80 + "\n\n")
            f.write("DATA SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Features analyzed: {len(feature_df.columns) - 1}\n")
            f.write(f"  - Numerical features: {len(numerical_features)}\n")
            f.write(f"  - Categorical features: {len(categorical_features) // 3}\n")
            f.write(f"Encoded features: {len(feature_names)}\n")
            f.write(f"Model performance: R² = {r2_score(y_test, model.predict(X_test)):.4f}\n\n")

            f.write("TOP 15 FEATURES BY MODEL IMPORTANCE\n")
            f.write("-" * 80 + "\n")
            for i, row in importance_df.head(15).iterrows():
                category = feature_categories.get(row['Feature'], 'Other')
                f.write(f"{i+1}. {row['Feature']} ({category}): {row['Importance']:.4f}\n")
            f.write("\n")

            f.write("TOP 15 FEATURES BY PERMUTATION IMPORTANCE\n")
            f.write("-" * 80 + "\n")
            for i, row in perm_importance_df.head(15).iterrows():
                category = feature_categories.get(row['Feature'], 'Other')
                f.write(f"{i+1}. {row['Feature']} ({category}): {row['Importance']:.4f} (±{row['Std']:.4f})\n")
            f.write("\n")

            f.write("TOP 15 FEATURES BY CORRELATION WITH ENGAGEMENT\n")
            f.write("-" * 80 + "\n")
            for i, row in correlation_df.head(15).iterrows():
                category = feature_categories.get(row['Feature'], 'Other')
                f.write(f"{i+1}. {row['Feature']} ({category}): {row['Correlation']:.4f}\n")
            f.write("\n")

            f.write("FEATURE IMPORTANCE BY CATEGORY\n")
            f.write("-" * 80 + "\n")
            category_importance = defaultdict(list)
            for _, row in importance_df.iterrows():
                category = feature_categories.get(row['Feature'], 'Other')
                category_importance[category].append(row['Importance'])
            category_means = {cat: np.mean(values) for cat, values in category_importance.items() if values}
            sorted_categories = sorted(category_means.items(), key=lambda x: x[1], reverse=True)
            for i, (category, importance) in enumerate(sorted_categories):
                count = len(category_importance[category])
                f.write(f"{i+1}. {category}: {importance:.4f} (features: {count})\n")
            f.write("\n")

            f.write("STRUCTURE-SPECIFIC FEATURE IMPORTANCE\n")
            f.write("-" * 80 + "\n")
            structure_features = [f for f in feature_names if 'structure_' in f]
            if structure_features:
                structure_importance = importance_df[importance_df['Feature'].isin(structure_features)]
                f.write("Model Importance:\n")
                for i, row in structure_importance.iterrows():
                    structure_type = row['Feature'].replace('structure_', '')
                    f.write(f"- {structure_type}: {row['Importance']:.4f}\n")
                f.write("\n")
            else:
                f.write("No structure-specific features found in the dataset.\n\n")

            f.write("TONE-SPECIFIC FEATURE IMPORTANCE\n")
            f.write("-" * 80 + "\n")
            tone_features = [f for f in feature_names if 'tone_' in f]
            if tone_features:
                tone_importance = importance_df[importance_df['Feature'].isin(tone_features)]
                f.write("Model Importance:\n")
                for i, row in tone_importance.iterrows():
                    tone_type = row['Feature'].replace('tone_', '')
                    f.write(f"- {tone_type}: {row['Importance']:.4f}\n")
                f.write("\n")
            else:
                f.write("No tone-specific features found in the dataset.\n\n")

        logger.info("Saved detailed results to importance_results/feature_importance_details.txt")
        logger.info("Feature importance analysis complete!")

        print("\nFeature Importance Analysis Results:")
        print(f"Top 5 Features by Model Importance:")
        for i, row in importance_df.head(5).iterrows():
            print(f"  {i+1}. {row['Feature']}: {row['Importance']:.4f}")
        print(f"\nTop 5 Features by Permutation Importance:")
        for i, row in perm_importance_df.head(5).iterrows():
            print(f"  {i+1}. {row['Feature']}: {row['Importance']:.4f}")
        print(f"\nTop 5 Features by Correlation:")
        for i, row in correlation_df.head(5).iterrows():
            print(f"  {i+1}. {row['Feature']}: {row['Correlation']:.4f}")
        print("\nResults saved to importance_results/ directory")
    finally:
        os.chdir(cwd)

    if run_id:
        manifest = read_manifest(run_id, base_dir)
        update_stage(run_id, base_dir, manifest, "20-permutation-correlation", in_path, [
            os.path.join(reports_dir, 'importance_results', 'model_importance.png'),
            os.path.join(reports_dir, 'importance_results', 'permutation_importance.png'),
            os.path.join(reports_dir, 'importance_results', 'feature_correlations.png'),
            os.path.join(reports_dir, 'importance_results', 'consolidated_importance.png'),
            os.path.join(reports_dir, 'importance_results', 'feature_interactions.png'),
            os.path.join(reports_dir, 'importance_results', 'category_importance.png'),
            os.path.join(reports_dir, 'importance_results', 'feature_importance_details.txt'),
        ], sig, extra={"rows": len(df), "encoded_features": len(feature_names)})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Permutation importance and correlations for features")
    parser.add_argument("--run-id", dest="run_id", default=None, help="Use 'latest' to use most recent run")
    parser.add_argument("--base-dir", dest="base_dir", default="data/processed")
    parser.add_argument("--input", dest="input_path", default=None, help="Only used when --run-id is not provided")
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    args = parser.parse_args()

    try:
        run_analysis(
            run_id=args.run_id,
            base_dir=args.base_dir,
            input_path=args.input_path,
            seed=args.seed,
        )
    except Exception as e:
        logger.error(f"Error in feature importance analysis: {e}", exc_info=True)
        print(f"Analysis failed with error: {e}")