"""
Streamlit Web Interface for the Hybrid Recommendation System.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import pickle
import os
import sys
from datetime import datetime
from difflib import get_close_matches
from collections import defaultdict
from contextlib import nullcontext

import requests

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.data_preprocessing import DataPreprocessor
from src.collaborative_filtering import CollaborativeFiltering
from src.content_based_filtering import ContentBasedFiltering
from src.hybrid_model import HybridRecommendationSystem
from src.evaluation import RecommendationEvaluator

# Page configuration
st.set_page_config(
    page_title="Hybrid Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/DevanshSrajput/hybrid-recommendation-system',
        'Report a bug': 'https://github.com/DevanshSrajput/hybrid-recommendation-system/issues',
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
/* Main header styling */
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.sub-header {
    font-size: 1.2rem;
    color: #666;
    text-align: center;
    margin-bottom: 2rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    margin: 0.5rem 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
}

.metric-label {
    font-size: 0.9rem;
    opacity: 0.8;
}

/* Recommendation cards */
.recommendation-card {
    background-color: #ffffff;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #1f77b4;
    margin-bottom: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transition: transform 0.2s;
}

.recommendation-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

/* Status indicators */
.status-success {
    background-color: #d4edda;
    color: #155724;
    padding: 0.75rem;
    border-radius: 5px;
    border-left: 4px solid #28a745;
}

.status-warning {
    background-color: #fff3cd;
    color: #856404;
    padding: 0.75rem;
    border-radius: 5px;
    border-left: 4px solid #ffc107;
}

.status-info {
    background-color: #d1ecf1;
    color: #0c5460;
    padding: 0.75rem;
    border-radius: 5px;
    border-left: 4px solid #17a2b8;
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem 0;
    color: #666;
    border-top: 1px solid #eee;
    margin-top: 3rem;
}

/* Sidebar styling */
.sidebar .sidebar-content {
    background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
}

/* Progress bars */
.progress-text {
    text-align: center;
    font-weight: bold;
    margin-bottom: 0.5rem;
}

/* Animation for loading */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid #f3f3f3;
    border-top: 3px solid #3498db;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

/* Modern navigation styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.75rem;
    justify-content: center;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.72);
    padding: 0.75rem 1.25rem;
    border-radius: 999px;
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #475467;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(99, 102, 241, 0.12);
    color: #4338ca;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #ffffff;
    box-shadow: 0 12px 25px rgba(99, 102, 241, 0.25);
    border-color: transparent;
}

/* Data control + training cards */
.control-panel-card,
.model-training-card {
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.08), rgba(129, 140, 248, 0.12));
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    border: 1px solid rgba(79, 70, 229, 0.18);
    box-shadow: 0 16px 35px rgba(79, 70, 229, 0.08);
    margin-bottom: 1.5rem;
}

.model-training-card {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(59, 130, 246, 0.12));
    border: 1px solid rgba(14, 165, 233, 0.18);
}

.control-panel-card h3,
.model-training-card h3 {
    margin: 0 0 0.35rem 0;
    font-weight: 700;
    color: #312e81;
}

.control-panel-card p,
.model-training-card p {
    margin: 0 0 1rem 0;
    color: #4b5563;
}

.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(16, 185, 129, 0.15);
    color: #047857;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
}

.status-chip.warning {
    background: rgba(234, 179, 8, 0.18);
    color: #92400e;
}

.control-label {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4c1d95;
    margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'poster_cache' not in st.session_state:
    st.session_state.poster_cache = {}
if 'dataset_size' not in st.session_state:
    st.session_state.dataset_size = "100k"
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

if isinstance(st.session_state.dataset_size, str):
    st.session_state.dataset_size = st.session_state.dataset_size.lower()

POSTER_PLACEHOLDER = "https://placehold.co/300x450?text=Poster+Unavailable"


def get_omdb_api_key() -> str | None:
    """Retrieve OMDb API key from environment or Streamlit secrets."""
    api_key = os.environ.get("OMDB_API_KEY")
    if api_key:
        return api_key
    try:
        if hasattr(st, "secrets") and "OMDB_API_KEY" in st.secrets:
            return st.secrets["OMDB_API_KEY"]
    except Exception:
        # Streamlit secrets may not be configured outside Streamlit runtime
        return None
    return None


def fetch_movie_poster(title: str, year: float | None = None) -> str:
    """Fetch movie poster URL using OMDb API or return a placeholder."""
    normalized_year = None
    if year is not None and not pd.isna(year):
        try:
            normalized_year = int(year)
        except (ValueError, TypeError):
            normalized_year = None

    cache_key = (title.lower(), normalized_year)
    if cache_key in st.session_state.poster_cache:
        return st.session_state.poster_cache[cache_key]

    api_key = get_omdb_api_key()
    if not api_key:
        st.session_state.poster_cache[cache_key] = POSTER_PLACEHOLDER
        return POSTER_PLACEHOLDER

    params = {"t": title, "apikey": api_key}
    if normalized_year:
        params["y"] = str(normalized_year)

    try:
        response = requests.get("https://www.omdbapi.com/", params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        poster_url = data.get("Poster") if data.get("Response") == "True" else None
        if poster_url and poster_url != "N/A":
            st.session_state.poster_cache[cache_key] = poster_url
            return poster_url
    except Exception:
        pass

    st.session_state.poster_cache[cache_key] = POSTER_PLACEHOLDER
    return POSTER_PLACEHOLDER


def format_dataset_label(value: str) -> str:
    mapping = {"100k": "100K", "1m": "1M", "10m": "10M"}
    return mapping.get(str(value).lower(), str(value).upper())


def render_recommendation_cards(recommendations, data):
    """Render recommendation cards with posters."""
    if not recommendations:
        st.warning("No recommendations found for the current input.")
        return

    movies_df = data.get('movies', pd.DataFrame())
    columns_per_row = 5
    for row_start in range(0, len(recommendations), columns_per_row):
        cols = st.columns(columns_per_row)
        for col, rec in zip(cols, recommendations[row_start:row_start + columns_per_row]):
            item_id, score, method = rec
            movie_info = movies_df[movies_df['item_id'] == item_id]
            year = None
            base_title = f"Movie ID {item_id}"
            if movie_info.empty:
                display_title = base_title
                genres = ""
            else:
                movie_row = movie_info.iloc[0]
                display_title = format_display_title(movie_row)
                base_title = movie_row.get('clean_title') or movie_row.get('title') or display_title
                genres = movie_row.get('genres') or ''
                year = movie_row.get('year')

            poster_url = fetch_movie_poster(str(base_title), year)

            with col:
                st.image(poster_url, width=220)
                st.markdown(f"**{display_title}**")
                if genres:
                    st.caption(genres)
                st.caption(f"Score: {score:.3f} • {method.title()} model")


def format_display_title(movie_row: pd.Series) -> str:
    """Return a nicely formatted movie title with optional year."""
    if movie_row is None:
        return "Unknown Title"

    base_title = movie_row.get('clean_title') or movie_row.get('title') or f"Movie ID {movie_row.get('item_id')}"
    base_title = str(base_title)
    year = movie_row.get('year')
    if year is not None and not pd.isna(year):
        try:
            return f"{base_title} ({int(year)})"
        except (TypeError, ValueError):
            return base_title
    return base_title


def match_movie_title(query: str, movies_df: pd.DataFrame):
    """Match a movie title using exact or fuzzy matching."""
    query_lower = query.strip().lower()
    if not query_lower:
        return None, "Please enter a movie title.", "warning"

    if 'search_title' not in movies_df.columns:
        if 'clean_title' in movies_df.columns:
            search_source = movies_df['clean_title']
        elif 'title' in movies_df.columns:
            search_source = movies_df['title']
        else:
            search_source = pd.Series(["" for _ in range(len(movies_df))], index=movies_df.index)

        if 'title' in movies_df.columns:
            fallback = movies_df['title']
        else:
            fallback = pd.Series(["" for _ in range(len(movies_df))], index=movies_df.index)

        search_series = search_source.fillna(fallback).astype(str).str.lower()
        movies_df = movies_df.assign(search_title=search_series)

    exact_matches = movies_df[movies_df['search_title'] == query_lower]
    if not exact_matches.empty:
        return exact_matches.iloc[0], None, "info"

    potential_matches = get_close_matches(
        query_lower,
        movies_df['search_title'].dropna().tolist(),
        n=1,
        cutoff=0.6
    )

    if potential_matches:
        approx_matches = movies_df[movies_df['search_title'] == potential_matches[0]]
        if not approx_matches.empty:
            matched_row = approx_matches.iloc[0]
            return matched_row, (
                f"Couldn't find an exact match for `{query}`, showing results for `{format_display_title(matched_row)}` instead."
            ), "info"

    return None, f"We couldn't find `{query}` in the current dataset.", "warning"


def handle_load_data():
    """Load dataset according to the current selection."""
    dataset_size = str(st.session_state.get('dataset_size', '100k')).lower()
    label_map = {"100k": "100K", "1m": "1M", "10m": "10M"}
    display_size = label_map.get(dataset_size, dataset_size)
    try:
        with st.spinner(f"Loading MovieLens {display_size} dataset..."):
            data, error = load_data(dataset_size)
        if error:
            st.session_state.error_message = error
            st.error(f"❌ Error loading data: {error}")
        else:
            st.session_state.data = data
            st.session_state.data_loaded = True
            st.session_state.models_trained = False
            st.session_state.models = {}
            st.session_state.poster_cache = {}
            st.session_state.last_update = datetime.now()
            st.session_state.error_message = None
            st.success(f"✅ Loaded MovieLens {display_size} dataset successfully!")
            st.rerun()
    except Exception as e:
        st.session_state.error_message = str(e)
        st.error(f"❌ Error loading data: {e}")


def compute_model_comparison(data, models, use_spinner: bool = True):
    """Compute and cache model comparison metrics."""
    if not models or data is None:
        return None

    comparison_key = (
        data.get("metadata", {}).get("dataset_size"),
        len(data.get("test_ratings", [])),
        len(data.get("train_ratings", [])),
    )

    cached_key = st.session_state.get("model_comparison_key")
    cached_df = st.session_state.get("model_comparison")
    if cached_df is not None and cached_key == comparison_key:
        if not st.session_state.get("model_comparison_summary"):
            st.session_state["model_comparison_summary"] = describe_best_model(cached_df)
        return cached_df

    evaluator = RecommendationEvaluator()
    spinner_message = "Evaluating model performance across collaborative, content-based, and hybrid engines..."
    spinner_context = st.spinner(spinner_message) if use_spinner else nullcontext()
    with spinner_context:
        try:
            comparison_df = evaluator.compare_models(
                models,
                data["test_ratings"],
                data["movies"],
            )
        except Exception as exc:
            st.error(f"Unable to evaluate models: {exc}")
            return None

    st.session_state["model_comparison"] = comparison_df
    st.session_state["model_comparison_key"] = comparison_key
    st.session_state["model_comparison_summary"] = describe_best_model(comparison_df)
    return comparison_df


def describe_best_model(comparison_df: pd.DataFrame) -> str:
    """Create a short natural-language summary of the top model."""
    if comparison_df is None or comparison_df.empty:
        return ""

    priority_metrics = [metric for metric in ["f1@10", "precision@10", "recall@10"] if metric in comparison_df.columns]
    if not priority_metrics:
        return ""

    chosen_metric = priority_metrics[0]
    best_idx = comparison_df[chosen_metric].idxmax()
    best_value = comparison_df.loc[best_idx, chosen_metric]
    model_label = str(best_idx).replace('_', ' ').title()

    rmse_text = ""
    if "rmse" in comparison_df.columns and not pd.isna(comparison_df.loc[best_idx, "rmse"]):
        rmse_text = f" with RMSE {comparison_df.loc[best_idx, 'rmse']:.3f}"

    return (
        f"**{model_label}** is leading based on {chosen_metric.upper()} = {best_value:.3f}{rmse_text}. "
        "Consider it the default choice unless you prioritize another metric."
    )


def render_data_controls():
    """Render dataset selection and load controls in a modern card."""
    dataset_options = ["100k", "1m", "10m"]
    load_triggered = False

    with st.container():
        st.markdown('<div class="control-panel-card">', unsafe_allow_html=True)
        st.markdown("<h3>📦 Data Pipeline</h3>", unsafe_allow_html=True)
        st.markdown("<p>Choose a dataset slice and refresh preprocessing whenever you need fresh insights.</p>", unsafe_allow_html=True)
        cols = st.columns([2, 1, 1], gap="large")
        with cols[0]:
            st.markdown('<span class="control-label">Dataset Variant</span>', unsafe_allow_html=True)
            current_value = st.session_state.dataset_size if st.session_state.dataset_size in dataset_options else "100k"
            try:
                current_index = dataset_options.index(current_value)
            except ValueError:
                current_index = 0
            selected_option = st.selectbox(
                "Dataset size",
                dataset_options,
                index=current_index,
                label_visibility="collapsed",
                format_func=format_dataset_label,
            )
            if selected_option != current_value:
                st.session_state.dataset_size = selected_option
        with cols[1]:
            st.markdown('<span class="control-label">Actions</span>', unsafe_allow_html=True)
            load_triggered = st.button("🚀 Load / Refresh", key="load_data_button", use_container_width=True)
        with cols[2]:
            st.markdown('<span class="control-label">Status</span>', unsafe_allow_html=True)
            last_update = st.session_state.get('last_update')
            if st.session_state.get('data_loaded') and isinstance(last_update, datetime):
                formatted = last_update.strftime("%b %d, %Y · %H:%M")
                st.markdown(f'<div class="status-chip">🕒 Last sync: {formatted}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-chip warning">📂 Dataset not loaded</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if load_triggered:
        handle_load_data()


def train_models(data):
    """Train all recommendation models and persist them in session state."""
    if data is None:
        st.error("No data available to train models. Load a dataset first.")
        return
    try:
        with st.spinner("Training models..."):
            models = {}
            models['collaborative'] = CollaborativeFiltering()
            models['collaborative'].fit(data['user_item_matrix'])

            models['content_based'] = ContentBasedFiltering()
            models['content_based'].fit(data['movies'], data['train_ratings'])

            models['hybrid'] = HybridRecommendationSystem()
            models['hybrid'].fit(data['user_item_matrix'], data['train_ratings'], data['movies'])

            st.session_state.models = models
            st.session_state.models_trained = True
            st.session_state.pop('model_comparison', None)
            st.session_state.pop('model_comparison_key', None)
            st.session_state.pop('model_comparison_summary', None)
        st.success("✅ Models trained successfully!")
        compute_model_comparison(data, models)
        st.rerun()
    except Exception as e:
        st.error(f"❌ Error training models: {e}")


def render_model_training_panel(data):
    """Display model training status with a modern card layout."""
    with st.container():
        st.markdown('<div class="model-training-card">', unsafe_allow_html=True)
        st.markdown("<h3>⚙️ Model Orchestration</h3>", unsafe_allow_html=True)
        st.markdown("<p>Train or refresh the hybrid engine to unlock personalized movie insights.</p>", unsafe_allow_html=True)
        col_status, col_action = st.columns([3, 1], gap="large")

        with col_status:
            if st.session_state.get('models_trained'):
                st.markdown('<div class="status-chip">✅ Models ready</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-chip warning">⚠️ Training required</div>', unsafe_allow_html=True)

        with col_action:
            if st.session_state.get('models_trained'):
                st.button(
                    "🔁 Retrain",
                    key="retrain_models",
                    use_container_width=True,
                    on_click=train_models,
                    args=(data,)
                )
            else:
                st.button(
                    "🚀 Train Models",
                    type="primary",
                    key="train_models_primary",
                    use_container_width=True,
                    on_click=train_models,
                    args=(data,)
                )

        stats_df = st.session_state.get('model_comparison')
        if isinstance(stats_df, pd.DataFrame) and not stats_df.empty:
            st.markdown("<hr style='margin:1rem 0 1.2rem 0; border: none; border-top: 1px solid rgba(79, 70, 229, 0.25);' />", unsafe_allow_html=True)
            st.markdown("##### Latest Model Snapshot")
            default_metrics = ["precision@10", "recall@10", "f1@10", "rmse"]
            available_metrics = [metric for metric in default_metrics if metric in stats_df.columns]
            preview_df = stats_df[available_metrics] if available_metrics else stats_df
            st.dataframe(preview_df.astype(float).round(4), use_container_width=True, height=180)

            summary_text = st.session_state.get('model_comparison_summary')
            if summary_text:
                st.markdown(summary_text)
        st.markdown('</div>', unsafe_allow_html=True)

# Define page functions BEFORE they are called
def show_dashboard(data):
    """Show main dashboard."""
    st.header("📊 System Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", f"{data['all_ratings']['user_id'].nunique():,}")
    
    with col2:
        st.metric("Total Movies", f"{len(data['movies']):,}")
    
    with col3:
        st.metric("Total Ratings", f"{len(data['all_ratings']):,}")
    
    with col4:
        avg_rating = data['all_ratings']['rating'].mean()
        st.metric("Avg Rating", f"{avg_rating:.2f}")

    metadata = data.get("metadata", {})
    if metadata:
        st.markdown("---")
        dens_col, sparse_col, thresh_col = st.columns(3)
        with dens_col:
            density = metadata.get("density")
            if density is not None:
                st.metric("Interaction Density", f"{density*100:.2f}%")
        with sparse_col:
            sparsity = metadata.get("sparsity")
            if sparsity is not None:
                st.metric("Sparsity", f"{sparsity*100:.2f}%")
        with thresh_col:
            st.metric(
                "Min Interactions",
                f"Users ≥ {metadata.get('min_user_interactions', '—')} | Items ≥ {metadata.get('min_item_interactions', '—')}"
            )

def show_recommendations_page(models, data):
    """Show recommendations page."""
    st.header("🎯 Get Recommendations")

    if not get_omdb_api_key():
        st.info(
            "Add an `OMDB_API_KEY` to your environment or Streamlit secrets to see official posters. "
            "We'll fall back to placeholders when the key is missing."
        )

    tab_user, tab_movie = st.tabs(["By User ID", "By Movie Title"])

    with tab_user:
        st.subheader("Personalized for a user")
        user_id = st.number_input(
            "Enter User ID:",
            min_value=0,
            max_value=int(data['all_ratings']['user_id'].max()),
            key="user_id_input"
        )
        model_choice = st.selectbox(
            "Choose Model:",
            ["hybrid", "collaborative", "content_based"],
            key="user_model_choice"
        )
        num_recs = st.slider(
            "Number of Recommendations:",
            1,
            20,
            10,
            key="user_num_recs"
        )

        if st.button("Get Recommendations", key="user_recs_button"):
            with st.spinner("Generating recommendations..."):
                try:
                    model = models[model_choice]
                    recommendations = model.recommend_items(int(user_id), int(num_recs))
                    st.subheader(f"Top {num_recs} picks for User {int(user_id)}")
                    render_recommendation_cards(recommendations, data)
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")

    with tab_movie:
        st.subheader("Find movies similar to your favorites")

        movies_df = data['movies'].copy()

        if movies_df.empty:
            st.warning("Movie metadata is unavailable. Load the dataset to explore movie-based recommendations.")
        else:
            if 'clean_title' in movies_df.columns:
                search_source = movies_df['clean_title']
            else:
                search_source = movies_df['title']

            fallback_titles = movies_df['title'] if 'title' in movies_df.columns else pd.Series(["" for _ in range(len(movies_df))], index=movies_df.index)
            movies_df['search_title'] = search_source.fillna(fallback_titles).astype(str).str.lower()
            movies_df['display_title'] = movies_df.apply(format_display_title, axis=1)
            movies_df = movies_df.dropna(subset=['display_title'])

            unique_display_df = movies_df.drop_duplicates('display_title', keep='first')
            display_to_row = {row['display_title']: row for _, row in unique_display_df.iterrows()}
            movie_options = sorted(display_to_row.keys())

            selected_display_titles = st.multiselect(
                "Select favorite movies (choose one or more):",
                movie_options,
                key="movie_multiselect"
            )

            custom_titles_input = st.text_input(
                "Optional: add additional titles (comma-separated)",
                placeholder="e.g. Toy Story, The Matrix",
                key="movie_title_input"
            )

            num_similar = st.slider(
                "Number of recommendation results:",
                1,
                20,
                8,
                key="movie_num_recs"
            )

            if st.button("Get Movie-based Recommendations", key="movie_recs_button"):
                favorite_rows = []
                messages = {"info": set(), "warning": set()}

                for display_title in selected_display_titles:
                    movie_row = display_to_row.get(display_title)
                    if movie_row is not None:
                        favorite_rows.append(movie_row)

                custom_titles = [title.strip() for title in custom_titles_input.split(',') if title.strip()]

                for custom_title in custom_titles:
                    matched_row, message, level = match_movie_title(custom_title, movies_df)
                    if matched_row is not None:
                        favorite_rows.append(matched_row)
                    if message:
                        messages[level].add(message)

                # Remove duplicate favorites by item_id
                unique_favorites = {}
                for row in favorite_rows:
                    item_id = int(row['item_id'])
                    if item_id not in unique_favorites:
                        unique_favorites[item_id] = row
                favorite_rows = list(unique_favorites.values())

                if not favorite_rows:
                    st.warning("Please select or enter at least one movie title to continue.")
                else:
                    for msg in sorted(messages['info']):
                        st.info(msg)
                    for msg in sorted(messages['warning']):
                        st.warning(msg)

                    st.markdown("#### Your favorites")
                    favorite_cards = [
                        (int(row['item_id']), 5.0, 'favorite selection')
                        for row in favorite_rows
                    ]
                    render_recommendation_cards(favorite_cards, data)

                    content_model = models.get('content_based')
                    if content_model is None:
                        st.error("Content-based model isn't available. Please train models first.")
                        return

                    favorite_ids = {int(row['item_id']) for row in favorite_rows}
                    aggregated_scores = defaultdict(float)

                    with st.spinner("Searching for similar movies..."):
                        try:
                            search_depth = max(int(num_similar) * 5, int(num_similar) + 1)
                            for row in favorite_rows:
                                similar_items = content_model.get_similar_items(
                                    int(row['item_id']),
                                    n_items=search_depth
                                )
                                for sim_item_id, sim_score in similar_items:
                                    sim_item_id = int(sim_item_id)
                                    if sim_item_id in favorite_ids:
                                        continue
                                    aggregated_scores[sim_item_id] += float(sim_score)
                        except Exception as e:
                            st.error(f"Error finding similar movies: {e}")
                            return

                    if not aggregated_scores:
                        st.warning("We couldn't find similar movies for the selected titles. Try different favorites or increase the dataset size.")
                    else:
                        ranked_items = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)[:int(num_similar)]
                        recommendations = [
                            (int(item_id), float(score), 'content-based')
                            for item_id, score in ranked_items
                        ]
                        st.markdown("#### Recommendations inspired by your picks")
                        render_recommendation_cards(recommendations, data)

def show_analytics_page(data):
    """Show analytics page."""
    st.header("📊 Data Analytics")

    models_available = st.session_state.get('models_trained') and st.session_state.get('models')
    if models_available:
        st.subheader("Model Performance Comparison")
        comparison_df = compute_model_comparison(data, st.session_state.get('models'))
        if comparison_df is not None and not comparison_df.empty:
            display_df = comparison_df.copy()
            numeric_cols = display_df.select_dtypes(include=[np.number]).columns
            display_df[numeric_cols] = display_df[numeric_cols].astype(float).round(4)
            st.dataframe(display_df, use_container_width=True)

            metric_candidates = [m for m in ["precision@10", "recall@10", "f1@10"] if m in display_df.columns]
            if metric_candidates:
                comparison_long = display_df[metric_candidates].reset_index(names="Model").melt(
                    id_vars="Model",
                    value_vars=metric_candidates,
                    var_name="Metric",
                    value_name="Score",
                )
                fig = px.bar(
                    comparison_long,
                    x="Model",
                    y="Score",
                    color="Metric",
                    barmode="group",
                    title="Key Top-K Metrics",
                )
                fig.update_layout(legend_title_text="Metric", yaxis_title="Score")
                st.plotly_chart(fig, use_container_width=True)

            summary = describe_best_model(display_df)
            if summary:
                st.success(summary)

            st.markdown("---")
        else:
            st.info("Model evaluation metrics will appear here after a successful training run.")
    else:
        st.info("Train the models from the Model Orchestration panel to unlock comparative analytics.")
    
    # Rating distribution
    fig = px.histogram(data['all_ratings'], x='rating', title='Rating Distribution')
    st.plotly_chart(fig, width=1000)
    
    # Most popular movies
    popular_movies = data['all_ratings'].groupby('item_id').size().reset_index(name='count')
    popular_movies = popular_movies.merge(data['movies'], on='item_id')
    popular_movies = popular_movies.nlargest(10, 'count')
    
    st.subheader("Most Popular Movies")
    st.dataframe(popular_movies[['title', 'count']], width="stretch")

def show_explore_data_page(data):
    """Show data exploration page."""
    st.header("🔍 Explore Data")
    
    tab1, tab2, tab3 = st.tabs(["Ratings", "Movies", "Statistics"])
    
    with tab1:
        st.subheader("Ratings Data")
        st.dataframe(data['all_ratings'].head(100), width="stretch")
    
    with tab2:
        st.subheader("Movies Data")
        st.dataframe(data['movies'].head(100), width="stretch")
    
    with tab3:
        st.subheader("Dataset Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Sparsity", f"{(1 - data['user_item_matrix'].astype(bool).sum().sum() / data['user_item_matrix'].size)*100:.2f}%")
        
        with col2:
            st.metric("Avg Ratings per User", f"{data['all_ratings'].groupby('user_id').size().mean():.1f}")

def show_about_page():
    """Show about page."""
    st.header("ℹ️ About")
    st.markdown("""
    ## Hybrid Recommendation System
    
    This system combines multiple recommendation approaches:
    
    - **Collaborative Filtering**: Recommends based on user similarity
    - **Content-Based Filtering**: Recommends based on item features
    - **Hybrid Approach**: Combines both methods for better accuracy
    
    ### Features:
    - Real-time recommendations
    - Multiple algorithms
    - Performance evaluation
    - Interactive web interface
    

    """)

@st.cache_data(show_spinner=False)
def load_data(dataset_size="100k"):
    """Load and cache data with progress indication."""
    try:
        preprocessor = DataPreprocessor()
        data = preprocessor.preprocess_all(dataset_size)
        return data, None
    except Exception as e:
        return None, str(e)

# Main header
st.markdown('<h1 class="main-header">🎬 Hybrid Recommendation System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Movie Recommendations</p>', unsafe_allow_html=True)

render_data_controls()

# Main app logic
if st.session_state.data_loaded:
    data = st.session_state.get('data')
    if data is None:
        st.error("Data is not available. Refresh the dataset via the Data Pipeline panel above.")
        st.stop()

    render_model_training_panel(data)

    tabs = st.tabs(["🏠 Dashboard", "🎯 Recommendations", "📊 Analytics", "🔍 Explore Data", "ℹ️ About"])

    with tabs[0]:
        show_dashboard(data)

    with tabs[1]:
        if st.session_state.get('models_trained') and st.session_state.get('models'):
            show_recommendations_page(st.session_state.models, data)
        else:
            st.warning("Train the models using the panel above to unlock personalized recommendations.")

    with tabs[2]:
        show_analytics_page(data)

    with tabs[3]:
        show_explore_data_page(data)

    with tabs[4]:
        show_about_page()

else:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="status-info">
        <h3>🚀 Getting Started</h3>
        <p>Welcome to the Hybrid Recommendation System! Use the Data Pipeline panel above to load a dataset and begin exploring AI-powered movie recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

# Add footer
st.markdown("""
<div class="footer">
    <p>© 2024 Hybrid Recommendation System | Built with ❤️ </p>
</div>
""", unsafe_allow_html=True)