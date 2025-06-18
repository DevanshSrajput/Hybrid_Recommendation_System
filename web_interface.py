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

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_preprocessing import DataPreprocessor
from collaborative_filtering import CollaborativeFiltering
from content_based_filtering import ContentBasedFiltering
from hybrid_model import HybridRecommendationSystem
from evaluation import RecommendationEvaluator

# Page configuration
st.set_page_config(
    page_title="Hybrid Recommendation System - DevanshSrajput",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/DevanshSrajput/hybrid-recommendation-system',
        'Report a bug': 'https://github.com/DevanshSrajput/hybrid-recommendation-system/issues',
        'About': 'Hybrid Recommendation System by DevanshSrajput'
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

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

def show_recommendations_page(models, data):
    """Show recommendations page."""
    st.header("🎯 Get Recommendations")
    
    user_id = st.number_input("Enter User ID:", min_value=0, max_value=data['all_ratings']['user_id'].max())
    model_choice = st.selectbox("Choose Model:", ["hybrid", "collaborative", "content_based"])
    num_recs = st.slider("Number of Recommendations:", 1, 20, 10)
    
    if st.button("Get Recommendations"):
        with st.spinner("Generating recommendations..."):
            try:
                model = models[model_choice]
                recommendations = model.recommend_items(user_id, num_recs)
                
                st.subheader(f"Top {num_recs} Recommendations for User {user_id}")
                
                for i, (item_id, score, method) in enumerate(recommendations, 1):
                    movie_info = data['movies'][data['movies']['item_id'] == item_id]
                    if not movie_info.empty:
                        title = movie_info.iloc[0]['title']
                        st.write(f"{i}. **{title}** (Score: {score:.3f})")
                    else:
                        st.write(f"{i}. Movie ID: {item_id} (Score: {score:.3f})")
                        
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

def show_analytics_page(data):
    """Show analytics page."""
    st.header("📊 Data Analytics")
    
    # Rating distribution
    fig = px.histogram(data['all_ratings'], x='rating', title='Rating Distribution')
    st.plotly_chart(fig, use_container_width=True)
    
    # Most popular movies
    popular_movies = data['all_ratings'].groupby('item_id').size().reset_index(name='count')
    popular_movies = popular_movies.merge(data['movies'], on='item_id')
    popular_movies = popular_movies.nlargest(10, 'count')
    
    st.subheader("Most Popular Movies")
    st.dataframe(popular_movies[['title', 'count']], use_container_width=True)

def show_explore_data_page(data):
    """Show data exploration page."""
    st.header("🔍 Explore Data")
    
    tab1, tab2, tab3 = st.tabs(["Ratings", "Movies", "Statistics"])
    
    with tab1:
        st.subheader("Ratings Data")
        st.dataframe(data['all_ratings'].head(100), use_container_width=True)
    
    with tab2:
        st.subheader("Movies Data")
        st.dataframe(data['movies'].head(100), use_container_width=True)
    
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
    
    ### Author: DevanshSrajput
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

# Sidebar for dataset selection
st.sidebar.header("Settings")
dataset_size = st.sidebar.selectbox(
    "Select dataset size:",
    ["100k", "1M", "10M"],
    index=0,
    help="Choose the size of the dataset to load."
)

# Load data button
if st.sidebar.button("🚀 Load Data", type="primary"):
    try:
        with st.spinner("Loading and preprocessing data..."):
            data, error = load_data(dataset_size)
            if error:
                st.session_state.error_message = error
                st.error(f"❌ Error loading data: {error}")
            else:
                st.session_state.data = data
                st.session_state.data_loaded = True
                st.success("✅ Data loaded successfully!")
                st.rerun()
    except Exception as e:
        st.session_state.error_message = str(e)
        st.error(f"❌ Error loading data: {e}")

# Main app logic
if st.session_state.data_loaded:
    data = st.session_state.data
    
    # Main header
    st.markdown('<h1 class="main-header">🎬 Hybrid Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Movie Recommendations by DevanshSrajput</p>', unsafe_allow_html=True)
    
    # Navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Select Page:",
        ["🏠 Dashboard", "🎯 Recommendations", "📊 Analytics", "🔍 Explore Data", "ℹ️ About"]
    )
    
    # Train models section
    st.sidebar.header("Model Training")
    
    if not st.session_state.models_trained:
        if st.sidebar.button("🤖 Train Models", type="primary"):
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
                    st.success("✅ Models trained successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error training models: {e}")
    else:
        st.sidebar.success("✅ Models trained!")
    
    # Page content
    if page == "🏠 Dashboard":
        show_dashboard(data)
    elif page == "🎯 Recommendations":
        if st.session_state.models_trained:
            show_recommendations_page(st.session_state.models, data)
        else:
            st.warning("Please train models first!")
    elif page == "📊 Analytics":
        show_analytics_page(data)
    elif page == "🔍 Explore Data":
        show_explore_data_page(data)
    elif page == "ℹ️ About":
        show_about_page()

else:
    # Main header for welcome screen
    st.markdown('<h1 class="main-header">🎬 Hybrid Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Movie Recommendations by DevanshSrajput</p>', unsafe_allow_html=True)
    
    # Add vertical space before the Getting Started block
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="status-info">
        <h3>🚀 Getting Started</h3>
        <p>Welcome to the Hybrid Recommendation System! Please load the dataset first using the sidebar to begin exploring AI-powered movie recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

# Add footer
st.markdown("""
<div class="footer">
    <p>© 2024 Hybrid Recommendation System | Built with ❤️ by DevanshSrajput</p>
</div>
""", unsafe_allow_html=True)