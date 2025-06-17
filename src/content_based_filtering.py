"""
Content-Based Filtering implementation for the recommendation system.
Uses item features to recommend similar items.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import re

from .utils import setup_logging, get_top_k_items

logger = setup_logging()

class ContentBasedFiltering:
    """Content-Based Filtering Recommendation System."""
    
    def __init__(self, similarity_metric: str = 'cosine'):
        """
        Initialize Content-Based Filtering model.
        
        Args:
            similarity_metric: 'cosine' or 'linear'
        """
        self.similarity_metric = similarity_metric
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        self.scaler = StandardScaler()
        self.mlb = MultiLabelBinarizer()
        
        self.item_features = None
        self.feature_matrix = None
        self.similarity_matrix = None
        self.item_profiles = None
        
    def fit(self, movies_df: pd.DataFrame, ratings_df: Optional[pd.DataFrame] = None) -> None:
        """
        Fit the content-based filtering model.
        
        Args:
            movies_df: DataFrame with movie features
            ratings_df: Optional ratings DataFrame for user profiles
        """
        logger.info("Training Content-Based Filtering model...")
        
        self.item_features = movies_df.copy()
        
        # Prepare features
        self._prepare_features()
        
        # Calculate item similarity matrix
        self._calculate_similarity_matrix()
        
        # Build user profiles if ratings provided
        if ratings_df is not None:
            self._build_user_profiles(ratings_df)
        
        logger.info("Content-Based Filtering model trained successfully!")
    
    def _prepare_features(self) -> None:
        """Prepare and combine different item features."""
        features = []
        
        # Text features (title, genres)
        text_features = self._prepare_text_features()
        features.append(text_features)
        
        # Categorical features (genres as binary)
        if 'genres' in self.item_features.columns:
            genre_features = self._prepare_genre_features()
            features.append(genre_features)
        
        # Numerical features (year, etc.)
        numerical_features = self._prepare_numerical_features()
        if numerical_features is not None:
            features.append(numerical_features)
        
        # Combine all features
        self.feature_matrix = np.hstack(features)
        
        logger.info(f"Feature matrix shape: {self.feature_matrix.shape}")
    
    def _prepare_text_features(self) -> np.ndarray:
        """Prepare text-based features using TF-IDF."""
        # Combine title and genres for text analysis
        text_data = []
        
        for _, row in self.item_features.iterrows():
            text_parts = []
            
            if 'clean_title' in row and pd.notna(row['clean_title']):
                text_parts.append(str(row['clean_title']))
            elif 'title' in row and pd.notna(row['title']):
                # Clean title by removing year
                clean_title = re.sub(r'\s*\(\d{4}\)', '', str(row['title']))
                text_parts.append(clean_title)
            
            if 'genres' in row and pd.notna(row['genres']):
                # Replace | with space for genre processing
                genres = str(row['genres']).replace('|', ' ')
                text_parts.append(genres)
            
            text_data.append(' '.join(text_parts))
        
        # Apply TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_data)
        
        return tfidf_matrix.toarray()
    
    def _prepare_genre_features(self) -> np.ndarray:
        """Prepare genre features as binary vectors."""
        genres_list = []
        
        for genres in self.item_features['genres']:
            if pd.notna(genres):
                genre_list = str(genres).split('|')
            else:
                genre_list = []
            genres_list.append(genre_list)
        
        # Convert to binary matrix
        genre_matrix = self.mlb.fit_transform(genres_list)
        
        return genre_matrix.astype(float)
    
    def _prepare_numerical_features(self) -> Optional[np.ndarray]:
        """Prepare numerical features."""
        numerical_cols = []
        
        # Check for year column
        if 'year' in self.item_features.columns:
            numerical_cols.append('year')
        
        if not numerical_cols:
            return None
        
        # Fill missing values and scale
        numerical_data = self.item_features[numerical_cols].fillna(
            self.item_features[numerical_cols].mean()
        )
        
        scaled_data = self.scaler.fit_transform(numerical_data)
        
        return scaled_data
    
    def _calculate_similarity_matrix(self) -> None:
        """Calculate item similarity matrix."""
        if self.similarity_metric == 'cosine':
            self.similarity_matrix = cosine_similarity(self.feature_matrix)
        elif self.similarity_metric == 'linear':
            self.similarity_matrix = linear_kernel(self.feature_matrix)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        # Set diagonal to 0 to avoid self-similarity
        np.fill_diagonal(self.similarity_matrix, 0)
    
    def _build_user_profiles(self, ratings_df: pd.DataFrame) -> None:
        """Build user profiles based on rated items."""
        logger.info("Building user profiles...")
        
        self.item_profiles = {}
        
        # Group ratings by user
        user_ratings = ratings_df.groupby('user_id')
        
        for user_id, user_data in user_ratings:
            # Get weighted average of item features
            item_indices = user_data['item_id'].values
            ratings = user_data['rating'].values
            
            # Normalize ratings to use as weights
            weights = (ratings - ratings.min()) / (ratings.max() - ratings.min() + 1e-8)
            
            # Calculate weighted average of features
            user_profile = np.zeros(self.feature_matrix.shape[1])
            total_weight = 0
            
            for item_idx, weight in zip(item_indices, weights):
                if item_idx < len(self.feature_matrix):
                    user_profile += weight * self.feature_matrix[item_idx]
                    total_weight += weight
            
            if total_weight > 0:
                user_profile /= total_weight
            
            self.item_profiles[user_id] = user_profile
        
        logger.info(f"Built profiles for {len(self.item_profiles)} users")
    
    def get_similar_items(self, item_id: int, n_items: int = 10) -> List[Tuple[int, float]]:
        """
        Get items similar to a given item.
        
        Args:
            item_id: Item ID
            n_items: Number of similar items to return
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        if item_id >= len(self.similarity_matrix):
            return []
        
        similarities = self.similarity_matrix[item_id]
        similar_items = [(i, sim) for i, sim in enumerate(similarities) if i != item_id]
        
        return sorted(similar_items, key=lambda x: x[1], reverse=True)[:n_items]
    
    def recommend_items_by_profile(self, user_id: int, n_recommendations: int = 10,
                                  exclude_rated: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Recommend items based on user profile.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            exclude_rated: List of item IDs to exclude
            
        Returns:
            List of (item_id, predicted_score) tuples
        """
        if user_id not in self.item_profiles:
            return []
        
        user_profile = self.item_profiles[user_id]
        
        # Calculate similarity between user profile and all items
        item_scores = cosine_similarity([user_profile], self.feature_matrix)[0]
        
        # Create recommendations
        recommendations = {}
        for item_id, score in enumerate(item_scores):
            if exclude_rated and item_id in exclude_rated:
                continue
            recommendations[item_id] = score
        
        return get_top_k_items(recommendations, n_recommendations)
    
    def recommend_items_by_history(self, user_ratings: List[Tuple[int, float]], 
                                  n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Recommend items based on user's rating history.
        
        Args:
            user_ratings: List of (item_id, rating) tuples
            n_recommendations: Number of recommendations
            
        Returns:
            List of (item_id, predicted_score) tuples
        """
        item_scores = {}
        rated_items = [item_id for item_id, _ in user_ratings]
        
        # For each rated item, find similar items
        for item_id, rating in user_ratings:
            if item_id >= len(self.similarity_matrix):
                continue
            
            similar_items = self.get_similar_items(item_id, n_items=50)
            
            # Weight similarities by the user's rating
            weight = (rating - 1) / 4  # Normalize rating to 0-1
            
            for sim_item_id, similarity in similar_items:
                if sim_item_id not in rated_items:
                    if sim_item_id not in item_scores:
                        item_scores[sim_item_id] = 0
                    item_scores[sim_item_id] += weight * similarity
        
        return get_top_k_items(item_scores, n_recommendations)
    
    def get_item_features_explanation(self, item_id: int, top_features: int = 10) -> Dict:
        """Get explanation of item features."""
        if item_id >= len(self.feature_matrix):
            return {}
        
        item_vector = self.feature_matrix[item_id]
        
        # Get top TF-IDF features
        tfidf_features = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = item_vector[:len(tfidf_features)]
        
        top_tfidf_indices = np.argsort(tfidf_scores)[-top_features:]
        top_tfidf_features = [
            (tfidf_features[i], tfidf_scores[i]) 
            for i in top_tfidf_indices if tfidf_scores[i] > 0
        ]
        
        explanation = {
            'item_id': item_id,
            'top_text_features': sorted(top_tfidf_features, key=lambda x: x[1], reverse=True)
        }
        
        # Add genre information if available
        if hasattr(self, 'mlb'):
            genre_start_idx = len(tfidf_features)
            genre_end_idx = genre_start_idx + len(self.mlb.classes_)
            
            if genre_end_idx <= len(item_vector):
                genre_scores = item_vector[genre_start_idx:genre_end_idx]
                active_genres = [
                    self.mlb.classes_[i] for i, score in enumerate(genre_scores) if score > 0
                ]
                explanation['genres'] = active_genres
        
        return explanation
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'similarity_metric': self.similarity_metric,
            'n_items': len(self.item_features) if self.item_features is not None else 0,
            'feature_dimensions': self.feature_matrix.shape[1] if self.feature_matrix is not None else 0,
            'n_user_profiles': len(self.item_profiles) if self.item_profiles else 0,
            'tfidf_vocabulary_size': len(self.tfidf_vectorizer.vocabulary_) if hasattr(self.tfidf_vectorizer, 'vocabulary_') else 0
        }