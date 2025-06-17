"""
Hybrid Recommendation System that combines collaborative filtering and content-based filtering.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from .collaborative_filtering import CollaborativeFiltering, MatrixFactorizationCF
from .content_based_filtering import ContentBasedFiltering
from .utils import setup_logging, get_top_k_items

logger = setup_logging()

class HybridRecommendationSystem:
    """Hybrid Recommendation System combining multiple approaches."""
    
    def __init__(self, 
                 combination_method: str = 'weighted',
                 cf_weight: float = 0.6,
                 cb_weight: float = 0.4,
                 cf_method: str = 'collaborative',
                 cf_similarity: str = 'cosine',
                 cb_similarity: str = 'cosine'):
        """
        Initialize Hybrid Recommendation System.
        
        Args:
            combination_method: 'weighted', 'switching', or 'mixed'
            cf_weight: Weight for collaborative filtering
            cb_weight: Weight for content-based filtering
            cf_method: 'collaborative' or 'matrix_factorization'
            cf_similarity: Similarity metric for collaborative filtering
            cb_similarity: Similarity metric for content-based filtering
        """
        self.combination_method = combination_method
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        
        # Normalize weights
        total_weight = self.cf_weight + self.cb_weight
        self.cf_weight /= total_weight
        self.cb_weight /= total_weight
        
        # Initialize models
        if cf_method == 'collaborative':
            self.cf_model = CollaborativeFiltering(
                method='user_based', 
                similarity_metric=cf_similarity
            )
        else:
            self.cf_model = MatrixFactorizationCF()
            
        self.cb_model = ContentBasedFiltering(similarity_metric=cb_similarity)
        
        # Store data for switching logic
        self.user_item_matrix = None
        self.ratings_df = None
        self.movies_df = None
        
    def fit(self, user_item_matrix: pd.DataFrame, 
            ratings_df: pd.DataFrame, 
            movies_df: pd.DataFrame) -> None:
        """
        Fit the hybrid model.
        
        Args:
            user_item_matrix: User-item interaction matrix
            ratings_df: Ratings DataFrame
            movies_df: Movies DataFrame with features
        """
        logger.info("Training Hybrid Recommendation System...")
        
        # Store data
        self.user_item_matrix = user_item_matrix
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        
        # Train collaborative filtering model
        logger.info("Training collaborative filtering component...")
        if isinstance(self.cf_model, MatrixFactorizationCF):
            self.cf_model.fit(ratings_df)
        else:
            self.cf_model.fit(user_item_matrix)
        
        # Train content-based filtering model
        logger.info("Training content-based filtering component...")
        self.cb_model.fit(movies_df, ratings_df)
        
        logger.info("Hybrid model training completed!")
    
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        if self.combination_method == 'weighted':
            return self._predict_weighted(user_id, item_id)
        elif self.combination_method == 'switching':
            return self._predict_switching(user_id, item_id)
        else:
            return self._predict_mixed(user_id, item_id)
    
    def _predict_weighted(self, user_id: int, item_id: int) -> float:
        """Predict using weighted combination."""
        cf_pred = self.cf_model.predict_rating(user_id, item_id)
        
        # For content-based prediction, we need the user's rating history
        user_ratings = self._get_user_ratings(user_id)
        if len(user_ratings) > 0:
            cb_recommendations = self.cb_model.recommend_items_by_history(
                user_ratings, n_recommendations=100
            )
            
            # Find the item in CB recommendations
            cb_pred = None
            for rec_item_id, score in cb_recommendations:
                if rec_item_id == item_id:
                    cb_pred = score * 5  # Scale score to rating range
                    break
            
            if cb_pred is None:
                cb_pred = self.cf_model.global_mean  # Fallback
        else:
            cb_pred = self.cf_model.global_mean
        
        # Combine predictions
        prediction = self.cf_weight * cf_pred + self.cb_weight * cb_pred
        return np.clip(prediction, 1, 5)
    
    def _predict_switching(self, user_id: int, item_id: int) -> float:
        """Predict using switching strategy based on data availability."""
        # Check if user has enough ratings for CF
        user_ratings = self._get_user_ratings(user_id)
        
        if len(user_ratings) >= 5:  # Enough data for CF
            return self.cf_model.predict_rating(user_id, item_id)
        else:  # Use CB for new users
            if len(user_ratings) > 0:
                cb_recommendations = self.cb_model.recommend_items_by_history(
                    user_ratings, n_recommendations=100
                )
                
                for rec_item_id, score in cb_recommendations:
                    if rec_item_id == item_id:
                        return score * 5  # Scale to rating range
            
            # Fallback to global mean
            return self.cf_model.global_mean
    
    def _predict_mixed(self, user_id: int, item_id: int) -> float:
        """Predict using mixed strategy (different weights based on confidence)."""
        user_ratings = self._get_user_ratings(user_id)
        
        # Adjust weights based on user's rating history
        if len(user_ratings) < 3:
            # New user - favor content-based
            cf_weight = 0.2
            cb_weight = 0.8
        elif len(user_ratings) > 20:
            # Active user - favor collaborative
            cf_weight = 0.8
            cb_weight = 0.2
        else:
            # Regular user - balanced
            cf_weight = self.cf_weight
            cb_weight = self.cb_weight
        
        # Get predictions
        cf_pred = self.cf_model.predict_rating(user_id, item_id)
        
        if len(user_ratings) > 0:
            cb_recommendations = self.cb_model.recommend_items_by_history(
                user_ratings, n_recommendations=100
            )
            
            cb_pred = self.cf_model.global_mean
            for rec_item_id, score in cb_recommendations:
                if rec_item_id == item_id:
                    cb_pred = score * 5
                    break
        else:
            cb_pred = self.cf_model.global_mean
        
        prediction = cf_weight * cf_pred + cb_weight * cb_pred
        return np.clip(prediction, 1, 5)
    
    def recommend_items(self, user_id: int, n_recommendations: int = 10,
                       exclude_rated: bool = True) -> List[Tuple[int, float, str]]:
        """
        Recommend items for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            exclude_rated: Whether to exclude already rated items
            
        Returns:
            List of (item_id, score, method) tuples
        """
        if self.combination_method == 'weighted':
            return self._recommend_weighted(user_id, n_recommendations, exclude_rated)
        elif self.combination_method == 'switching':
            return self._recommend_switching(user_id, n_recommendations, exclude_rated)
        else:
            return self._recommend_mixed(user_id, n_recommendations, exclude_rated)
    
    def _recommend_weighted(self, user_id: int, n_recommendations: int,
                           exclude_rated: bool) -> List[Tuple[int, float, str]]:
        """Recommend using weighted combination."""
        # Get CF recommendations
        cf_recommendations = self.cf_model.recommend_items(
            user_id, n_recommendations * 2, exclude_rated
        )
        
        # Get CB recommendations
        user_ratings = self._get_user_ratings(user_id)
        if len(user_ratings) > 0:
            rated_items = [item_id for item_id, _ in user_ratings] if exclude_rated else None
            
            if user_id in self.cb_model.item_profiles:
                cb_recommendations = self.cb_model.recommend_items_by_profile(
                    user_id, n_recommendations * 2, rated_items
                )
            else:
                cb_recommendations = self.cb_model.recommend_items_by_history(
                    user_ratings, n_recommendations * 2
                )
        else:
            cb_recommendations = []
        
        # Combine recommendations
        combined_scores = {}
        
        # Add CF scores
        for item_id, score in cf_recommendations:
            combined_scores[item_id] = self.cf_weight * score
        
        # Add CB scores
        for item_id, score in cb_recommendations:
            cb_score = score * 5  # Scale to rating range
            if item_id in combined_scores:
                combined_scores[item_id] += self.cb_weight * cb_score
            else:
                combined_scores[item_id] = self.cb_weight * cb_score
        
        # Get top recommendations
        recommendations = get_top_k_items(combined_scores, n_recommendations)
        
        # Add method information
        return [(item_id, score, 'hybrid') for item_id, score in recommendations]
    
    def _recommend_switching(self, user_id: int, n_recommendations: int,
                            exclude_rated: bool) -> List[Tuple[int, float, str]]:
        """Recommend using switching strategy."""
        user_ratings = self._get_user_ratings(user_id)
        
        if len(user_ratings) >= 5:
            # Use collaborative filtering
            recommendations = self.cf_model.recommend_items(
                user_id, n_recommendations, exclude_rated
            )
            return [(item_id, score, 'collaborative') for item_id, score in recommendations]
        else:
            # Use content-based filtering
            if len(user_ratings) > 0:
                recommendations = self.cb_model.recommend_items_by_history(
                    user_ratings, n_recommendations
                )
                return [(item_id, score * 5, 'content-based') for item_id, score in recommendations]
            else:
                # Return popular items (based on average ratings)
                item_popularity = self.ratings_df.groupby('item_id')['rating'].agg(['mean', 'count'])
                item_popularity = item_popularity[item_popularity['count'] >= 10]  # Minimum ratings
                popular_items = item_popularity.sort_values('mean', ascending=False).head(n_recommendations)
                
                return [(item_id, rating, 'popular') for item_id, rating in popular_items['mean'].items()]
    
    def _recommend_mixed(self, user_id: int, n_recommendations: int,
                        exclude_rated: bool) -> List[Tuple[int, float, str]]:
        """Recommend using mixed strategy."""
        # Use weighted approach with dynamic weights
        return self._recommend_weighted(user_id, n_recommendations, exclude_rated)
    
    def _get_user_ratings(self, user_id: int) -> List[Tuple[int, float]]:
        """Get user's rating history."""
        if self.ratings_df is None:
            return []
        
        user_data = self.ratings_df[self.ratings_df['user_id'] == user_id]
        return [(row['item_id'], row['rating']) for _, row in user_data.iterrows()]
    
    def get_explanation(self, user_id: int, item_id: int) -> Dict:
        """Get explanation for a recommendation."""
        explanation = {
            'user_id': user_id,
            'item_id': item_id,
            'combination_method': self.combination_method,
            'cf_weight': self.cf_weight,
            'cb_weight': self.cb_weight
        }
        
        # Get CF explanation
        if hasattr(self.cf_model, 'get_similar_users'):
            similar_users = self.cf_model.get_similar_users(user_id, k=5)
            explanation['similar_users'] = similar_users
        
        # Get CB explanation
        cb_explanation = self.cb_model.get_item_features_explanation(item_id)
        explanation['content_features'] = cb_explanation
        
        # Get movie information
        if self.movies_df is not None:
            movie_info = self.movies_df[self.movies_df['item_id'] == item_id]
            if not movie_info.empty:
                explanation['movie_info'] = movie_info.iloc[0].to_dict()
        
        return explanation
    
    def evaluate_cold_start(self, new_user_ratings: List[Tuple[int, float]], 
                           test_items: List[int]) -> Dict:
        """Evaluate performance for cold start users."""
        results = {}
        
        # Test content-based recommendations
        cb_recommendations = self.cb_model.recommend_items_by_history(
            new_user_ratings, n_recommendations=len(test_items)
        )
        
        cb_recommended_items = [item_id for item_id, _ in cb_recommendations]
        cb_precision = len(set(cb_recommended_items) & set(test_items)) / len(cb_recommended_items)
        cb_recall = len(set(cb_recommended_items) & set(test_items)) / len(test_items)
        
        results['content_based'] = {
            'precision': cb_precision,
            'recall': cb_recall,
            'f1': 2 * cb_precision * cb_recall / (cb_precision + cb_recall) if (cb_precision + cb_recall) > 0 else 0
        }
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information."""
        return {
            'combination_method': self.combination_method,
            'cf_weight': self.cf_weight,
            'cb_weight': self.cb_weight,
            'cf_model_info': self.cf_model.get_model_info() if hasattr(self.cf_model, 'get_model_info') else {},
            'cb_model_info': self.cb_model.get_model_info() if hasattr(self.cb_model, 'get_model_info') else {},
            'n_users': len(self.ratings_df['user_id'].unique()) if self.ratings_df is not None else 0,
            'n_items': len(self.ratings_df['item_id'].unique()) if self.ratings_df is not None else 0,
            'n_ratings': len(self.ratings_df) if self.ratings_df is not None else 0
        }