"""
Collaborative Filtering implementation for the recommendation system.
Includes both user-based and item-based collaborative filtering.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

from .utils import setup_logging, get_top_k_items

logger = setup_logging()

class CollaborativeFiltering:
    """Collaborative Filtering Recommendation System."""
    
    def __init__(self, method: str = 'user_based', similarity_metric: str = 'cosine'):
        """
        Initialize Collaborative Filtering model.
        
        Args:
            method: 'user_based' or 'item_based'
            similarity_metric: 'cosine', 'pearson', or 'euclidean'
        """
        self.method = method
        self.similarity_metric = similarity_metric
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user_means = None
        self.item_means = None
        self.global_mean = None
        
    def fit(self, user_item_matrix: pd.DataFrame) -> None:
        """
        Fit the collaborative filtering model.
        
        Args:
            user_item_matrix: User-item interaction matrix
        """
        logger.info(f"Training {self.method} collaborative filtering model...")
        
        self.user_item_matrix = user_item_matrix.copy()
        
        # Calculate means
        self.global_mean = self.user_item_matrix.values[self.user_item_matrix.values > 0].mean()
        self.user_means = self.user_item_matrix.mean(axis=1)
        self.item_means = self.user_item_matrix.mean(axis=0)
        
        # Calculate similarity matrix
        self._calculate_similarity_matrix()
        
        logger.info("Collaborative filtering model trained successfully!")
    
    def _calculate_similarity_matrix(self) -> None:
        """Calculate similarity matrix based on the method."""
        if self.method == 'user_based':
            matrix = self.user_item_matrix.values
        else:  # item_based
            matrix = self.user_item_matrix.T.values
        
        if self.similarity_metric == 'cosine':
            # Handle zero vectors by adding small epsilon
            matrix_normalized = matrix.copy()
            matrix_normalized[matrix_normalized == 0] = 1e-10
            self.similarity_matrix = cosine_similarity(matrix_normalized)
            
        elif self.similarity_metric == 'pearson':
            self.similarity_matrix = np.corrcoef(matrix)
            # Replace NaN with 0
            self.similarity_matrix = np.nan_to_num(self.similarity_matrix)
            
        elif self.similarity_metric == 'euclidean':
            # Convert to similarity (inverse of distance)
            distances = squareform(pdist(matrix, metric='euclidean'))
            self.similarity_matrix = 1 / (1 + distances)
        
        # Set diagonal to 0 to avoid self-similarity
        np.fill_diagonal(self.similarity_matrix, 0)
    
    def predict_rating(self, user_id: int, item_id: int, k: int = 50) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_id: User ID
            item_id: Item ID
            k: Number of neighbors to consider
            
        Returns:
            Predicted rating
        """
        if self.method == 'user_based':
            return self._predict_user_based(user_id, item_id, k)
        else:
            return self._predict_item_based(user_id, item_id, k)
    
    def _predict_user_based(self, user_id: int, item_id: int, k: int) -> float:
        """Predict rating using user-based collaborative filtering."""
        # Get similar users
        user_similarities = self.similarity_matrix[user_id]
        
        # Find users who rated the item
        item_ratings = self.user_item_matrix.iloc[:, item_id]
        rated_users = item_ratings[item_ratings > 0].index
        
        if len(rated_users) == 0:
            return self.item_means.iloc[item_id] if self.item_means.iloc[item_id] > 0 else self.global_mean
        
        # Get similarities for users who rated the item
        similarities = user_similarities[rated_users]
        ratings = item_ratings[rated_users]
        
        # Get top-k similar users
        if len(similarities) > k:
            top_k_indices = np.argsort(similarities)[-k:]
            similarities = similarities[top_k_indices]
            ratings = ratings.iloc[top_k_indices]
        
        # Calculate weighted average
        if np.sum(np.abs(similarities)) == 0:
            return self.user_means.iloc[user_id] if self.user_means.iloc[user_id] > 0 else self.global_mean
        
        # Mean-centered prediction
        user_mean = self.user_means.iloc[user_id]
        similar_user_means = self.user_means.iloc[ratings.index]
        
        numerator = np.sum(similarities * (ratings - similar_user_means))
        denominator = np.sum(np.abs(similarities))
        
        prediction = user_mean + (numerator / denominator)
        
        # Clip to valid rating range
        return np.clip(prediction, 1, 5)
    
    def _predict_item_based(self, user_id: int, item_id: int, k: int) -> float:
        """Predict rating using item-based collaborative filtering."""
        # Get similar items
        item_similarities = self.similarity_matrix[item_id]
        
        # Find items rated by the user
        user_ratings = self.user_item_matrix.iloc[user_id, :]
        rated_items = user_ratings[user_ratings > 0].index
        
        if len(rated_items) == 0:
            return self.item_means.iloc[item_id] if self.item_means.iloc[item_id] > 0 else self.global_mean
        
        # Get similarities for items rated by the user
        similarities = item_similarities[rated_items]
        ratings = user_ratings[rated_items]
        
        # Get top-k similar items
        if len(similarities) > k:
            top_k_indices = np.argsort(similarities)[-k:]
            similarities = similarities[top_k_indices]
            ratings = ratings.iloc[top_k_indices]
        
        # Calculate weighted average
        if np.sum(np.abs(similarities)) == 0:
            return self.item_means.iloc[item_id] if self.item_means.iloc[item_id] > 0 else self.global_mean
        
        numerator = np.sum(similarities * ratings)
        denominator = np.sum(np.abs(similarities))
        
        prediction = numerator / denominator
        
        # Clip to valid rating range
        return np.clip(prediction, 1, 5)
    
    def recommend_items(self, user_id: int, n_recommendations: int = 10, 
                       exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """
        Recommend items for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            exclude_rated: Whether to exclude already rated items
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        user_ratings = self.user_item_matrix.iloc[user_id, :]
        
        if exclude_rated:
            unrated_items = user_ratings[user_ratings == 0].index
        else:
            unrated_items = user_ratings.index
        
        # Predict ratings for unrated items
        predictions = {}
        for item_id in unrated_items:
            pred_rating = self.predict_rating(user_id, item_id)
            predictions[item_id] = pred_rating
        
        # Get top recommendations
        recommendations = get_top_k_items(predictions, n_recommendations)
        
        logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
        return recommendations
    
    def get_similar_users(self, user_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """Get most similar users."""
        if self.method != 'user_based':
            logger.warning("Similar users only available for user-based method")
            return []
        
        similarities = self.similarity_matrix[user_id]
        similar_users = [(i, sim) for i, sim in enumerate(similarities) if i != user_id]
        return sorted(similar_users, key=lambda x: x[1], reverse=True)[:k]
    
    def get_similar_items(self, item_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """Get most similar items."""
        if self.method != 'item_based':
            logger.warning("Similar items only available for item-based method")
            return []
        
        similarities = self.similarity_matrix[item_id]
        similar_items = [(i, sim) for i, sim in enumerate(similarities) if i != item_id]
        return sorted(similar_items, key=lambda x: x[1], reverse=True)[:k]
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'method': self.method,
            'similarity_metric': self.similarity_metric,
            'n_users': self.user_item_matrix.shape[0] if self.user_item_matrix is not None else 0,
            'n_items': self.user_item_matrix.shape[1] if self.user_item_matrix is not None else 0,
            'sparsity': 1 - (self.user_item_matrix > 0).sum().sum() / self.user_item_matrix.size if self.user_item_matrix is not None else 0,
            'global_mean': self.global_mean
        }

class MatrixFactorizationCF:
    """Matrix Factorization based Collaborative Filtering using SVD."""
    
    def __init__(self, n_factors: int = 50, learning_rate: float = 0.01, 
                 regularization: float = 0.1, n_epochs: int = 100):
        """
        Initialize Matrix Factorization model.
        
        Args:
            n_factors: Number of latent factors
            learning_rate: Learning rate for gradient descent
            regularization: Regularization parameter
            n_epochs: Number of training epochs
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None
        
    def fit(self, ratings_df: pd.DataFrame) -> None:
        """
        Fit the matrix factorization model.
        
        Args:
            ratings_df: DataFrame with columns ['user_id', 'item_id', 'rating']
        """
        logger.info("Training Matrix Factorization model...")
        
        # Get unique users and items
        users = ratings_df['user_id'].unique()
        items = ratings_df['item_id'].unique()
        
        n_users = len(users)
        n_items = len(items)
        
        # Initialize parameters
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.global_mean = ratings_df['rating'].mean()
        
        # Create user and item mappings
        self.user_mapping = {user: idx for idx, user in enumerate(users)}
        self.item_mapping = {item: idx for idx, item in enumerate(items)}
        
        # Training loop
        for epoch in range(self.n_epochs):
            for _, row in ratings_df.iterrows():
                user_idx = self.user_mapping[row['user_id']]
                item_idx = self.item_mapping[row['item_id']]
                rating = row['rating']
                
                # Prediction
                pred = self._predict_rating_idx(user_idx, item_idx)
                error = rating - pred
                
                # Update parameters
                user_factors_old = self.user_factors[user_idx].copy()
                
                self.user_factors[user_idx] += self.learning_rate * (
                    error * self.item_factors[item_idx] - 
                    self.regularization * self.user_factors[user_idx]
                )
                
                self.item_factors[item_idx] += self.learning_rate * (
                    error * user_factors_old - 
                    self.regularization * self.item_factors[item_idx]
                )
                
                self.user_biases[user_idx] += self.learning_rate * (
                    error - self.regularization * self.user_biases[user_idx]
                )
                
                self.item_biases[item_idx] += self.learning_rate * (
                    error - self.regularization * self.item_biases[item_idx]
                )
        
        logger.info("Matrix Factorization model trained successfully!")
    
    def _predict_rating_idx(self, user_idx: int, item_idx: int) -> float:
        """Predict rating using internal indices."""
        prediction = (
            self.global_mean + 
            self.user_biases[user_idx] + 
            self.item_biases[item_idx] + 
            np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        )
        return np.clip(prediction, 1, 5)
    
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """Predict rating for a user-item pair."""
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return self.global_mean
        
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[item_id]
        
        return self._predict_rating_idx(user_idx, item_idx)
    
    def recommend_items(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Recommend items for a user."""
        if user_id not in self.user_mapping:
            return []
        
        user_idx = self.user_mapping[user_id]
        predictions = {}
        
        for item_id, item_idx in self.item_mapping.items():
            pred_rating = self._predict_rating_idx(user_idx, item_idx)
            predictions[item_id] = pred_rating
        
        return get_top_k_items(predictions, n_recommendations)