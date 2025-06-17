"""
Evaluation module for the recommendation system.
Implements various metrics to evaluate recommendation quality.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import setup_logging

logger = setup_logging()

class RecommendationEvaluator:
    """Evaluator for recommendation systems."""
    
    def __init__(self):
        self.results = {}
    
    def precision_at_k(self, recommended_items: List[int], 
                      relevant_items: List[int], k: int) -> float:
        """Calculate Precision@K."""
        if not recommended_items or k == 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_recommended = len(set(recommended_k) & set(relevant_items))
        
        return relevant_recommended / len(recommended_k)
    
    def recall_at_k(self, recommended_items: List[int], 
                   relevant_items: List[int], k: int) -> float:
        """Calculate Recall@K."""
        if not relevant_items:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_recommended = len(set(recommended_k) & set(relevant_items))
        
        return relevant_recommended / len(relevant_items)
    
    def f1_at_k(self, recommended_items: List[int], 
               relevant_items: List[int], k: int) -> float:
        """Calculate F1@K."""
        precision = self.precision_at_k(recommended_items, relevant_items, k)
        recall = self.recall_at_k(recommended_items, relevant_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def average_precision(self, recommended_items: List[int], 
                         relevant_items: List[int]) -> float:
        """Calculate Average Precision."""
        if not relevant_items:
            return 0.0
        
        score = 0.0
        num_hits = 0.0
        
        for i, item in enumerate(recommended_items):
            if item in relevant_items:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        return score / len(relevant_items)
    
    def ndcg_at_k(self, recommended_items: List[int], 
                  relevant_items: List[int], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K."""
        recommended_k = recommended_items[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommended_k):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)
        
        # Calculate IDCG (Ideal DCG)
        idcg = 0.0
        for i in range(min(len(relevant_items), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def coverage(self, all_recommendations: List[List[int]], 
                total_items: int) -> float:
        """Calculate catalog coverage."""
        unique_items = set()
        for recommendations in all_recommendations:
            unique_items.update(recommendations)
        
        return len(unique_items) / total_items
    
    def diversity(self, recommendations: List[int], 
                 similarity_matrix: np.ndarray) -> float:
        """Calculate intra-list diversity."""
        if len(recommendations) < 2:
            return 0.0
        
        total_similarity = 0.0
        count = 0
        
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                if (recommendations[i] < len(similarity_matrix) and 
                    recommendations[j] < len(similarity_matrix)):
                    total_similarity += similarity_matrix[recommendations[i]][recommendations[j]]
                    count += 1
        
        if count == 0:
            return 0.0
        
        # Return 1 - average similarity (higher diversity = lower similarity)
        return 1.0 - (total_similarity / count)
    
    def novelty(self, recommendations: List[int], 
               item_popularity: Dict[int, float]) -> float:
        """Calculate novelty of recommendations."""
        if not recommendations:
            return 0.0
        
        total_novelty = 0.0
        count = 0
        
        for item in recommendations:
            if item in item_popularity:
                # Novelty = -log2(popularity)
                novelty = -np.log2(item_popularity[item] + 1e-10)
                total_novelty += novelty
                count += 1
        
        return total_novelty / count if count > 0 else 0.0
    
    def evaluate_model(self, model, test_data: pd.DataFrame, 
                      movies_df: pd.DataFrame, k_values: List[int] = [5, 10, 20]) -> Dict:
        """Evaluate a recommendation model comprehensively."""
        logger.info("Starting model evaluation...")
        
        results = {
            'precision': {k: [] for k in k_values},
            'recall': {k: [] for k in k_values},
            'f1': {k: [] for k in k_values},
            'ndcg': {k: [] for k in k_values},
            'rmse': [],
            'mae': [],
            'coverage': 0.0,
            'diversity': [],
            'novelty': []
        }
        
        # Calculate item popularity for novelty
        item_popularity = test_data.groupby('item_id').size() / len(test_data)
        item_popularity = item_popularity.to_dict()
        
        # Group test data by user
        user_groups = test_data.groupby('user_id')
        all_recommendations = []
        
        for user_id, user_data in user_groups:
            # Get user's test items (relevant items)
            relevant_items = user_data['item_id'].tolist()
            actual_ratings = user_data['rating'].tolist()
            
            # Get recommendations
            try:
                if hasattr(model, 'recommend_items'):
                    recommendations = model.recommend_items(user_id, max(k_values), exclude_rated=True)
                    recommended_items = [item_id for item_id, _, _ in recommendations] if len(recommendations) > 0 and len(recommendations[0]) == 3 else [item_id for item_id, _ in recommendations]
                    predicted_ratings = [score for _, score, _ in recommendations] if len(recommendations) > 0 and len(recommendations[0]) == 3 else [score for _, score in recommendations]
                else:
                    continue
            except Exception as e:
                logger.warning(f"Error getting recommendations for user {user_id}: {e}")
                continue
            
            if not recommended_items:
                continue
            
            all_recommendations.append(recommended_items)
            
            # Calculate metrics for different k values
            for k in k_values:
                results['precision'][k].append(
                    self.precision_at_k(recommended_items, relevant_items, k)
                )
                results['recall'][k].append(
                    self.recall_at_k(recommended_items, relevant_items, k)
                )
                results['f1'][k].append(
                    self.f1_at_k(recommended_items, relevant_items, k)
                )
                results['ndcg'][k].append(
                    self.ndcg_at_k(recommended_items, relevant_items, k)
                )
            
            # Calculate rating prediction errors
            for item_id, actual_rating in zip(relevant_items, actual_ratings):
                try:
                    predicted_rating = model.predict_rating(user_id, item_id)
                    results['rmse'].append((actual_rating - predicted_rating) ** 2)
                    results['mae'].append(abs(actual_rating - predicted_rating))
                except:
                    continue
            
            # Calculate diversity (if similarity matrix available)
            if hasattr(model, 'cb_model') and hasattr(model.cb_model, 'similarity_matrix'):
                diversity_score = self.diversity(recommended_items[:10], model.cb_model.similarity_matrix)
                results['diversity'].append(diversity_score)
            
            # Calculate novelty
            novelty_score = self.novelty(recommended_items[:10], item_popularity)
            results['novelty'].append(novelty_score)
        
        # Calculate final metrics
        final_results = {}
        
        for k in k_values:
            final_results[f'precision@{k}'] = np.mean(results['precision'][k]) if results['precision'][k] else 0.0
            final_results[f'recall@{k}'] = np.mean(results['recall'][k]) if results['recall'][k] else 0.0
            final_results[f'f1@{k}'] = np.mean(results['f1'][k]) if results['f1'][k] else 0.0
            final_results[f'ndcg@{k}'] = np.mean(results['ndcg'][k]) if results['ndcg'][k] else 0.0
        
        final_results['rmse'] = np.sqrt(np.mean(results['rmse'])) if results['rmse'] else 0.0
        final_results['mae'] = np.mean(results['mae']) if results['mae'] else 0.0
        final_results['coverage'] = self.coverage(all_recommendations, len(movies_df))
        final_results['diversity'] = np.mean(results['diversity']) if results['diversity'] else 0.0
        final_results['novelty'] = np.mean(results['novelty']) if results['novelty'] else 0.0
        
        logger.info("Model evaluation completed!")
        return final_results
    
    def compare_models(self, models: Dict, test_data: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Compare multiple models."""
        comparison_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
            results = self.evaluate_model(model, test_data, movies_df)
            comparison_results[model_name] = results
        
        # Convert to DataFrame for easy comparison
        comparison_df = pd.DataFrame(comparison_results).T
        return comparison_df
    
    def plot_comparison(self, comparison_df: pd.DataFrame, save_path: str = None):
        """Plot model comparison results."""
        metrics_to_plot = ['precision@10', 'recall@10', 'f1@10', 'rmse', 'coverage', 'diversity']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in comparison_df.columns:
                ax = axes[i]
                comparison_df[metric].plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title(f'{metric.upper()}')
                ax.set_ylabel('Score')
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_metrics_over_k(self, model, test_data: pd.DataFrame, movies_df: pd.DataFrame, 
                           k_values: List[int] = None, save_path: str = None):
        """Plot metrics over different k values."""
        if k_values is None:
            k_values = [1, 3, 5, 10, 15, 20, 25, 30]
        
        metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'ndcg': []
        }
        
        for k in k_values:
            results = self.evaluate_model(model, test_data, movies_df, [k])
            metrics['precision'].append(results[f'precision@{k}'])
            metrics['recall'].append(results[f'recall@{k}'])
            metrics['f1'].append(results[f'f1@{k}'])
            metrics['ndcg'].append(results[f'ndcg@{k}'])
        
        plt.figure(figsize=(12, 8))
        
        for metric_name, values in metrics.items():
            plt.plot(k_values, values, marker='o', label=metric_name.upper())
        
        plt.xlabel('K (Number of Recommendations)')
        plt.ylabel('Score')
        plt.title('Recommendation Metrics vs K')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def statistical_significance_test(self, results1: List[float], results2: List[float]) -> Dict:
        """Perform statistical significance test between two sets of results."""
        from scipy import stats
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(results1, results2)
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(results1, results2)
        
        return {
            't_test': {'statistic': t_stat, 'p_value': p_value},
            'wilcoxon': {'statistic': wilcoxon_stat, 'p_value': wilcoxon_p},
            'significant': p_value < 0.05
        }