"""
Command Line Interface for the recommendation system.
"""

import argparse
import sys
import pandas as pd
from typing import List, Dict, Any
from .data_preprocessing import DataPreprocessor
from .collaborative_filtering import CollaborativeFiltering
from .content_based_filtering import ContentBasedFiltering
from .hybrid_model import HybridRecommendationSystem
from .evaluation import RecommendationEvaluator
from .utils import setup_logging

logger = setup_logging()

class RecommendationCLI:
    """Command Line Interface for the recommendation system."""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.models = {}
        self.data = {}
        self.evaluator = RecommendationEvaluator()
    
    def load_data(self, dataset_size: str = "100k"):
        """Load and preprocess data."""
        print("Loading and preprocessing data...")
        self.data = self.preprocessor.preprocess_all(dataset_size)
        print("Data loaded successfully!")
        print(f"- Training ratings: {len(self.data['train_ratings'])}")
        print(f"- Test ratings: {len(self.data['test_ratings'])}")
        print(f"- Movies: {len(self.data['movies'])}")
        print(f"- Users: {self.data['user_item_matrix'].shape[0]}")
        print(f"- Items: {self.data['user_item_matrix'].shape[1]}")
    
    def train_models(self):
        """Train all recommendation models."""
        if not self.data:
            print("Please load data first using --load-data")
            return
        
        print("\n" + "="*50)
        print("TRAINING RECOMMENDATION MODELS")
        print("="*50)
        
        # Train Collaborative Filtering
        print("\n1. Training Collaborative Filtering...")
        self.models['collaborative'] = CollaborativeFiltering(method='user_based')
        self.models['collaborative'].fit(self.data['user_item_matrix'])
        print("✓ Collaborative Filtering trained!")
        
        # Train Content-Based Filtering
        print("\n2. Training Content-Based Filtering...")
        self.models['content_based'] = ContentBasedFiltering()
        self.models['content_based'].fit(self.data['movies'], self.data['train_ratings'])
        print("✓ Content-Based Filtering trained!")
        
        # Train Hybrid Model
        print("\n3. Training Hybrid Model...")
        self.models['hybrid'] = HybridRecommendationSystem(
            combination_method='weighted',
            cf_weight=0.6,
            cb_weight=0.4
        )
        self.models['hybrid'].fit(
            self.data['user_item_matrix'],
            self.data['train_ratings'],
            self.data['movies']
        )
        print("✓ Hybrid Model trained!")
        
        print("\n🎉 All models trained successfully!")
    
    def get_recommendations(self, user_id: int, method: str = 'hybrid', top_k: int = 10):
        """Get recommendations for a user."""
        if method not in self.models:
            print(f"Model '{method}' not found. Available models: {list(self.models.keys())}")
            return
        
        model = self.models[method]
        
        print(f"\n🎬 TOP {top_k} RECOMMENDATIONS FOR USER {user_id} ({method.upper()})")
        print("="*60)
        
        try:
            # Get recommendations
            recommendations = model.recommend_items(user_id, top_k)
            
            if not recommendations:
                print("No recommendations found for this user.")
                return
            
            # Display recommendations with movie info
            for i, rec in enumerate(recommendations, 1):
                if len(rec) == 3:
                    item_id, score, rec_method = rec
                else:
                    item_id, score = rec
                    rec_method = method
                
                # Get movie info
                movie_info = self.data['movies'][self.data['movies']['item_id'] == item_id]
                if not movie_info.empty:
                    title = movie_info.iloc[0]['title']
                    genres = movie_info.iloc[0].get('genres', 'Unknown')
                    year = movie_info.iloc[0].get('year', 'Unknown')
                    
                    print(f"{i:2d}. {title}")
                    print(f"    Score: {score:.3f} | Genres: {genres} | Year: {year}")
                    if len(rec) == 3:
                        print(f"    Method: {rec_method}")
                else:
                    print(f"{i:2d}. Movie ID: {item_id} (Score: {score:.3f})")
                print()
        
        except Exception as e:
            print(f"Error getting recommendations: {e}")
    
    def show_user_profile(self, user_id: int):
        """Show user's rating history and profile."""
        user_ratings = self.data['train_ratings'][self.data['train_ratings']['user_id'] == user_id]
        
        if user_ratings.empty:
            print(f"No rating history found for user {user_id}")
            return
        
        print(f"\n👤 USER {user_id} PROFILE")
        print("="*40)
        print(f"Total ratings: {len(user_ratings)}")
        print(f"Average rating: {user_ratings['rating'].mean():.2f}")
        print(f"Rating distribution:")
        print(user_ratings['rating'].value_counts().sort_index())
        
        print(f"\n🎬 RECENT MOVIES RATED:")
        print("-"*40)
        
        # Show top 10 highest rated movies
        top_rated = user_ratings.nlargest(10, 'rating')
        
        for _, rating in top_rated.iterrows():
            movie_info = self.data['movies'][self.data['movies']['item_id'] == rating['item_id']]
            if not movie_info.empty:
                title = movie_info.iloc[0]['title']
                genres = movie_info.iloc[0].get('genres', 'Unknown')
                print(f"⭐ {rating['rating']}/5 - {title} ({genres})")
    
    def evaluate_models(self):
        """Evaluate all trained models."""
        if not self.models:
            print("Please train models first using --train")
            return
        
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Evaluate each model
        results = self.evaluator.compare_models(
            self.models, 
            self.data['test_ratings'], 
            self.data['movies']
        )
        
        print("\n📊 EVALUATION RESULTS:")
        print("-"*50)
        print(results.round(4))
        
        # Save results
        results.to_csv('data/evaluation_results.csv')
        print("\n💾 Results saved to 'data/evaluation_results.csv'")
        
        # Plot comparison
        try:
            self.evaluator.plot_comparison(results, 'data/model_comparison.png')
            print("📈 Comparison plot saved to 'data/model_comparison.png'")
        except Exception as e:
            print(f"Could not save plot: {e}")
    
    def show_model_info(self):
        """Show information about trained models."""
        if not self.models:
            print("No models trained yet.")
            return
        
        print("\n🤖 MODEL INFORMATION")
        print("="*40)
        
        for model_name, model in self.models.items():
            print(f"\n{model_name.upper()}:")
            print("-" * 20)
            if hasattr(model, 'get_model_info'):
                info = model.get_model_info()
                for key, value in info.items():
                    print(f"  {key}: {value}")
            else:
                print("  Model info not available")
    
    def interactive_mode(self):
        """Start interactive mode."""
        print("\n🎬 INTERACTIVE RECOMMENDATION MODE")
        print("="*50)
        print("Commands:")
        print("  rec <user_id> [method] [top_k] - Get recommendations")
        print("  profile <user_id>              - Show user profile")
        print("  models                         - Show model info")
        print("  eval                          - Evaluate models")
        print("  quit                          - Exit")
        print("-"*50)
        
        while True:
            try:
                command = input("\n>>> ").strip().split()
                
                if not command:
                    continue
                
                if command[0] == 'quit':
                    break
                
                elif command[0] == 'rec':
                    if len(command) < 2:
                        print("Usage: rec <user_id> [method] [top_k]")
                        continue
                    
                    user_id = int(command[1])
                    method = command[2] if len(command) > 2 else 'hybrid'
                    top_k = int(command[3]) if len(command) > 3 else 10
                    
                    self.get_recommendations(user_id, method, top_k)
                
                elif command[0] == 'profile':
                    if len(command) < 2:
                        print("Usage: profile <user_id>")
                        continue
                    
                    user_id = int(command[1])
                    self.show_user_profile(user_id)
                
                elif command[0] == 'models':
                    self.show_model_info()
                
                elif command[0] == 'eval':
                    self.evaluate_models()
                
                else:
                    print(f"Unknown command: {command[0]}")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nGoodbye! 👋")

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Hybrid Recommendation System')
    
    parser.add_argument('--load-data', type=str, default='100k',
                       help='Load and preprocess data (100k, 1m, 10m)')
    parser.add_argument('--train', action='store_true',
                       help='Train all models')
    parser.add_argument('--user-id', type=int,
                       help='User ID for recommendations')
    parser.add_argument('--method', type=str, default='hybrid',
                       choices=['collaborative', 'content_based', 'hybrid'],
                       help='Recommendation method')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of recommendations')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate models')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive mode')
    
    args = parser.parse_args()
    
    cli = RecommendationCLI()
    
    # Load data
    if args.load_data:
        cli.load_data(args.load_data)
    
    # Train models
    if args.train:
        cli.train_models()
    
    # Get recommendations
    if args.user_id is not None:
        cli.get_recommendations(args.user_id, args.method, args.top_k)
    
    # Evaluate models
    if args.evaluate:
        cli.evaluate_models()
    
    # Interactive mode
    if args.interactive:
        cli.interactive_mode()
    
    # If no specific action, show help
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nQuick start:")
        print("  python -m src.interface --load-data 100k --train --interactive")

if __name__ == "__main__":
    main()