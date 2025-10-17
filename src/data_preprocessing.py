import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.utils import setup_logging, download_file, extract_zip, create_directories  # Changed from relative to absolute
import streamlit as st

logger = setup_logging()

class DataPreprocessor:
    """Handles data preprocessing for the recommendation system."""

    DEFAULT_THRESHOLD_MAP: Dict[str, Tuple[int, int]] = {
        "100k": (10, 10),
        "1m": (18, 22),
        "10m": (25, 30)
    }

    def __init__(
        self,
        data_path: str = "data/",
        min_user_interactions: int = 8,
        min_item_interactions: int = 8,
        iterative_pruning: bool = True,
    ):
        self.data_path = data_path
        self.raw_data_path = os.path.join(data_path, "raw")
        self.processed_data_path = os.path.join(data_path, "processed")

        # Create directories
        create_directories([self.raw_data_path, self.processed_data_path])

        # Initialize encoders
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.base_min_user_interactions = max(1, min_user_interactions)
        self.base_min_item_interactions = max(1, min_item_interactions)
        self.iterative_pruning = iterative_pruning
        
    def download_movielens_data(self, size: str = "100k") -> None:
        """Download MovieLens dataset."""
        size = size.lower()
        urls = {
            "100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
            "1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
            "10m": "https://files.grouplens.org/datasets/movielens/ml-10m.zip"
        }
        
        if size not in urls:
            raise ValueError(f"Size must be one of {list(urls.keys())}")
        
        zip_filename = os.path.join(self.raw_data_path, f"ml-{size}.zip")
        
        if not os.path.exists(zip_filename):
            logger.info(f"Downloading MovieLens {size} dataset...")
            download_file(urls[size], zip_filename)
            
        # Extract if not already extracted
        extract_path_map = {
            "100k": os.path.join(self.raw_data_path, "ml-100k"),
            "1m": os.path.join(self.raw_data_path, "ml-1m"),
            "10m": os.path.join(self.raw_data_path, "ml-10M100K"),
        }
        extract_path = extract_path_map.get(size, os.path.join(self.raw_data_path, f"ml-{size}"))
        if not os.path.exists(extract_path):
            logger.info("Extracting dataset...")
            extract_zip(zip_filename, self.raw_data_path)
    
    def load_movielens_data(self, size: str = "100k") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load MovieLens data."""
        # Download if not exists
        self.download_movielens_data(size)
        size = size.lower()
        
        if size == "100k":
            # Load ratings
            ratings_path = os.path.join(self.raw_data_path, "ml-100k", "u.data")
            ratings = pd.read_csv(
                ratings_path, 
                sep='\t', 
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                encoding='latin-1'
            )
            
            # Load movies
            movies_path = os.path.join(self.raw_data_path, "ml-100k", "u.item")
            movies = pd.read_csv(
                movies_path,
                sep='|',
                names=['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + 
                      [f'genre_{i}' for i in range(19)],
                encoding='latin-1'
            )
            
        elif size == "1m":
            # Load ratings
            ratings_path = os.path.join(self.raw_data_path, "ml-1m", "ratings.dat")
            ratings = pd.read_csv(
                ratings_path,
                sep='::',
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                engine='python',
                encoding='latin-1'
            )
            
            # Load movies
            movies_path = os.path.join(self.raw_data_path, "ml-1m", "movies.dat")
            movies = pd.read_csv(
                movies_path,
                sep='::',
                names=['item_id', 'title', 'genres'],
                engine='python',
                encoding='latin-1'
            )
        elif size == "10m":
            ratings_path = os.path.join(self.raw_data_path, "ml-10M100K", "ratings.dat")
            ratings = pd.read_csv(
                ratings_path,
                sep='::',
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                engine='python',
                encoding='latin-1'
            )

            movies_path = os.path.join(self.raw_data_path, "ml-10M100K", "movies.dat")
            movies = pd.read_csv(
                movies_path,
                sep='::',
                names=['item_id', 'title', 'genres'],
                engine='python',
                encoding='latin-1'
            )
        else:
            raise ValueError(f"Unsupported dataset size: {size}")

        logger.info(f"Loaded {len(ratings)} ratings and {len(movies)} movies")
        return ratings, movies
    
    def clean_ratings_data(
        self,
        ratings: pd.DataFrame,
        min_user_interactions: int,
        min_item_interactions: int,
    ) -> pd.DataFrame:
        """Clean and validate ratings data, aggressively pruning sparse users/items."""
        # Remove duplicates
        ratings = ratings.drop_duplicates()

        # Remove invalid ratings
        ratings = ratings[ratings['rating'].between(1, 5)]

        if len(ratings) == 0:
            return ratings

        min_user = max(2, min_user_interactions)
        min_item = max(2, min_item_interactions)

        previous_shape = None
        iteration = 0
        while previous_shape != ratings.shape:
            previous_shape = ratings.shape
            iteration += 1

            user_counts = ratings['user_id'].value_counts()
            valid_users = user_counts[user_counts >= min_user].index

            item_counts = ratings['item_id'].value_counts()
            valid_items = item_counts[item_counts >= min_item].index

            ratings = ratings[
                ratings['user_id'].isin(valid_users)
                & ratings['item_id'].isin(valid_items)
            ]

            if not self.iterative_pruning:
                break

        logger.info(
            "After cleaning (min_user=%s, min_item=%s, iterations=%s): %s ratings, %s users, %s items",
            min_user,
            min_item,
            iteration,
            len(ratings),
            ratings['user_id'].nunique(),
            ratings['item_id'].nunique(),
        )

        return ratings

    @staticmethod
    def calculate_density(ratings: pd.DataFrame) -> float:
        """Compute interaction density for the given ratings table."""
        n_users = ratings['user_id'].nunique()
        n_items = ratings['item_id'].nunique()
        total_cells = n_users * n_items
        if total_cells == 0:
            return 0.0
        return len(ratings) / total_cells

    def determine_thresholds(self, dataset_size: str) -> Tuple[int, int]:
        """Determine pruning thresholds for a given dataset size."""
        dataset_size = dataset_size.lower()
        return self.DEFAULT_THRESHOLD_MAP.get(
            dataset_size,
            (self.base_min_user_interactions, self.base_min_item_interactions),
        )
    
    def encode_ids(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Encode user and item IDs to consecutive integers."""
        ratings = ratings.copy()
        # Fit on all unique IDs in the full ratings set
        self.user_encoder.fit(ratings['user_id'])
        self.item_encoder.fit(ratings['item_id'])
        ratings['user_id'] = self.user_encoder.transform(ratings['user_id'])
        ratings['item_id'] = self.item_encoder.transform(ratings['item_id'])
        return ratings
    
    def create_user_item_matrix(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Create user-item interaction matrix."""
        return ratings.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        )
    
    def prepare_content_features(self, movies: pd.DataFrame, size: str = "100k") -> pd.DataFrame:
        """Prepare content-based features from movie data."""
        movies = movies.copy()
        
        if size == "100k":
            # For 100k dataset, genres are binary columns
            genre_columns = [col for col in movies.columns if col.startswith('genre_')]
            movies['genres'] = movies[genre_columns].apply(
                lambda row: '|'.join([f'genre_{i}' for i, val in enumerate(row) if val == 1]), 
                axis=1
            )
        
        # Extract year from title
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
        movies['year'] = pd.to_numeric(movies['year'], errors='coerce')
        
        # Clean title
        movies['clean_title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
        
        return movies
    
    def split_data(self, ratings: pd.DataFrame, 
                   test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        train_ratings, test_ratings = train_test_split(
            ratings, 
            test_size=test_size, 
            random_state=random_state,
            stratify=ratings['user_id']
        )
        return train_ratings, test_ratings
    
    def save_processed_data(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """Save processed data to files."""
        for name, df in data_dict.items():
            filepath = os.path.join(self.processed_data_path, f"{name}.csv")
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {name} data to {filepath}")
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """Load processed data from file."""
        filepath = os.path.join(self.processed_data_path, f"{filename}.csv")
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            raise FileNotFoundError(f"Processed data file not found: {filepath}")
    
    def preprocess_all(self, dataset_size: str = "100k") -> Dict[str, Any]:
        """Full preprocessing pipeline."""
        # Load data
        ratings, movies = self.load_movielens_data(dataset_size)

        min_user_interactions, min_item_interactions = self.determine_thresholds(dataset_size)
        ratings = self.clean_ratings_data(
            ratings,
            min_user_interactions=min_user_interactions,
            min_item_interactions=min_item_interactions,
        )

        if ratings.empty:
            raise ValueError(
                "No ratings remain after sparsity reduction. Consider lowering the minimum interaction thresholds."
            )
        
        # Get unique IDs from ratings BEFORE any encoding
        unique_user_ids = ratings['user_id'].unique()
        unique_item_ids = ratings['item_id'].unique()
        
        # Filter movies to only include items that exist in ratings
        movies = movies[movies['item_id'].isin(unique_item_ids)]
        
        # Fit encoders on the filtered data
        self.user_encoder.fit(unique_user_ids)
        self.item_encoder.fit(unique_item_ids)
        
        # Transform ratings
        ratings['user_id'] = self.user_encoder.transform(ratings['user_id'])
        ratings['item_id'] = self.item_encoder.transform(ratings['item_id'])
        
        # Transform movies (now safe because we filtered first)
        movies['item_id'] = self.item_encoder.transform(movies['item_id'])

        # Compute sparsity metrics prior to splitting
        density = self.calculate_density(ratings)
        sparsity = 1.0 - density

        # Split data
        train_ratings, test_ratings = self.split_data(ratings)
        
        # Create user-item matrix
        user_item_matrix = self.create_user_item_matrix(train_ratings)
        
        # Prepare content features
        movies = self.prepare_content_features(movies, dataset_size)
        
        metadata = {
            "dataset_size": dataset_size,
            "num_ratings": int(len(ratings)),
            "num_users": int(ratings['user_id'].nunique()),
            "num_items": int(ratings['item_id'].nunique()),
            "density": float(density),
            "sparsity": float(sparsity),
            "min_user_interactions": int(min_user_interactions),
            "min_item_interactions": int(min_item_interactions),
        }

        logger.info(
            "Post-cleaning density: %.4f (sparsity: %.4f)",
            density,
            sparsity,
        )

        return {
            "train_ratings": train_ratings,
            "test_ratings": test_ratings,
            "user_item_matrix": user_item_matrix,
            "movies": movies,
            "all_ratings": ratings,
            "metadata": metadata,
        }

@st.cache_data(show_spinner=False)
def load_data(dataset_size="100k"):
    """Load and cache data with progress indication."""
    try:
        st.write("Starting data preprocessing...")
        preprocessor = DataPreprocessor()
        st.write("DataPreprocessor created, loading data...")
        data = preprocessor.preprocess_all(dataset_size)
        st.write("Data loaded successfully!")
        return data, None
    except Exception as e:
        st.write(f"Error occurred: {str(e)}")
        return None, str(e)

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_all("100k")
    
    print("Data preprocessing completed!")
    print(f"Train ratings: {len(data['train_ratings'])}")
    print(f"Test ratings: {len(data['test_ratings'])}")
    print(f"User-item matrix shape: {data['user_item_matrix'].shape}")
    print(f"Movies: {len(data['movies'])}")