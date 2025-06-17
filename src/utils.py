"""
Utility functions for the recommendation system.
"""

import os
import logging
import zipfile
import requests
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from tqdm import tqdm

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('recommendation_system.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def download_file(url: str, filename: str, chunk_size: int = 8192) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))

def extract_zip(zip_path: str, extract_to: str) -> None:
    """Extract a zip file."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def create_directories(paths: List[str]) -> None:
    """Create directories if they don't exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)

def normalize_ratings(ratings: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """Normalize ratings using specified method."""
    if method == 'minmax':
        min_val, max_val = ratings.min(), ratings.max()
        return (ratings - min_val) / (max_val - min_val)
    elif method == 'zscore':
        return (ratings - ratings.mean()) / ratings.std()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def calculate_sparsity(matrix: np.ndarray) -> float:
    """Calculate sparsity of a matrix."""
    return 1.0 - (np.count_nonzero(matrix) / matrix.size)

def get_top_k_items(scores: Dict[int, float], k: int = 10) -> List[Tuple[int, float]]:
    """Get top K items from a score dictionary."""
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default

class Timer:
    """Context manager for timing code execution."""
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        self.duration = self.end - self.start

def validate_user_item_data(df: pd.DataFrame, 
                           user_col: str = 'user_id',
                           item_col: str = 'item_id',
                           rating_col: str = 'rating') -> bool:
    """Validate user-item interaction data."""
    required_cols = [user_col, item_col, rating_col]
    
    if not all(col in df.columns for col in required_cols):
        return False
    
    if df[required_cols].isnull().any().any():
        return False
    
    if df[rating_col].dtype not in ['int64', 'float64']:
        return False
    
    return True

import time