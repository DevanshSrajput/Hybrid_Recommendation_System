
__version__ = "1.0.0"
__author__ = "Devansh Rajput"
__email__ = "dksdevansh@gmail.com"

from .collaborative_filtering import CollaborativeFiltering
from .content_based_filtering import ContentBasedFiltering
from .hybrid_model import HybridRecommendationSystem
from .evaluation import RecommendationEvaluator

__all__ = [
    'CollaborativeFiltering',
    'ContentBasedFiltering', 
    'HybridRecommendationSystem',
    'RecommendationEvaluator'
]