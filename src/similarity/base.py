"""Base class for similarity search algorithms."""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np


class BaseSimilarityFinder(ABC):
    """Abstract base class for similarity search implementations."""
    
    def __init__(self, settings: Dict[str, Any]) -> None:
        """
        Initialize similarity finder.
        
        Args:
            settings: Similarity settings from configuration
        """
        self.settings: Dict[str, Any] = settings
        self.num_similar: int = settings.get("num_similar", 4)
    
    def find_similar_with_processed_data(self, query_data: Union[np.ndarray, List[float]], 
                                       labeled_data: List[Tuple[Any, Union[np.ndarray, List[float]]]],
                                       top_k: Optional[int] = None) -> List[Tuple[Any, np.ndarray, float]]:
        """
        Find most similar labeled samples using pre-processed data.
        
        Args:
            query_data: Pre-processed query data (numpy array or list)
            labeled_data: List of (index/identifier, processed_data) tuples
            top_k: Number of similar samples to return (uses self.num_similar if None)
        
        Returns:
            List of (index/identifier, processed_data, distance) tuples, sorted by distance (closest first)
        """
        if top_k is None:
            top_k = self.num_similar
        
        if len(labeled_data) == 0:
            print("No labeled samples available for comparison.")
            return []
        
        # Convert to numpy arrays
        q_data = np.array(query_data, dtype=np.float64)
        
        # Compute distances
        results = []
        for identifier, series_data in labeled_data:
            series_array = np.array(series_data, dtype=np.float64)
            
            # Check length compatibility
            if len(series_array) != len(q_data):
                continue
            
            # Compute distance using the specific algorithm
            dist = self._compute_distance(q_data, series_array)
            results.append((identifier, series_data, dist))
        
        # Sort by distance and return top_k
        results.sort(key=lambda t: t[2])
        return results[:top_k]
    
    def visualize_similar_with_processed_data(self, query_index: Any, query_data: Union[np.ndarray, List[float]],
                                            similar_results: List[Tuple[Any, np.ndarray, float]],
                                            color_map: Optional[Dict[int, str]] = None,
                                            dataset: Optional[Any] = None,
                                            labeling_type: Optional[str] = None,
                                            num_prices: Optional[int] = None) -> None:
        """
        Visualize the query sample and similar results using pre-processed data.
        
        Args:
            query_index: Query sample identifier (index or other identifier)
            query_data: Pre-processed query data
            similar_results: List of (index, processed_data, distance) tuples from find_similar_with_processed_data
            color_map: Optional color mapping for classes (for classification)
            dataset: Optional dataset object to access labels
            labeling_type: Optional labeling type ('predict' or 'classify')
            num_prices: Optional number of prices for predict mode
        """
        # Default implementation - should be overridden by subclasses
        print("visualize_similar_with_processed_data not implemented in base class")
    
    @abstractmethod
    def _compute_distance(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Compute distance between two processed data arrays.
        
        Args:
            data1: First processed data array
            data2: Second processed data array
        
        Returns:
            Distance value
        """
        pass


