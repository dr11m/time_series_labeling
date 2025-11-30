"""Soft-DTW similarity search implementation."""
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List, Tuple, Dict, Any, Optional, Union
from src.similarity.base import BaseSimilarityFinder


def soft_dtw_numpy(x: np.ndarray, y: np.ndarray, gamma: float = 0.05) -> float:
    """
    Soft-DTW (numpy), works for short series (L ~ 16-100) fast and stable.
    
    Args:
        x: 1D numpy array
        y: 1D numpy array (same length as x)
        gamma: softness parameter > 0
    
    Returns:
        Soft-DTW distance
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Lx = x.shape[0]
    Ly = y.shape[0]
    if Lx != Ly:
        raise ValueError("soft_dtw_numpy expects equal lengths")
    L = Lx

    # cost matrix: squared euclidean
    D = (x.reshape(-1, 1) - y.reshape(1, -1)) ** 2  # (L,L)

    INF = 1e12
    R = np.full((L + 1, L + 1), INF, dtype=np.float64)
    R[0, 0] = 0.0

    # dynamic programming with soft-min
    for i in range(1, L + 1):
        for j in range(1, L + 1):
            a = R[i - 1, j]  # top
            b = R[i, j - 1]  # left
            c = R[i - 1, j - 1]  # diag
            arr = np.array([-a / gamma, -b / gamma, -c / gamma], dtype=np.float64)
            m = arr.max()
            lse = m + np.log(np.exp(arr - m).sum())
            softmin = -gamma * lse
            R[i, j] = D[i - 1, j - 1] + softmin

    return float(R[L, L])


class SoftDTWSimilarityFinder(BaseSimilarityFinder):
    """Similarity search using Soft-DTW algorithm with normalization."""
    
    def __init__(self, settings: Dict[str, Any]) -> None:
        """
        Initialize Soft-DTW similarity finder.
        
        Args:
            settings: Similarity settings including gamma, num_similar
        """
        super().__init__(settings)
        self.gamma: float = settings.get("gamma", 0.05)
        self.use_prefilter: bool = settings.get("use_prefilter", True)
        self.prefilter_m: int = settings.get("prefilter_m", 200)
    
    def _normalize_0_1(self, values: np.ndarray) -> np.ndarray:
        """
        Min-max normalize to 0-1 range.
        
        Args:
            values: Array of values
        
        Returns:
            Values normalized to 0-1 range
        """
        vmin, vmax = values.min(), values.max()
        if vmax - vmin < 1e-10:
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)
    
    def _compute_distance(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Compute Soft-DTW distance between two processed data arrays.
        Always applies min-max 0-1 normalization for proper comparison.
        
        Args:
            data1: First processed data array
            data2: Second processed data array
        
        Returns:
            Soft-DTW distance
        """
        # Always apply min-max 0-1 normalization
        data1 = self._normalize_0_1(data1)
        data2 = self._normalize_0_1(data2)
        return soft_dtw_numpy(data1, data2, gamma=self.gamma)
    
    def visualize_similar_with_processed_data(self, query_index: Any, query_data: Union[np.ndarray, List[float]],
                                            similar_results: List[Tuple[Any, np.ndarray, float]],
                                            color_map: Optional[Dict[int, str]] = None,
                                            dataset: Optional[Any] = None,
                                            labeling_type: Optional[str] = None,
                                            num_prices: Optional[int] = None) -> None:
        """
        Visualize query and similar samples using pre-processed data.
        
        Args:
            query_index: Query sample identifier (index or other identifier)
            query_data: Pre-processed query data
            similar_results: List of (index, processed_data, distance) tuples
            color_map: Optional color mapping for classification labels
            dataset: Optional dataset object to access labels
            labeling_type: Optional labeling type ('predict', 'classify', or 'anomaly_detection')
            num_prices: Optional number of prices for predict mode
        """
        n_similar = len(similar_results)
        
        if n_similar == 0:
            print("No similar samples to display.")
            return
        
        # Convert query_data to numpy array if needed
        query_data = np.asarray(query_data, dtype=np.float64)
        
        # Calculate grid layout
        # Row 1: 1 plot (query, centered)
        # Rows 2+: 3 plots per row for similar series
        n_rows = 1 + math.ceil(n_similar / 3)
        
        # Create figure
        fig = plt.figure(figsize=(15, 4 * n_rows))
        
        # Plot query series in first row (centered)
        ax_query = plt.subplot(n_rows, 3, 2)  # Middle column of first row
        x_q = np.arange(len(query_data))
        ax_query.plot(x_q, query_data, marker='o', color='blue', alpha=0.9, linewidth=2)
        
        # Add labels for query if available
        query_title = f"Query Sample\nIndex: {query_index}"
        if dataset and dataset.labels is not None and labeling_type == 'predict' and num_prices:
            query_labels = dataset.labels[query_index]
            if not np.all(np.isnan(query_labels[:num_prices])):
                labeled_prices = []
                for j in range(num_prices):
                    if not np.isnan(query_labels[j]):
                        labeled_prices.append(f"{query_labels[j]:.3f}")
                        # Draw horizontal line for labeled price
                        ax_query.axhline(y=query_labels[j], color='orange', linestyle='--',
                                        alpha=0.7, linewidth=2)
                if labeled_prices:
                    query_title += f"\nLabels: {', '.join(labeled_prices)}"
        elif dataset and dataset.labels is not None and labeling_type == 'classify':
            query_label = dataset.labels[query_index]
            if not np.isnan(query_label):
                class_num = int(query_label)
                class_name = color_map.get(class_num, f"Class {class_num}") if color_map else f"Class {class_num}"
                query_title += f"\nLabel: {class_name}"
        elif dataset and dataset.labels is not None and labeling_type == 'anomaly_detection':
            query_labels = dataset.labels[query_index]
            if query_labels.ndim == 1 and len(query_labels) == len(query_data):
                # Mark anomalies as red points
                anomaly_mask = query_labels == 1
                if np.any(anomaly_mask):
                    x_anomaly = x_q[anomaly_mask]
                    y_anomaly = query_data[anomaly_mask]
                    ax_query.scatter(x_anomaly, y_anomaly, color='red', s=200, 
                                   zorder=6, edgecolors='darkred', linewidths=2, 
                                   alpha=0.9, marker='X', label='Anomaly')
                    anomaly_count = int(np.sum(anomaly_mask))
                    query_title += f"\nAnomalies: {anomaly_count}"
        
        ax_query.set_title(query_title, fontweight='bold', fontsize=12)
        ax_query.grid(alpha=0.3)
        
        # Plot similar series in subsequent rows (3 per row)
        for i, (index, processed_data, dist) in enumerate(similar_results):
            row = 1 + (i // 3)
            col = i % 3
            ax_idx = row * 3 + col + 1
            
            ax = plt.subplot(n_rows, 3, ax_idx)
            
            x = np.arange(len(processed_data))
            
            # Use green color by default (can be customized with color_map if needed)
            color = 'green'
            
            ax.plot(x, processed_data, marker='o', color=color, alpha=0.9, linewidth=2)
            
            # Add labels to title and plot
            similar_title = f"Similar #{i+1} (dist={dist:.4f})\nIndex: {index}"
            
            if dataset and dataset.labels is not None and labeling_type == 'predict' and num_prices:
                sample_labels = dataset.labels[index]
                if not np.all(np.isnan(sample_labels[:num_prices])):
                    labeled_prices = []
                    for j in range(num_prices):
                        if not np.isnan(sample_labels[j]):
                            labeled_prices.append(f"{sample_labels[j]:.3f}")
                            # Draw horizontal line for labeled price
                            ax.axhline(y=sample_labels[j], color='orange', linestyle='--',
                                     alpha=0.7, linewidth=2)
                    if labeled_prices:
                        similar_title += f"\nLabels: {', '.join(labeled_prices)}"
            elif dataset and dataset.labels is not None and labeling_type == 'classify':
                sample_label = dataset.labels[index]
                if not np.isnan(sample_label):
                    class_num = int(sample_label)
                    class_name = color_map.get(class_num, f"Class {class_num}") if color_map else f"Class {class_num}"
                    similar_title += f"\nLabel: {class_name}"
            elif dataset and dataset.labels is not None and labeling_type == 'anomaly_detection':
                sample_labels = dataset.labels[index]
                if sample_labels.ndim == 1 and len(sample_labels) == len(processed_data):
                    # Mark anomalies as red points
                    anomaly_mask = sample_labels == 1
                    if np.any(anomaly_mask):
                        x_anomaly = x[anomaly_mask]
                        y_anomaly = np.asarray(processed_data)[anomaly_mask]
                        ax.scatter(x_anomaly, y_anomaly, color='red', s=200, 
                                 zorder=6, edgecolors='darkred', linewidths=2, 
                                 alpha=0.9, marker='X', label='Anomaly')
                        anomaly_count = int(np.sum(anomaly_mask))
                        similar_title += f"\nAnomalies: {anomaly_count}"
            
            ax.set_title(similar_title, fontsize=10)
            ax.grid(alpha=0.3)
        
        fig.suptitle(f"Top {n_similar} Similar Samples (Soft-DTW, γ={self.gamma})", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()


