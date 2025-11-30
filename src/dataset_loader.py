"""
Dataset loader for NumPy format datasets.

Supports loading datasets from individual files:
- prices.npy - required, shape (N × n_prices)
- timestamps.npy - required, shape (N,)
- ids.npy - optional, shape (N,)
- cluster_ids.npy - optional, shape (N,)
- labels.npy - optional, loaded if exists (saved during labeling)
"""
import os
import json
import numpy as np
from typing import Dict, Optional
from pathlib import Path


class NumpyDataset:
    """Dataset stored as NumPy arrays."""
    
    def __init__(self, prices_file: str, timestamps_file: str,
                 prices: np.ndarray, timestamps: np.ndarray,
                 ids: Optional[np.ndarray] = None,
                 cluster_ids: Optional[np.ndarray] = None, 
                 metadata: Optional[Dict] = None,
                 labels: Optional[np.ndarray] = None,
                 labels_file: Optional[str] = None,
                 predicted_prices: Optional[np.ndarray] = None):
        """
        Initialize NumPy dataset.
        
        Args:
            prices_file: Path to prices.npy file
            timestamps_file: Path to timestamps.npy file
            prices: Price data array, shape (N × n_prices)
            timestamps: Timestamp array, shape (N,)
            ids: Optional IDs array, shape (N,)
            cluster_ids: Optional cluster IDs, shape (N,)
            metadata: Optional metadata dictionary
            labels: Optional labels array (loaded if exists)
            labels_file: Path to labels.npy file (for saving)
        """
        self.prices_file = prices_file
        self.timestamps_file = timestamps_file
        self.labels_file = labels_file
        self.prices = prices
        self.timestamps = timestamps
        self.ids = ids
        self.cluster_ids = cluster_ids
        self.metadata = metadata or {}
        self.labels = labels
        self.predicted_prices = predicted_prices
        
        # Determine folder and prefix from prices file path
        prices_path = Path(prices_file)
        self.folder = str(prices_path.parent)
        # Use folder name as prefix for display
        self.prefix = prices_path.parent.name or "dataset"
        
        # Validate dimensions
        n_samples = len(prices)
        if len(timestamps) != n_samples:
            raise ValueError(f"Prices and timestamps must have same length: {len(prices)} vs {len(timestamps)}")
        
        if ids is not None and len(ids) != n_samples:
            raise ValueError(f"IDs length mismatch: {len(ids)} vs {n_samples}")
        
        if cluster_ids is not None and len(cluster_ids) != n_samples:
            raise ValueError(f"Cluster IDs length mismatch: {len(cluster_ids)} vs {n_samples}")
        
        if labels is not None:
            if len(labels) != n_samples:
                raise ValueError(f"Labels length mismatch: {len(labels)} vs {n_samples}")
        
        if predicted_prices is not None:
            if len(predicted_prices) != n_samples:
                raise ValueError(f"Predicted prices length mismatch: {len(predicted_prices)} vs {n_samples}")
    
    def __len__(self) -> int:
        """Number of samples in dataset."""
        return len(self.prices)
    
    def get_sample_name(self, index: int) -> str:
        """
        Get display name for a sample.
        
        Includes prefix and optional cluster_id/id if available.
        """
        parts = []
        
        if self.cluster_ids is not None:
            cluster_id = self.cluster_ids[index]
            parts.append(f"cluster_{cluster_id}")
        
        if self.ids is not None:
            sample_id = self.ids[index]
            parts.append(f"id_{sample_id}")
        
        parts.append(f"idx_{index}")
        return "_".join(parts) if parts else f"sample_{index}"
    
    def has_labels(self, index: int) -> bool:
        """Check if sample at index has labels."""
        if self.labels is None:
            return False
        
        if self.labels.ndim == 1:
            # Classify mode
            return not (np.isnan(self.labels[index]) if np.issubdtype(self.labels.dtype, np.floating) 
                      else self.labels[index] == -1)
        else:
            # Predict mode
            return not np.all(np.isnan(self.labels[index]))
    
    def get_description(self) -> str:
        """Get dataset description from metadata."""
        return self.metadata.get("description", "")


def load_numpy_dataset(prices_file: str, timestamps_file: str,
                      ids_file: Optional[str] = None,
                      cluster_ids_file: Optional[str] = None,
                      labels_file: Optional[str] = None,
                      metadata_file: Optional[str] = None,
                      predicted_prices_file: Optional[str] = None) -> NumpyDataset:
    """
    Load a NumPy dataset from individual files.
    
    Args:
        prices_file: Path to prices.npy file (required)
        timestamps_file: Path to timestamps.npy file (required)
        ids_file: Path to ids.npy file (optional)
        cluster_ids_file: Path to cluster_ids.npy file (optional)
        labels_file: Path to labels.npy file (optional)
        
    Returns:
        NumpyDataset instance
        
    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If data shapes are incompatible
    """
    # Check required files
    if not os.path.exists(prices_file):
        raise FileNotFoundError(f"Required file not found: {prices_file}")
    if not os.path.exists(timestamps_file):
        raise FileNotFoundError(f"Required file not found: {timestamps_file}")
    
    # Load required data
    prices = np.load(prices_file, allow_pickle=True)
    timestamps = np.load(timestamps_file, allow_pickle=True)
    
    # Load optional files
    ids = None
    if ids_file and os.path.exists(ids_file):
        ids = np.load(ids_file, allow_pickle=True)
    
    cluster_ids = None
    if cluster_ids_file and os.path.exists(cluster_ids_file):
        cluster_ids = np.load(cluster_ids_file, allow_pickle=True)
    
    labels = None
    resolved_labels_file = None

    labels_candidates = []
    if labels_file:
        labels_candidates.append(Path(labels_file))

    prices_path = Path(prices_file)
    prices_dir = prices_path.parent
    prefix = prices_dir.name or "dataset"

    labels_candidates.append(prices_dir / f"{prefix}_labels.npy")
    labels_candidates.append(prices_dir / "labels.npy")

    seen_paths = set()
    for candidate in labels_candidates:
        if candidate is None:
            continue
        candidate_path = candidate
        if not candidate_path.is_absolute():
            candidate_path = prices_dir / candidate_path
        candidate_path = candidate_path.resolve()
        if candidate_path in seen_paths:
            continue
        seen_paths.add(candidate_path)
        if candidate_path.exists():
            labels = np.load(candidate_path, allow_pickle=True)
            resolved_labels_file = str(candidate_path)
            break

    if resolved_labels_file is not None:
        labels_file = resolved_labels_file
    
    predicted_prices = None
    if predicted_prices_file and os.path.exists(predicted_prices_file):
        predicted_prices = np.load(predicted_prices_file, allow_pickle=True)
    
    # Load metadata from file or check default location
    metadata = None
    if metadata_file and os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        # Try to load from default location (same folder as prices.npy)
        prices_path = Path(prices_file)
        default_metadata_file = prices_path.parent / "metadata.json"
        if default_metadata_file.exists():
            with open(default_metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
    
    return NumpyDataset(
        prices_file=prices_file,
        timestamps_file=timestamps_file,
        prices=prices,
        timestamps=timestamps,
        ids=ids,
        cluster_ids=cluster_ids,
        metadata=metadata,
        labels=labels,
        labels_file=labels_file,
        predicted_prices=predicted_prices
    )

