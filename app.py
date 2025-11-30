"""
Time Series Labeling Application

A universal tool for labeling time series with support for:
- Prediction: Label future price values
- Classification: Categorize time series into classes

Uses pluggable similarity search methods.
"""
import sys
import os
import warnings
import datetime
from typing import Dict, Any

# Show all warnings and errors
warnings.filterwarnings("default")

# Import settings and UI
from src.settings.settings_window import open_settings_window
from src.settings.settings_manager import SettingsManager

# Import dataset loader
from src.dataset_loader import NumpyDataset, load_numpy_dataset

# Import labeling types
from src.labeling_types.predict import PredictLabeling
from src.labeling_types.classify import ClassifyLabeling
from src.labeling_types.anomaly_detection import AnomalyDetectionLabeling
from src.labeling_types.base import BaseLabelingType

# Import similarity
from src.similarity.soft_dtw import SoftDTWSimilarityFinder
from src.similarity.base import BaseSimilarityFinder


def create_similarity_finder(settings: Dict[str, Any]) -> BaseSimilarityFinder:
    """
    Create similarity finder based on settings.
    
    Args:
        settings: Application settings dictionary
    
    Returns:
        Similarity finder instance
    """
    similarity_settings = settings.get("similarity", {})
    method = similarity_settings.get("method", "soft_dtw")
    
    if method == "soft_dtw":
        return SoftDTWSimilarityFinder(similarity_settings)
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def load_dataset(settings: Dict[str, Any]) -> NumpyDataset:
    """
    Load dataset from file paths specified in settings.
    
    Args:
        settings: Application settings dictionary
    
    Returns:
        NumpyDataset instance
    """
    data_settings = settings.get("data", {})
    prices_file = data_settings.get("prices_file", "")
    timestamps_file = data_settings.get("timestamps_file", "")
    ids_file = data_settings.get("ids_file", "")
    cluster_ids_file = data_settings.get("cluster_ids_file", "")
    labels_file = data_settings.get("labels_file", "")
    metadata_file = data_settings.get("metadata_file", "")
    predicted_prices_file = data_settings.get("predicted_prices_to_help_file", "")
    
    if not prices_file or not timestamps_file:
        print(f"Error: Required files not specified in settings")
        sys.exit(1)
    
    if not os.path.exists(prices_file):
        print(f"Error: Prices file not found: {prices_file}")
        sys.exit(1)
    
    if not os.path.exists(timestamps_file):
        print(f"Error: Timestamps file not found: {timestamps_file}")
        sys.exit(1)
    
    try:
        dataset = load_numpy_dataset(
            prices_file=prices_file,
            timestamps_file=timestamps_file,
            ids_file=ids_file if ids_file else None,
            cluster_ids_file=cluster_ids_file if cluster_ids_file else None,
            labels_file=labels_file if labels_file else None,
            metadata_file=metadata_file if metadata_file else None,
            predicted_prices_file=predicted_prices_file if predicted_prices_file else None
        )
        print(f"Loaded dataset from files")
        print(f"  - Samples: {len(dataset)}")
        print(f"  - Price shape: {dataset.prices.shape}")
        
        if dataset.labels is not None:
            import numpy as np
            if dataset.labels.ndim == 1:
                labeled_count = np.sum(~np.isnan(dataset.labels))
            else:
                labeled_count = np.sum(~np.all(np.isnan(dataset.labels), axis=1))
            print(f"  - Labeled: {labeled_count}/{len(dataset)}")
        
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_labeling_app(settings: Dict[str, Any], dataset: NumpyDataset, 
                        similarity_finder: BaseSimilarityFinder) -> BaseLabelingType:
    """
    Create appropriate labeling application based on settings.
    
    Args:
        settings: Application settings dictionary
        dataset: TimeSeriesDataset instance
        similarity_finder: Similarity finder instance
    
    Returns:
        Labeling application instance
    """
    labeling_type = settings.get("labeling_type", "predict")
    
    if labeling_type == "predict":
        print("Starting in PREDICT mode")
        print(f"  - Labeling {settings['predict']['num_prices']} price(s)")
        return PredictLabeling(dataset, settings, similarity_finder)
    
    elif labeling_type == "classify":
        print("Starting in CLASSIFY mode")
        print(f"  - Classifying into {settings['classify']['num_classes']} classes")
        return ClassifyLabeling(dataset, settings, similarity_finder)
    
    elif labeling_type == "anomaly_detection":
        print("Starting in ANOMALY DETECTION mode")
        print("  - Marking anomaly points in time series")
        return AnomalyDetectionLabeling(dataset, settings, similarity_finder)
    
    else:
        raise ValueError(f"Unknown labeling type: {labeling_type}")


def _save_settings_to_metadata(settings: Dict[str, Any], dataset: NumpyDataset) -> None:
    """
    Save all settings to metadata.json in the same folder as prices.npy.
    
    This allows the application to restore all configuration when loading the dataset again.
    """
    try:
        prices_dir = os.path.dirname(dataset.prices_file)
        metadata_path = os.path.join(prices_dir, "metadata.json")
        
        # Load existing metadata if it exists
        existing_metadata = {}
        if os.path.exists(metadata_path):
            try:
                import json
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load existing metadata: {e}")
                import traceback
                traceback.print_exc()
        
        # Update metadata with all settings
        existing_metadata["labeling_settings"] = {
            "labeling_type": settings.get("labeling_type"),
            "predict": settings.get("predict", {}),
            "classify": settings.get("classify", {}),
            "anomaly_detection": settings.get("anomaly_detection", {}),
            "labeling": settings.get("labeling", {}),
            "similarity": settings.get("similarity", {}),
            "data": {
                "prices_file": settings.get("data", {}).get("prices_file", ""),
                "timestamps_file": settings.get("data", {}).get("timestamps_file", ""),
                "ids_file": settings.get("data", {}).get("ids_file", ""),
                "cluster_ids_file": settings.get("data", {}).get("cluster_ids_file", ""),
                "labels_file": settings.get("data", {}).get("labels_file", ""),
                "metadata_file": settings.get("data", {}).get("metadata_file", "")
            },
            "saved_at": datetime.datetime.now().isoformat()
        }
        
        # Save metadata
        os.makedirs(prices_dir, exist_ok=True)
        import json
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(existing_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Settings saved to {metadata_path}")
        
    except Exception as e:
        print(f"Error saving settings to metadata: {e}")
        import traceback
        traceback.print_exc()
        raise


def print_settings_summary(settings: Dict[str, Any]) -> None:
    """Print a summary of active settings."""
    print("\n" + "="*60)
    print("SETTINGS SUMMARY")
    print("="*60)
    
    print(f"Labeling Type: {settings['labeling_type'].upper()}")
    
    # Similarity
    similarity = settings.get('similarity', {})
    print(f"Similarity Method: {similarity.get('method', 'soft_dtw')}")
    print(f"Similar Series Count: {similarity.get('num_similar', 4)}")
    
    # Data
    data = settings.get('data', {})
    print(f"Prices file: {data.get('prices_file', 'N/A')}")
    print(f"Timestamps file: {data.get('timestamps_file', 'N/A')}")
    if data.get('ids_file'):
        print(f"IDs file: {data.get('ids_file')}")
    if data.get('cluster_ids_file'):
        print(f"Cluster IDs file: {data.get('cluster_ids_file')}")
    if data.get('labels_file'):
        print(f"Labels file: {data.get('labels_file')}")
    
    print("="*60 + "\n")


def main():
    """Main application entry point."""
    print("="*60)
    print("TIME SERIES LABELING TOOL")
    print("="*60)
    print()
    
    # Open settings window
    print("Opening settings window...")
    settings = open_settings_window()
    
    if settings is None:
        print("Settings window closed without saving. Exiting.")
        return
    
    # Print settings summary
    print_settings_summary(settings)
    
    # Load dataset from folder and prefix (after settings configured)
    print("Loading dataset...")
    dataset = load_dataset(settings)
    
    # Save all settings to metadata.json for future loading
    _save_settings_to_metadata(settings, dataset)
    
    # Create similarity finder
    print("Initializing similarity search...")
    similarity_finder = create_similarity_finder(settings)
    
    # Create labeling application
    print("Launching labeling interface...")
    labeling_app = create_labeling_app(settings, dataset, similarity_finder)
    
    # Start labeling
    print("\n" + "="*60)
    print("LABELING STARTED")
    print("="*60)
    print()
    
    if settings['labeling_type'] == 'predict':
        print("Controls:")
        print("  - Click on plot to label prices")
        print("  - D: Show similar series")
        print("  - Arrow keys (←/→): Navigate")
        print("  - Ctrl+S: Save progress")
        print("  - Q: Quit")
    elif settings['labeling_type'] == 'classify':
        num_classes = settings['classify']['num_classes']
        keys = ', '.join([str(i) for i in range(1, num_classes + 1)])
        print("Controls:")
        print(f"  - {keys}: Classify into class")
        print("  - S: Show similar series")
        print("  - Arrow keys (←/→): Navigate")
        print("  - Q: Quit")
    else:  # anomaly_detection
        print("Controls:")
        print("  - Click on plot to mark/unmark anomaly point")
        print("  - D: Show similar series")
        print("  - Arrow keys (←/→): Navigate")
        print("  - Ctrl+S: Save progress")
        print("  - Q: Quit")
    
    print()
    
    try:
        labeling_app.start()
        print("\nLabeling session completed!")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving progress...")
        try:
            labeling_app.save_progress()
            print("Progress saved.")
        except Exception as e:
            print(f"\nError saving progress: {e}")
            import traceback
            traceback.print_exc()
            raise
    except Exception as e:
        print(f"\nError during labeling: {e}")
        import traceback
        traceback.print_exc()
        print("\nAttempting to save progress...")
        try:
            labeling_app.save_progress()
            print("Progress saved.")
        except Exception as save_error:
            print(f"Could not save progress: {save_error}")
            import traceback
            traceback.print_exc()
        # Re-raise the original exception so program crashes
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFatal error in main: {e}")
        import traceback
        traceback.print_exc()
        raise
