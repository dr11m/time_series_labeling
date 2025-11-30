"""Settings manager for loading and saving application settings."""
import json
import os
from typing import Dict, Any
from pathlib import Path


DEFAULT_SETTINGS = {
    "labeling_type": "predict",
    "predict": {
        "num_prices": 2
    },
    "classify": {
        "num_classes": 5
    },
    "anomaly_detection": {
        # No specific settings needed
    },
    "labeling": {
        "right_padding": 0,
        "y_padding_percent": 15,
        "num_points_from_end": 0
    },
    "similarity": {
        "method": "soft_dtw",
        "num_similar": 4,
        "gamma": 0.05
    },
    "data": {
        "prices_file": "",
        "timestamps_file": "",
        "ids_file": "",
        "cluster_ids_file": "",
        "labels_file": "",
        "metadata_file": "",
        "predicted_prices_to_help_file": ""
    }
}


class SettingsManager:
    """Manage application settings with JSON persistence."""
    
    SETTINGS_FILE = "settings.json"
    
    @classmethod
    def load(cls) -> Dict[str, Any]:
        """Load settings from settings.json or create with defaults if not exists."""
        if os.path.exists(cls.SETTINGS_FILE):
            try:
                with open(cls.SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                print(f" Settings loaded from {cls.SETTINGS_FILE}")
                return settings
            except Exception as e:
                print(f"Warning: Error loading settings: {e}. Using defaults.")
                return DEFAULT_SETTINGS.copy()
        else:
            print(f" Settings file not found. Creating with defaults.")
            cls.save(DEFAULT_SETTINGS)
            return DEFAULT_SETTINGS.copy()
    
    @classmethod
    def save(cls, settings: Dict[str, Any]) -> None:
        """Save settings to settings.json."""
        try:
            with open(cls.SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            print(f" Settings saved to {cls.SETTINGS_FILE}")
        except Exception as e:
            print(f"✗ Error saving settings: {e}")
    
    @classmethod
    def get_default_settings(cls) -> Dict[str, Any]:
        """Get a copy of default settings."""
        return DEFAULT_SETTINGS.copy()


