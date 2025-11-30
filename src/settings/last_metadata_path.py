"""Module for managing last used metadata file path."""
import os
import json
from pathlib import Path


LAST_METADATA_PATH_FILE = Path(__file__).parent / "last_metadata_path.json"


def save_last_metadata_path(metadata_path: str) -> None:
    """
    Save the path to the last used metadata.json file.
    
    Args:
        metadata_path: Path to metadata.json file
    """
    try:
        data = {"metadata_path": metadata_path}
        with open(LAST_METADATA_PATH_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f" Last metadata path saved: {metadata_path}")
    except Exception as e:
        print(f"Warning: Could not save last metadata path: {e}")


def load_last_metadata_path() -> str:
    """
    Load the path to the last used metadata.json file.
    
    Returns:
        Path to metadata.json file or empty string if not found
    """
    if not LAST_METADATA_PATH_FILE.exists():
        return ""
    
    try:
        with open(LAST_METADATA_PATH_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        metadata_path = data.get("metadata_path", "")
        
        # Verify file still exists
        if metadata_path and os.path.exists(metadata_path):
            return metadata_path
        else:
            return ""
    except Exception as e:
        print(f"Warning: Could not load last metadata path: {e}")
        return ""

