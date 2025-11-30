"""Base class for labeling types."""
import datetime
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import dates as mdates
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from src.dataset_loader import NumpyDataset

if TYPE_CHECKING:
    from src.similarity.base import BaseSimilarityFinder


class BaseLabelingType(ABC):
    """Abstract base class for labeling implementations."""

    BACKUP_INTERVAL: int = 15
    
    def __init__(self, dataset: NumpyDataset, settings: Dict[str, Any], 
                 similarity_finder: Optional['BaseSimilarityFinder'] = None) -> None:
        """
        Initialize labeling type.
        
        Args:
            dataset: NumPy dataset to label
            settings: Application settings dictionary
            similarity_finder: Similarity search algorithm instance
        """
        self.dataset: NumpyDataset = dataset
        self.settings: Dict[str, Any] = settings
        self.similarity_finder: Optional['BaseSimilarityFinder'] = similarity_finder
        self.current_index: int = 0
        self._previous_label_count: int = 0
        self._last_backup_count: int = 0
        
        # Matplotlib setup
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None
    
    @abstractmethod
    def setup_ui(self) -> None:
        """Setup the matplotlib figure and UI elements."""
        pass
    
    @abstractmethod
    def show_data(self) -> None:
        """Display current time series."""
        pass
    
    @abstractmethod
    def handle_key_press(self, event: Any) -> None:
        """Handle keyboard events."""
        pass
    
    @abstractmethod
    def handle_mouse_click(self, event: Any) -> None:
        """Handle mouse click events."""
        pass
    
    @abstractmethod
    def save_progress(self) -> None:
        """Save labeling progress to file."""
        pass
    
    def start(self) -> None:
        """Start the labeling interface."""
        self.setup_ui()
        self.show_data()
        plt.show()

    # ------------------------------------------------------------------
    # Helpers for labels persistence and backups
    # ------------------------------------------------------------------
    def _initialize_backup_tracking(self) -> None:
        """Initialize counters used for periodic backups."""
        total_labels = self._count_assigned_labels()
        self._previous_label_count = total_labels
        if total_labels >= self.BACKUP_INTERVAL:
            if total_labels % self.BACKUP_INTERVAL == 0:
                self._last_backup_count = total_labels
            else:
                self._last_backup_count = total_labels - (total_labels % self.BACKUP_INTERVAL)
        else:
            self._last_backup_count = 0

    def _count_assigned_labels(self) -> int:
        """Count non-NaN labels across the dataset."""
        labels = getattr(self.dataset, "labels", None)
        if labels is None:
            return 0
        try:
            return int(np.count_nonzero(~np.isnan(labels)))
        except TypeError:
            # Fallback for non-numeric labels (should not occur normally)
            if isinstance(labels, np.ndarray):
                return int(np.count_nonzero(labels != -1))
            return 0

    def _resolve_label_file_paths(self) -> Optional[Tuple[str, str, str]]:
        """Determine primary and backup labels file paths with dataset prefix."""
        data_settings = self.settings.setdefault("data", {})

        prices_file = data_settings.get("prices_file", "") or ""
        labels_file = data_settings.get("labels_file", "") or ""

        prices_dir: Optional[Path] = None
        if prices_file:
            prices_path = Path(prices_file)
            prices_dir = prices_path.parent

        labels_path: Optional[Path] = Path(labels_file) if labels_file else None

        if labels_path is not None:
            if not labels_path.is_absolute() and prices_dir is not None:
                labels_path = prices_dir / labels_path
            labels_dir = labels_path.parent if str(labels_path.parent) not in ("", ".") else prices_dir or Path(".")
        elif prices_dir is not None:
            labels_dir = prices_dir
            labels_path = labels_dir / "labels.npy"
        else:
            return None

        if labels_path is None:
            return None

        prefix = labels_dir.name if labels_dir.name else "dataset"
        desired_name = f"{prefix}_labels.npy"

        if labels_file == "" or labels_path.name == "labels.npy":
            labels_path = labels_dir / desired_name
        elif labels_path.name != desired_name:
            # Respect custom filenames but make sure we store resolved path for future sessions
            data_settings["labels_file"] = str(labels_path)

        if labels_path.name == desired_name:
            data_settings["labels_file"] = str(labels_path)

        backup_path = labels_dir / f"{prefix}_backup_labels.npy"

        return str(labels_path), str(backup_path), prefix

    def _maybe_handle_auto_backup(self, create_backup: bool = True) -> None:
        """Save automatic backups every BACKUP_INTERVAL labels."""
        total_labels = self._count_assigned_labels()
        previous_total = getattr(self, "_previous_label_count", 0)
        last_backup = getattr(self, "_last_backup_count", 0)

        if total_labels < previous_total and total_labels < last_backup:
            last_backup = total_labels - (total_labels % self.BACKUP_INTERVAL)

        if (
            create_backup
            and total_labels >= self.BACKUP_INTERVAL
            and total_labels % self.BACKUP_INTERVAL == 0
            and total_labels != last_backup
        ):
            resolved = self._resolve_label_file_paths()
            if resolved is not None:
                labels_file, backup_file, prefix = resolved
                os.makedirs(os.path.dirname(labels_file) or ".", exist_ok=True)
                np.save(labels_file, self.dataset.labels)
                np.save(backup_file, self.dataset.labels)
                self.dataset.labels_file = labels_file
                print(
                    f" Auto-backup saved to {backup_file} "
                    f"({total_labels} labels tracked for {prefix})"
                )
                last_backup = total_labels

        self._previous_label_count = total_labels
        self._last_backup_count = last_backup
    
    def go_forward(self) -> None:
        """Move to next time series."""
        if self.current_index < len(self.dataset) - 1:
            self.current_index += 1
            self.show_data()
            print(f"Moved forward to series {self.current_index + 1}/{len(self.dataset)}")
        else:
            print("Warning: This is the last series!")
    
    def go_backward(self) -> None:
        """Move to previous time series."""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_data()
            print(f"Moved backward to series {self.current_index + 1}/{len(self.dataset)}")
        else:
            print("Warning: This is the first series!")
    
    def find_first_unlabeled(self) -> None:
        """Find and move to first unlabeled sample."""
        if self.dataset.labels is None:
            # No labels exist, start at index 0
            self.current_index = 0
            return
        
        for i in range(len(self.dataset)):
            if not self.dataset.has_labels(i):
                self.current_index = i
                print(f"Found first unlabeled sample at index {i + 1}/{len(self.dataset)}")
                return
        print(" All samples are labeled!")
    
    def prepare_series_data(self, index: int, settings: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare series data - always use all raw data.
        
        Args:
            index: Sample index in dataset
            settings: Labeling-specific settings (predict or classify) - not used anymore
        
        Returns:
            Tuple of (y_values, x_values) - all raw data
        """
        # Get price values for this sample (y-axis data) - use all data
        y_values = self.dataset.prices[index].copy()
        
        # Get timestamps for X-axis (always use timestamps)
        # Handle both cases: timestamps can be per-sample array or shared array
        timestamps = self.dataset.timestamps
        if timestamps.dtype == object or timestamps.ndim > 1:
            # Per-sample timestamps (object array or 2D+ array)
            x_values = np.asarray(timestamps[index]).copy()
        else:
            # Shared timestamps for all samples (1D array)
            x_values = timestamps.copy()
        
        return y_values, x_values

    # ------------------------------------------------------------------
    # Helpers for X-axis formatting
    # ------------------------------------------------------------------
    def _detect_epoch_scale(self, sample_value: float) -> Optional[float]:
        """Detect if numeric value looks like UNIX epoch timestamp.

        Args:
            sample_value: Representative value from timestamps array.

        Returns:
            Divisor to convert the value to seconds (1, 1e3, 1e6, 1e9) or None.
        """
        # Test several common timestamp scales: seconds, milliseconds,
        # microseconds, nanoseconds.
        scales = (1.0, 1e3, 1e6, 1e9)
        time_window_start = datetime.datetime(1970, 1, 1)
        time_window_end = datetime.datetime(2100, 12, 31, 23, 59, 59)

        for scale in scales:
            try:
                seconds_value = sample_value / scale
                # Require sufficiently large magnitude to avoid treating simple
                # indices (0..N) as timestamps. 1e8 seconds ≈ 3 years.
                if abs(seconds_value) < 1e8:
                    continue

                candidate = datetime.datetime.fromtimestamp(seconds_value)
            except (OverflowError, OSError, ValueError):
                continue

            if time_window_start <= candidate <= time_window_end:
                return scale

        return None

    def _convert_x_axis_for_plot(self, x_values: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Convert raw X values to plotting-friendly format.

        Keeps numeric arrays untouched, converts epoch timestamps to matplotlib
        date numbers, and returns a flag indicating whether axis is datetime.

        Args:
            x_values: Array of raw X values (timestamps or indexes).

        Returns:
            Tuple of (converted_x_values, is_datetime_axis).
        """
        if x_values is None:
            return np.array([]), False

        x_array = np.asarray(x_values)
        if x_array.size == 0:
            return x_array.astype(float) if x_array.dtype != object else np.array([]), False

        # Case 1: numpy datetime64 values
        if np.issubdtype(x_array.dtype, np.datetime64):
            dt_list = x_array.astype('datetime64[ns]').tolist()
            if isinstance(dt_list, list):
                datetime_list = dt_list
            else:
                datetime_list = [dt_list]
            converted = mdates.date2num(datetime_list)
            return np.asarray(converted).reshape(x_array.shape), True

        # Case 2: numeric values that might be epoch timestamps
        if np.issubdtype(x_array.dtype, np.number):
            finite_values = x_array[np.isfinite(x_array)]
            if finite_values.size > 0:
                sample_value = float(np.median(finite_values))
                scale = self._detect_epoch_scale(sample_value)
                if scale is not None:
                    datetime_list = [
                        datetime.datetime.fromtimestamp(float(v) / scale)
                        if np.isfinite(v) else datetime.datetime.fromtimestamp(float(sample_value) / scale)
                        for v in x_array.flatten()
                    ]
                    converted = mdates.date2num(datetime_list)
                    return np.asarray(converted).reshape(x_array.shape), True

        # Fallback: attempt to return numeric array for plotting
        try:
            return x_array.astype(float), False
        except (TypeError, ValueError):
            return np.asarray(range(len(x_array)), dtype=float), False

    def _format_x_axis(self, is_datetime_axis: bool) -> None:
        """Apply consistent formatting to the X axis."""
        if self.ax is None or self.fig is None:
            return

        if is_datetime_axis:
            locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
            formatter = mdates.ConciseDateFormatter(locator)
            self.ax.xaxis.set_major_locator(locator)
            self.ax.xaxis.set_major_formatter(formatter)
            self.ax.figure.autofmt_xdate()
        else:
            # Reset to default formatting when not a datetime axis
            self.ax.xaxis.set_major_locator(MaxNLocator(nbins='auto'))
            scalar_formatter = ScalarFormatter()
            scalar_formatter.set_useOffset(False)
            self.ax.xaxis.set_major_formatter(scalar_formatter)


