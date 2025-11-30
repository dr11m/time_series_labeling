"""Anomaly detection labeling type for time series."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector
from typing import Dict, Any, Optional, Tuple
from src.labeling_types.base import BaseLabelingType
from src.dataset_loader import NumpyDataset
from src.similarity.base import BaseSimilarityFinder


class AnomalyDetectionLabeling(BaseLabelingType):
    """Labeling type for detecting anomalies in time series."""
    BACKUP_INTERVAL: int = 10
    
    def __init__(self, dataset: NumpyDataset, settings: Dict[str, Any],
                 similarity_finder: BaseSimilarityFinder = None):
        """
        Initialize anomaly detection labeling.
        
        Args:
            dataset: NumPy dataset
            settings: Application settings
            similarity_finder: Similarity search instance
        """
        super().__init__(dataset, settings, similarity_finder)
        
        # Initialize labels array if not exists
        # Shape: (N, sequence_length) - each element is 0 (normal) or 1 (anomaly)
        if self.dataset.labels is None:
            # Get sequence length from first sample
            sequence_length = len(self.dataset.prices[0]) if len(self.dataset) > 0 else 0
            self.dataset.labels = np.zeros((len(self.dataset), sequence_length), dtype=np.int32)
            print(f" Initialized labels array: shape ({len(self.dataset)}, {sequence_length})")
        else:
            # Validate shape
            if self.dataset.labels.ndim != 2:
                # Convert 1D to 2D if needed (for backward compatibility)
                sequence_length = len(self.dataset.prices[0]) if len(self.dataset) > 0 else 0
                if len(self.dataset.labels) == len(self.dataset) * sequence_length:
                    self.dataset.labels = self.dataset.labels.reshape((len(self.dataset), sequence_length))
                    print(f" Reshaped labels array to 2D: ({len(self.dataset)}, {sequence_length})")
                else:
                    # Create new array
                    new_labels = np.zeros((len(self.dataset), sequence_length), dtype=np.int32)
                    # Try to copy existing data
                    min_len = min(len(self.dataset.labels), len(self.dataset) * sequence_length)
                    new_labels.flat[:min_len] = self.dataset.labels.flat[:min_len]
                    self.dataset.labels = new_labels
                    print(f" Recreated labels array: shape ({len(self.dataset)}, {sequence_length})")
            else:
                # Check if sequence length matches
                expected_length = len(self.dataset.prices[0]) if len(self.dataset) > 0 else 0
                if self.dataset.labels.shape[1] != expected_length:
                    # Resize labels to match current sequence length
                    old_shape = self.dataset.labels.shape
                    new_labels = np.zeros((len(self.dataset), expected_length), dtype=np.int32)
                    # Copy overlapping part
                    min_cols = min(old_shape[1], expected_length)
                    new_labels[:, :min_cols] = self.dataset.labels[:, :min_cols]
                    self.dataset.labels = new_labels
                    print(f" Resized labels array from {old_shape[1]} to {expected_length} columns")
        
        # Ensure labels are int32 (0 or 1)
        self.dataset.labels = self.dataset.labels.astype(np.int32)
        # Clamp values to 0 or 1
        self.dataset.labels = np.clip(self.dataset.labels, 0, 1)
        
        # Prepare backup tracking state
        self._initialize_backup_tracking()
        
        # UI buttons storage
        self.control_buttons = {}
        
        # No cursor/hover effects for anomaly detection
        
        # Zoom region selection
        self.zoom_region: Optional[Tuple[float, float, float, float]] = None
        self.rectangle_selector: Optional[RectangleSelector] = None
        self.zoom_mode_active = False
        
        # Find first unlabeled sample
        self.find_first_unlabeled()
    
    def find_first_unlabeled(self) -> None:
        """Load last index from metadata or start from beginning."""
        last_index = self._load_last_index_from_metadata()
        if last_index is not None and 0 <= last_index < len(self.dataset):
            self.current_index = last_index
            print(f" Resumed anomaly detection from index {last_index + 1}/{len(self.dataset)}")
        else:
            self.current_index = 0
            print(f" Starting anomaly detection from beginning. Dataset has {len(self.dataset)} samples.")
        
        labeled_count = self._count_anomaly_labels()
        total_points = self.dataset.labels.size if self.dataset.labels is not None else 0
        print(f" Found {labeled_count}/{total_points} anomaly labels.")
    
    def _count_anomaly_labels(self) -> int:
        """Count total number of anomaly labels (1s) in the dataset."""
        if self.dataset.labels is None:
            return 0
        return int(np.sum(self.dataset.labels == 1))
    
    def _count_assigned_labels(self) -> int:
        """Count number of series (indices) that have at least one anomaly.

        This makes auto-backups trigger every N labeled indices rather than
        every N individual anomaly points.
        """
        labels = getattr(self.dataset, "labels", None)
        if labels is None or labels.size == 0:
            return 0
        try:
            # Count rows that contain at least one '1'
            return int(np.count_nonzero(np.any(labels == 1, axis=1)))
        except Exception:
            return 0
    
    def _maybe_handle_auto_backup(self, create_backup: bool = True) -> None:
        """Index-based auto-backup: every BACKUP_INTERVAL graphs (indices)."""
        if not create_backup:
            return
        # Backup when moving onto indices 10, 20, 30, ... (1-based)
        if (self.current_index + 1) % self.BACKUP_INTERVAL != 0:
            return
        resolved = self._resolve_label_file_paths()
        if resolved is None:
            return
        labels_file, backup_file, prefix = resolved
        try:
            os.makedirs(os.path.dirname(labels_file) or ".", exist_ok=True)
            # Save labels and backup
            np.save(labels_file, self.dataset.labels)
            np.save(backup_file, self.dataset.labels)
            self.dataset.labels_file = labels_file
            # Persist last index for resume
            self._save_last_index_to_metadata(self.current_index)
            print(
                f" Auto-backup (index-based) saved to {backup_file} at index {self.current_index + 1}/{len(self.dataset)} for {prefix}"
            )
        except Exception as e:
            print(f"Warning: Auto-backup failed: {e}")
    
    def _get_metadata_path(self) -> str:
        """Get path to metadata.json file."""
        prices_dir = os.path.dirname(self.dataset.prices_file)
        return os.path.join(prices_dir, "metadata.json")
    
    def _load_last_index_from_metadata(self) -> Optional[int]:
        """Load last index from metadata.json if exists."""
        metadata_path = self._get_metadata_path()
        
        if not os.path.exists(metadata_path):
            return None
        
        try:
            import json
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Check for anomaly_detection last_index
            if isinstance(metadata, dict):
                if 'anomaly_detection' in metadata and isinstance(metadata['anomaly_detection'], dict):
                    last_index = metadata['anomaly_detection'].get('last_index')
                    if isinstance(last_index, (int, float)):
                        return int(last_index)
            
            return None
        except Exception as e:
            print(f"Warning: Could not load last index from metadata: {e}")
            return None
    
    def _save_last_index_to_metadata(self, index: int) -> None:
        """Save current index to metadata.json."""
        metadata_path = self._get_metadata_path()
        
        # Load existing metadata or create new
        existing_metadata = {}
        if os.path.exists(metadata_path):
            try:
                import json
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load existing metadata: {e}")
        
        # Ensure metadata is a dict
        if not isinstance(existing_metadata, dict):
            existing_metadata = {}
        
        # Update anomaly_detection section
        if 'anomaly_detection' not in existing_metadata:
            existing_metadata['anomaly_detection'] = {}
        
        existing_metadata['anomaly_detection']['last_index'] = index
        
        # Save metadata
        try:
            import json
            os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(existing_metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save last index to metadata: {e}")
    
    def _get_processed_data(self, index: int) -> np.ndarray:
        """Get processed data for similarity search (same as shown in plot)."""
        labeling_settings = self.settings.get("labeling", {})
        y_values, _ = self.prepare_series_data(index, labeling_settings)
        return y_values
    
    def setup_ui(self):
        """Setup matplotlib figure with buttons."""
        plt.ioff()
        self.fig = plt.figure(figsize=(14, 8))
        
        # Main plot area (leave space for buttons at bottom)
        self.ax = plt.axes([0.1, 0.15, 0.8, 0.75])
        
        # Create buttons
        self._create_buttons()
        
        # Connect events
        self.fig.canvas.mpl_connect('key_press_event', self.handle_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.handle_mouse_click)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
    
    def _create_buttons(self):
        """Create control buttons."""
        button_width = 0.12
        button_height = 0.04
        start_x = 0.1
        y_predict = 0.08  # First row
        y_control = 0.02  # Second row
        
        # First row: Reset and Zoom buttons
        ax_reset = plt.axes([start_x, y_predict, button_width, button_height])
        btn_reset = Button(ax_reset, 'Reset Labels', color='#DC143C')
        btn_reset.on_clicked(lambda event: self.reset_labels())
        self.control_buttons['Reset Labels'] = btn_reset
        
        # Zoom region button
        ax_zoom = plt.axes([start_x + button_width + 0.02, y_predict, button_width, button_height])
        btn_zoom = Button(ax_zoom, 'Zoom Region', color='#9C27B0')  # Purple
        btn_zoom.on_clicked(lambda event: self.toggle_zoom_mode())
        self.control_buttons['Zoom Region'] = btn_zoom
        
        # Reset zoom button
        ax_reset_zoom = plt.axes([start_x + 2 * (button_width + 0.02), y_predict, button_width, button_height])
        btn_reset_zoom = Button(ax_reset_zoom, 'Reset Zoom', color='#FF9800')  # Orange
        btn_reset_zoom.on_clicked(lambda event: self.reset_zoom())
        self.control_buttons['Reset Zoom'] = btn_reset_zoom
        
        # Second row: Control buttons
        total_width = 0.8
        control_buttons_info = [
            ('Similar', self.show_similar_series, '#ffa500'),  # Orange
            ('Save', self.save_progress, '#4CAF50'),           # Green
            ('← Prev', self.go_backward, '#2196F3'),           # Blue
            ('Next →', self.go_forward, '#2196F3')             # Blue
        ]
        
        control_spacing = (total_width - (len(control_buttons_info) * button_width)) / (len(control_buttons_info) - 1)
        
        for i, (text, callback, color) in enumerate(control_buttons_info):
            x_pos = start_x + i * (button_width + control_spacing)
            ax_btn = plt.axes([x_pos, y_control, button_width, button_height])
            btn = Button(ax_btn, text, color=color)
            btn.on_clicked(lambda event, cb=callback: cb())
            self.control_buttons[text] = btn
    
    def reset_labels(self):
        """Reset all labels for current sample."""
        if self.current_index >= len(self.dataset):
            return
        
        self.dataset.labels[self.current_index] = 0
        print("  Labels reset for current sample")
        self._maybe_handle_auto_backup(create_backup=False)
        self.show_data()
    
    def toggle_zoom_mode(self):
        """Toggle zoom region selection mode."""
        self.zoom_mode_active = not self.zoom_mode_active
        if self.zoom_mode_active:
            self._setup_rectangle_selector()
            print(" Zoom mode ON: Click and drag to select region")
        else:
            self._remove_rectangle_selector()
            print(" Zoom mode OFF")
    
    def reset_zoom(self):
        """Reset zoom region to show full data."""
        self.zoom_region = None
        self.zoom_mode_active = False
        self._remove_rectangle_selector()
        print(" Zoom reset - showing full data")
        self.show_data()
    
    def _setup_rectangle_selector(self):
        """Setup RectangleSelector for zoom region selection."""
        self._remove_rectangle_selector()
        
        if self.ax is None:
            return
        
        self.rectangle_selector = RectangleSelector(
            self.ax,
            self._on_rectangle_select,
            useblit=True,
            button=[1],
            minspanx=0,
            minspany=0,
            spancoords='data',
            interactive=True
        )
        self.fig.canvas.draw_idle()
    
    def _remove_rectangle_selector(self):
        """Remove RectangleSelector."""
        if self.rectangle_selector is not None:
            self.rectangle_selector.set_active(False)
            self.rectangle_selector = None
    
    def _on_rectangle_select(self, eclick, erelease):
        """Handle rectangle selection for zoom region."""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return
        
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)
        
        self.zoom_region = (x_min, x_max, y_min, y_max)
        self.zoom_mode_active = False
        self._remove_rectangle_selector()
        
        self.show_data()
        print(f" Zoom region set: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}]")
        print(" Zoom mode auto-disabled - you can now label anomalies")
    
    def show_data(self):
        """Display current time series with anomaly markers."""
        if self.current_index >= len(self.dataset):
            self.show_completion_message()
            return
        
        # Prepare data according to settings
        labeling_settings = self.settings.get("labeling", {})
        y_values, raw_x_values = self.prepare_series_data(self.current_index, labeling_settings)
        x_values, is_datetime_axis = self._convert_x_axis_for_plot(raw_x_values)
        
        # Clear and plot
        self.ax.clear()
        
        # Flatten data for plotting
        x_values_flat = np.asarray(x_values).flatten()
        y_values_flat = np.asarray(y_values).flatten()
        
        # Get anomaly labels for this sample
        anomaly_labels = self.dataset.labels[self.current_index]
        
        # Get labeling display settings
        right_padding = labeling_settings.get("right_padding", 0)
        num_points_from_end = labeling_settings.get("num_points_from_end", 0)
        
        # Plot all points (blue)
        if len(x_values_flat) > 0:
            if len(x_values_flat) >= 2:
                self.ax.plot(x_values_flat, y_values_flat, marker='o', color='blue', 
                           markersize=6, label='Normal', alpha=0.7)
            else:
                self.ax.scatter(x_values_flat, y_values_flat, color='blue', s=50, 
                              label='Normal', alpha=0.7)
        
        # Mark anomaly points with red X marker on top of existing points
        if len(x_values_flat) == len(anomaly_labels):
            anomaly_mask = anomaly_labels == 1
            if np.any(anomaly_mask):
                x_anomaly = x_values_flat[anomaly_mask]
                y_anomaly = y_values_flat[anomaly_mask]
                self.ax.scatter(x_anomaly, y_anomaly, color='red', s=200, 
                              zorder=6, edgecolors='darkred', linewidths=2, 
                              alpha=0.9, marker='X', label='Anomaly')
        
        # Handle highlighting of last N points if enabled
        if num_points_from_end > 0 and len(x_values_flat) > 0:
            n_points = min(num_points_from_end, len(x_values_flat))
            if n_points > 0 and n_points < len(x_values_flat):
                # Re-highlight last N points in orange (on top of existing colors)
                end_x = x_values_flat[-n_points:]
                end_y = y_values_flat[-n_points:]
                self.ax.scatter(end_x, end_y, color='orange', s=100, zorder=7, 
                              edgecolors='darkorange', linewidths=1.5, alpha=0.8)
        
        # Extract X limits for data
        x_min_data = float(x_values_flat[0]) if len(x_values_flat) > 0 else 0
        x_max_data = float(x_values_flat[-1]) if len(x_values_flat) > 0 else 1
        
        # Add padding visualization if enabled
        if right_padding > 0 and len(x_values_flat) > 0:
            if len(x_values_flat) > 1:
                x_spacing = (x_max_data - x_min_data) / (len(x_values_flat) - 1) if len(x_values_flat) > 1 else 1.0
            else:
                x_spacing = 1.0
            
            padding_end_x = x_max_data + (right_padding * x_spacing)
            line_x_position = x_max_data + (right_padding * x_spacing * 0.65)
            
            self.ax.axvline(x=line_x_position, color='orange', linestyle='--', linewidth=2, alpha=0.7)
            
            x_range = x_max_data - x_min_data
            x_margin = x_range * 0.05 if x_range > 0 else 1.0
            self.ax.set_xlim(x_min_data - x_margin, padding_end_x)
        
        # Set Y-axis limits with margin
        if len(y_values) > 0:
            y_min = float(np.min(y_values))
            y_max = float(np.max(y_values))
            margin_percent = labeling_settings.get("y_padding_percent", 15)
            margin = (y_max - y_min) * (margin_percent / 100.0) if y_max != y_min else abs(y_min) * (margin_percent / 100.0) if y_min != 0 else 1.0
            if margin == 0:
                margin = max(abs(y_min), abs(y_max), 1.0) * 0.1
            self.ax.set_ylim(y_min - margin, y_max + margin)
        
        # Set X limits (padding already applied if enabled)
        if self.zoom_region is not None:
            self.ax.set_xlim(self.zoom_region[0], self.zoom_region[1])
            self.ax.set_ylim(self.zoom_region[2], self.zoom_region[3])
        else:
            if right_padding == 0:
                x_range = x_max_data - x_min_data
                x_margin = x_range * 0.05 if x_range > 0 else 1.0
                self.ax.set_xlim(x_min_data - x_margin, x_max_data + x_margin)
        
        # Add legend
        if len(anomaly_labels) == len(x_values_flat) and np.any(anomaly_labels == 1):
            self.ax.legend(loc='upper left', fontsize=9)
        
        # Labels and title
        self.ax.set_xlabel("Date" if is_datetime_axis else "Timestamp")
        self.ax.set_ylabel("Value")
        sample_name = self.dataset.get_sample_name(self.current_index)
        anomaly_count = int(np.sum(anomaly_labels == 1))
        title = f"Index: {self.current_index + 1}/{len(self.dataset)} - {sample_name} | Anomalies: {anomaly_count}"
        self.ax.set_title(title)
        self.ax.grid(True)
        
        # Recreate RectangleSelector if zoom mode is active
        if self.zoom_mode_active:
            self._setup_rectangle_selector()
        
        # Apply axis formatting
        self._format_x_axis(is_datetime_axis)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def go_forward(self):
        """Move to next time series."""
        if self.current_index < len(self.dataset) - 1:
            self.current_index += 1
            self.zoom_region = None
            self.zoom_mode_active = False
            self._remove_rectangle_selector()
            # Index-based auto-backup every 10 graphs
            self._maybe_handle_auto_backup()
            self.show_data()
            print(f"Moved forward to series {self.current_index + 1}/{len(self.dataset)}")
        else:
            print("Warning: This is the last series!")
    
    def go_backward(self):
        """Move to previous time series."""
        if self.current_index > 0:
            self.current_index -= 1
            self.zoom_region = None
            self.zoom_mode_active = False
            self._remove_rectangle_selector()
            self.show_data()
            print(f"Moved backward to series {self.current_index + 1}/{len(self.dataset)}")
        else:
            print("Warning: This is the first series!")
    
    
    def handle_key_press(self, event):
        """Handle keyboard events."""
        if event.key == "right":
            self.go_forward()
        elif event.key == "left":
            self.go_backward()
        elif event.key == "d":
            self.show_similar_series()
        elif event.key == "ctrl+s":
            self.save_progress()
        elif event.key == "q":
            plt.close()
    
    def handle_mouse_click(self, event):
        """Handle mouse clicks for anomaly labeling."""
        # Only process clicks on the main plot axes
        if event.inaxes != self.ax:
            return
        
        # Only process left mouse button clicks
        if event.button != 1:
            return
        
        # Don't process clicks if zoom mode is active
        if self.zoom_mode_active:
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        # Get current data
        labeling_settings = self.settings.get("labeling", {})
        y_values, raw_x_values = self.prepare_series_data(self.current_index, labeling_settings)
        x_values, _ = self._convert_x_axis_for_plot(raw_x_values)
        
        x_values_flat = np.asarray(x_values).flatten()
        
        # Find closest point by X coordinate
        if len(x_values_flat) == 0:
            return
        
        distances = np.abs(x_values_flat - x)
        closest_idx = np.argmin(distances)
        
        # Toggle anomaly label for closest point
        if closest_idx < len(self.dataset.labels[self.current_index]):
            current_label = self.dataset.labels[self.current_index, closest_idx]
            # Toggle: if 0 -> 1, if 1 -> 0
            new_label = 1 if current_label == 0 else 0
            self.dataset.labels[self.current_index, closest_idx] = new_label
            
            status = "marked as anomaly" if new_label == 1 else "unmarked (normal)"
            print(f" Point {closest_idx} {status} (X={x_values_flat[closest_idx]:.3f})")
            
            self._maybe_handle_auto_backup()
            self.show_data()
    
    def show_similar_series(self):
        """Show similar labeled series using similarity finder."""
        if self.similarity_finder is None:
            print("Warning: Similarity finder not configured")
            return
        
        if self.current_index >= len(self.dataset):
            return
        
        # Prepare processed data for current sample
        current_processed_data = self._get_processed_data(self.current_index)
        
        # Get processed data for all other samples (can compare with any, not just labeled ones)
        other_processed_data = []
        for i in range(len(self.dataset)):
            if i != self.current_index:
                processed_data = self._get_processed_data(i)
                # Keep only same-length series
                if len(processed_data) == len(current_processed_data) and len(processed_data) > 0:
                    other_processed_data.append((i, processed_data))
        
        if not other_processed_data:
            print("Warning: No other samples with matching length to compare")
            return
        
        # Find similar samples
        similar_results = self.similarity_finder.find_similar_with_processed_data(
            current_processed_data, other_processed_data)
        
        # Visualize results
        if similar_results:
            print(f" Found {len(similar_results)} similar samples")
            similar_with_indices = [(idx, data, dist) for idx, data, dist in similar_results]
            self.similarity_finder.visualize_similar_with_processed_data(
                self.current_index, current_processed_data, similar_with_indices,
                dataset=self.dataset, labeling_type='anomaly_detection')
        else:
            print("Error: No similar samples found")
    
    def save_progress(self):
        """Save labeling progress to labels.npy file and update metadata with last index."""
        resolved_paths = self._resolve_label_file_paths()
        
        if resolved_paths is None:
            print("Error: Error: Cannot determine labels file path")
            return
        
        labels_file, _, _ = resolved_paths
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(labels_file) or ".", exist_ok=True)
        
        # Save labels array
        np.save(labels_file, self.dataset.labels)
        
        # Update dataset labels_file path
        self.dataset.labels_file = labels_file
        
        # Save current index to metadata
        self._save_last_index_to_metadata(self.current_index)
        
        anomaly_count = self._count_anomaly_labels()
        total_points = self.dataset.labels.size
        print(f" Progress saved to {labels_file} ({anomaly_count}/{total_points} anomalies labeled, index {self.current_index + 1}/{len(self.dataset)})")
    
    def on_close(self, event):
        """Handle window close event."""
        print("\n Window closed. Saving progress...")
        self.save_progress()
    
    def show_completion_message(self):
        """Show completion message when at end of dataset."""
        self.ax.clear()
        self.ax.text(0.5, 0.5, " Anomaly Detection Mode\n\nPress Q to exit",
                    ha='center', va='center', transform=self.ax.transAxes,
                    fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        self.ax.set_title("Anomaly Detection")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

