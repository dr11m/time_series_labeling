"""Classification labeling type for time series."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Cursor, RectangleSelector
from typing import Dict, Any, List, Tuple, Optional
from src.labeling_types.base import BaseLabelingType
from src.dataset_loader import NumpyDataset
from src.similarity.base import BaseSimilarityFinder


class ClassifyLabeling(BaseLabelingType):
    """Labeling type for classifying time series into categories."""
    
    def __init__(self, dataset: NumpyDataset, settings: Dict[str, Any],
                 similarity_finder: BaseSimilarityFinder = None):
        """
        Initialize classify labeling.
        
        Args:
            dataset: NumPy dataset
            settings: Application settings
            similarity_finder: Similarity search instance
        """
        super().__init__(dataset, settings, similarity_finder)
        
        # Classify-specific settings
        classify_settings = settings.get("classify", {})
        self.num_classes = classify_settings.get("num_classes", 5)
        
        # Initialize labels array if not exists
        if self.dataset.labels is None:
            # Shape: (N,) - each element contains class index
            self.dataset.labels = np.full(len(self.dataset), np.nan, dtype=np.float64)
        
        # Button references for UI controls
        self.class_buttons: Dict[int, Button] = {}
        self.control_buttons: Dict[str, Button] = {}

        # Color map for classes (1 = red/unpredictable, 5 = green/predictable)
        self.color_map = self._generate_color_map()

        # Zoom/cursor state
        self.cursor_line = None
        self.cursor: Optional[Cursor] = None
        self.zoom_region: Optional[Tuple[float, float, float, float]] = None
        self.rectangle_selector: Optional[RectangleSelector] = None
        self.zoom_mode_active = False
        
        # Prepare backup tracking state
        self._initialize_backup_tracking()

        # Find first unlabeled sample
        self.find_first_unlabeled()
    
    def _generate_color_map(self):
        """Generate color map based on number of classes."""
        if self.num_classes == 5:
            return {
                1: '#d73027',  # Red
                2: '#fc8d59',  # Orange
                3: '#fee08b',  # Yellow
                4: '#d9ef8b',  # Light green
                5: '#1a9850'   # Green
            }
        else:
            # Generate gradient for other numbers
            colors = {}
            for i in range(1, self.num_classes + 1):
                # Gradient from red to green
                ratio = (i - 1) / max(1, self.num_classes - 1)
                r = int(255 * (1 - ratio))
                g = int(255 * ratio)
                colors[i] = f'#{r:02x}{g:02x}40'
            return colors
    
    
    def setup_ui(self):
        """Setup matplotlib figure with buttons."""
        plt.ioff()
        self.fig = plt.figure(figsize=(14, 8))  # Slightly taller for buttons
        
        # Main plot area (leave space for buttons at bottom)
        self.ax = plt.axes([0.1, 0.15, 0.8, 0.75])
        
        # Create buttons
        self._create_buttons()
        
        # Connect events
        self.fig.canvas.mpl_connect('key_press_event', self.handle_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.handle_mouse_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.handle_mouse_move)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
    
    def _create_buttons(self):
        """Create classification and control buttons."""
        # First row: Classification buttons
        button_width = 0.08
        button_height = 0.04
        start_x = 0.1
        y_class = 0.08  # First row
        y_control = 0.02  # Second row
        
        # Calculate spacing for class buttons
        total_width = 0.8
        if self.num_classes > 1:
            spacing = (total_width - (self.num_classes * button_width)) / (self.num_classes - 1)
        else:
            spacing = 0
        
        # Create class buttons (1st row)
        for i in range(1, self.num_classes + 1):
            x_pos = start_x + (i - 1) * (button_width + spacing)
            ax_btn = plt.axes([x_pos, y_class, button_width, button_height])
            
            # Use class color for button
            btn_color = self.color_map.get(i, '#cccccc')
            btn = Button(ax_btn, f'Class {i}', color=btn_color)
            btn.on_clicked(lambda event, cls=i: self.set_classification(cls))
            self.class_buttons[i] = btn
        
        # Second row: Control buttons
        control_buttons_info = [
            ('Zoom Region', self.toggle_zoom_mode, '#9C27B0'),   # Purple
            ('Reset Zoom', self.reset_zoom, '#FF9800'),          # Orange
            ('Similar', self.show_similar_series, '#ffa500'),  # Orange
            ('Save', self.save_progress, '#4CAF50'),           # Green
            ('Export All', self.export_xy_data, '#9C27B0'),    # Purple
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
        print(" Zoom mode auto-disabled - you can continue classifying")
    
    def show_data(self):
        """Display current time series."""
        if self.current_index >= len(self.dataset):
            self.show_completion_message()
            return
        
        # Prepare data according to settings (returns y_values, x_values as numpy arrays)
        classify_settings = self.settings.get("classify", {})
        y_values, raw_x_values = self.prepare_series_data(self.current_index, classify_settings)
        x_values, is_datetime_axis = self._convert_x_axis_for_plot(raw_x_values)
        
        # Determine color based on classification
        label = self.dataset.labels[self.current_index]
        if np.isnan(label):
            color = 'blue'
            class_label = "Not classified"
        else:
            current_class = int(label)
            color = self.color_map.get(current_class, 'blue')
            class_label = f"Class {current_class}"
        
        # Clear and plot
        self.ax.clear()
        self.cursor_line = None
        
        # Flatten data for plotting convenience
        x_values_flat = np.asarray(x_values).flatten()
        y_values_flat = np.asarray(y_values).flatten()
        
        # Get labeling display settings
        labeling_settings = self.settings.get("labeling", {})
        right_padding = labeling_settings.get("right_padding", 0)
        num_points_from_end = labeling_settings.get("num_points_from_end", 0)
        
        # Plot with optional highlighting of last N points
        if num_points_from_end > 0 and len(x_values_flat) > 0:
            n_points = min(num_points_from_end, len(x_values_flat))
            if n_points > 0 and n_points < len(x_values_flat):
                main_x = x_values_flat[:-n_points]
                main_y = y_values_flat[:-n_points]
                end_x = x_values_flat[-n_points:]
                end_y = y_values_flat[-n_points:]
                
                if len(main_x) > 0:
                    self.ax.scatter(main_x, main_y, c=color, s=90, alpha=0.8)
                    if len(main_x) >= 2:
                        self.ax.plot(main_x, main_y, color=color, alpha=0.6, linewidth=2)
                
                # Highlight last N points in orange
                self.ax.scatter(end_x, end_y, color='orange', s=110, zorder=5,
                                edgecolors='darkorange', linewidths=1.5, alpha=0.9)
                
                if len(end_x) >= 2:
                    self.ax.plot(end_x, end_y, color='orange', alpha=0.6, linewidth=2)
            else:
                # All points highlighted when n_points >= len(data)
                self.ax.scatter(x_values_flat, y_values_flat, color='orange', s=110, zorder=5,
                                edgecolors='darkorange', linewidths=1.5, alpha=0.9)
                if len(x_values_flat) >= 2:
                    self.ax.plot(x_values_flat, y_values_flat, color='orange', alpha=0.6, linewidth=2)
        else:
            # Default plotting without highlighting
            self.ax.scatter(x_values_flat, y_values_flat, c=color, s=90, alpha=0.8)
            if len(x_values_flat) >= 2:
                self.ax.plot(x_values_flat, y_values_flat, color=color, alpha=0.6, linewidth=2)
        
        # Extract X limits for data
        x_min_data = float(x_values_flat[0]) if len(x_values_flat) > 0 else 0
        x_max_data = float(x_values_flat[-1]) if len(x_values_flat) > 0 else 1
        
        # Add padding visualization if enabled
        if right_padding > 0 and len(x_values_flat) > 0:
            # Calculate padding space on X-axis
            if len(x_values_flat) > 1:
                # Calculate average spacing between points
                x_spacing = (x_max_data - x_min_data) / (len(x_values_flat) - 1) if len(x_values_flat) > 1 else 1.0
            else:
                x_spacing = 1.0
            
            # Calculate padding end position
            padding_end_x = x_max_data + (right_padding * x_spacing)
            
            # Calculate line position at 90% of padding
            line_x_position = x_max_data + (right_padding * x_spacing * 0.65)
            
            # Draw orange dashed vertical line at 90% of padding
            self.ax.axvline(x=line_x_position, color='orange', linestyle='--', linewidth=2, alpha=0.7)
            
            # Update X limits to include padding
            x_range = x_max_data - x_min_data
            x_margin = x_range * 0.05 if x_range > 0 else 1.0  # 5% margin on left side
            self.ax.set_xlim(x_min_data - x_margin, padding_end_x)
        
        # Set Y-axis limits with margin
        if len(y_values) > 0:
            y_min = float(np.min(y_values))
            y_max = float(np.max(y_values))
            margin_percent = 15  # Default margin
            margin = (y_max - y_min) * (margin_percent / 100.0) if y_max != y_min else abs(y_min) * (margin_percent / 100.0) if y_min != 0 else 1.0
            # Ensure margin is at least a small value if all values are the same
            if margin == 0:
                margin = max(abs(y_min), abs(y_max), 1.0) * 0.1
            self.ax.set_ylim(y_min - margin, y_max + margin)
        
        # Set X limits (padding already applied if enabled)
        if right_padding == 0:
            if len(x_values) > 0:
                x_min = float(x_values[0])
                x_max = float(x_values[-1])
                x_range = x_max - x_min
                x_margin = x_range * 0.05 if x_range > 0 else 1.0
                self.ax.set_xlim(x_min - x_margin, x_max + x_margin)

        # Apply zoom region if set
        if self.zoom_region is not None:
            self.ax.set_xlim(self.zoom_region[0], self.zoom_region[1])
            self.ax.set_ylim(self.zoom_region[2], self.zoom_region[3])
        
        # Title and labels
        sample_name = self.dataset.get_sample_name(self.current_index)
        prediction_hint = self._format_prediction_hint(self.current_index)
        title = f"Sample {self.current_index + 1}/{len(self.dataset)} | {sample_name}    —    {class_label}"
        if prediction_hint:
            title += f" • {prediction_hint}"
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        
        self.ax.set_xlabel("Date" if is_datetime_axis else "Timestamp", fontsize=12)
        self.ax.set_ylabel("Value", fontsize=12)
        self.ax.grid(True, alpha=0.3)
        
        # Instructions
        hint_text = f"Use buttons below or keyboard: 1-{self.num_classes} to classify • S for similar • Ctrl+S save • ←/→ navigate • Q quit"
        self.ax.text(0.5, 0.98, hint_text, ha='center', va='top', transform=self.ax.transAxes,
                    fontsize=9, style='italic', bbox=dict(boxstyle="round,pad=0.2", 
                    facecolor="lightyellow", alpha=0.7))
        
        self._format_x_axis(is_datetime_axis)

        # Cursor for precise inspection
        if not hasattr(self, 'cursor') or self.cursor is None:
            self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)

        # Recreate RectangleSelector if zoom mode is active
        if self.zoom_mode_active:
            self._setup_rectangle_selector()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    @staticmethod
    def _format_numeric_value(value: float) -> str:
        """Format numeric helper values for display."""
        if value == 0:
            return "0"
        abs_value = abs(value)
        if abs_value >= 1000 or abs_value < 0.01:
            return f"{value:.3g}"
        formatted = f"{value:.3f}"
        return formatted.rstrip("0").rstrip(".")

    def _format_prediction_hint(self, index: int) -> Optional[str]:
        """Format predicted helper data for display in title."""
        predicted = getattr(self.dataset, "predicted_prices", None)
        if predicted is None or index >= len(predicted):
            return None
        
        hint_value = predicted[index]
        if hint_value is None:
            return None
        
        try:
            if isinstance(hint_value, np.ndarray):
                if hint_value.size == 0:
                    return None
                if hint_value.ndim == 0:
                    hint_value = hint_value.item()
                else:
                    flat = hint_value.flatten()
                    if flat.size == 1:
                        hint_value = flat.item()
                    else:
                        formatted_values = []
                        for value in flat[:5]:
                            if isinstance(value, (np.integer, int)):
                                formatted_values.append(str(int(value)))
                            elif isinstance(value, (np.floating, float)):
                                if np.isnan(value):
                                    continue
                                formatted_values.append(self._format_numeric_value(float(value)))
                            else:
                                formatted_values.append(str(value))
                        if not formatted_values:
                            return None
                        if flat.size > 5:
                            formatted_values.append("…")
                        return f"Model hint: [{', '.join(formatted_values)}]"
            
            if isinstance(hint_value, (list, tuple)):
                if len(hint_value) == 0:
                    return None
                if len(hint_value) == 1:
                    hint_value = hint_value[0]
                else:
                    formatted_values = []
                    for value in hint_value[:5]:
                        if isinstance(value, (np.integer, int)):
                            formatted_values.append(str(int(value)))
                        elif isinstance(value, (np.floating, float)):
                            if np.isnan(value):
                                continue
                            formatted_values.append(self._format_numeric_value(float(value)))
                        else:
                            formatted_values.append(str(value))
                    if not formatted_values:
                        return None
                    if len(hint_value) > 5:
                        formatted_values.append("…")
                    return f"Model hint: [{', '.join(formatted_values)}]"
            
            if isinstance(hint_value, (np.integer, int)):
                int_value = int(hint_value)
                if 1 <= int_value <= self.num_classes:
                    return f"Model hint: Class {int_value}"
                return f"Model hint: {int_value}"
            
            if isinstance(hint_value, (np.floating, float)):
                float_value = float(hint_value)
                if np.isnan(float_value):
                    return None
                if float_value.is_integer():
                    int_value = int(float_value)
                    if 1 <= int_value <= self.num_classes:
                        return f"Model hint: Class {int_value}"
                formatted_value = self._format_numeric_value(float_value)
                return f"Model hint: {formatted_value}"
            
            return f"Model hint: {hint_value}"
        except Exception:
            return f"Model hint: {hint_value}"
    
    def handle_key_press(self, event):
        """Handle keyboard events."""
        # Class keys (1-9)
        if event.key in [str(i) for i in range(1, min(10, self.num_classes + 1))]:
            class_num = int(event.key)
            if class_num <= self.num_classes:
                self.set_classification(class_num)
        
        # Navigation
        elif event.key == "right" or event.key == " ":
            self.go_forward()
        elif event.key == "left":
            self.go_backward()
        
        # Similarity search
        elif event.key == "s":
            self.show_similar_series()
        
        # Save
        elif event.key == "ctrl+s":
            self.save_progress()
        
        # Export X/y
        elif event.key == "ctrl+e":
            self.export_xy_data()
        
        # Quit
        elif event.key == "q":
            plt.close()
    
    def handle_mouse_click(self, event):
        """Handle mouse clicks (reserved for zoom rectangle)."""
        if self.zoom_mode_active:
            # RectangleSelector handles the interaction
            return

    def handle_mouse_move(self, event):
        """Handle mouse movement to show cursor line."""
        if event.xdata is None or event.ydata is None:
            if self.cursor_line is not None:
                self.cursor_line.set_ydata([float('nan'), float('nan')])
                self.fig.canvas.draw_idle()
            return

        xlim = self.ax.get_xlim()
        y_pos = float(event.ydata)

        if self.cursor_line is None:
            self.cursor_line, = self.ax.plot(
                [xlim[0], xlim[1]],
                [y_pos, y_pos],
                color='red',
                linestyle='-',
                alpha=0.5,
                linewidth=1,
            )
        else:
            self.cursor_line.set_ydata([y_pos, y_pos])
            self.cursor_line.set_xdata([xlim[0], xlim[1]])

        self.fig.canvas.draw_idle()

    def go_forward(self):
        """Move to next time series with zoom reset."""
        if self.current_index < len(self.dataset) - 1:
            self.current_index += 1
            self.zoom_region = None
            self.zoom_mode_active = False
            self._remove_rectangle_selector()
            self.show_data()
            print(f"Moved forward to series {self.current_index + 1}/{len(self.dataset)}")
        else:
            print("Warning: This is the last series!")

    def go_backward(self):
        """Move to previous time series with zoom reset."""
        if self.current_index > 0:
            self.current_index -= 1
            self.zoom_region = None
            self.zoom_mode_active = False
            self._remove_rectangle_selector()
            self.show_data()
            print(f"Moved backward to series {self.current_index + 1}/{len(self.dataset)}")
        else:
            print("Warning: This is the first series!")
    
    def set_classification(self, class_num: int):
        """
        Set classification for current sample and move to next.
        
        Args:
            class_num: Classification number (1 to num_classes)
        """
        if self.current_index >= len(self.dataset):
            return
        
        # Save classification
        self.dataset.labels[self.current_index] = float(class_num)
        
        sample_name = self.dataset.get_sample_name(self.current_index)
        print(f" Sample {self.current_index + 1} ({sample_name}): Class {class_num}")

        # Handle periodic backups after label update
        self._maybe_handle_auto_backup()
        
        # Move to next
        self.current_index += 1
        
        if self.current_index < len(self.dataset):
            # Reset zoom when moving automatically
            self.zoom_region = None
            self.zoom_mode_active = False
            self._remove_rectangle_selector()
            self.show_data()
        else:
            print(" All samples classified!")
            self.save_progress()
            self.show_completion_message()
    
    def _get_processed_data(self, index: int) -> np.ndarray:
        """
        Get processed data for similarity search (same as shown in plot).
        
        Args:
            index: Sample index
            
        Returns:
            Processed y_values
        """
        classify_settings = self.settings.get("classify", {})
        y_values, _ = self.prepare_series_data(index, classify_settings)
        # No normalization or transformation needed - data is already preprocessed
        return y_values

    def show_similar_series(self):
        """Show similar labeled samples using similarity finder."""
        if self.similarity_finder is None:
            print("Warning: Similarity finder not configured")
            return
        
        if self.current_index >= len(self.dataset):
            return
        
        # Get processed data for current sample
        current_processed_data = self._get_processed_data(self.current_index)
        
        # Get processed data for all labeled samples
        labeled_processed_data = []
        for i in range(len(self.dataset)):
            if i != self.current_index and not np.isnan(self.dataset.labels[i]):
                processed_data = self._get_processed_data(i)
                labeled_processed_data.append((i, processed_data))
        
        print(f" Searching for similar samples to index {self.current_index}")
        print(f" Found {len(labeled_processed_data)} labeled samples for comparison")
        
        if len(labeled_processed_data) < 1:
            print("Warning: No classified samples available for comparison")
            return
        
        # Find similar samples using processed data
        similar_results = self.similarity_finder.find_similar_with_processed_data(
            current_processed_data, labeled_processed_data)
        
        # Visualize with color map
        if similar_results:
            print(f" Found {len(similar_results)} similar samples")
            similar_with_indices = [(idx, data, dist) for idx, data, dist in similar_results]
            self.similarity_finder.visualize_similar_with_processed_data(
                self.current_index, current_processed_data, similar_with_indices, 
                color_map=self.color_map, dataset=self.dataset, labeling_type='classify')
        else:
            print("Error: No similar samples found")
    
    def save_progress(self):
        """Save classification progress to labels.npy file."""
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

        classified_count = np.sum(~np.isnan(self.dataset.labels))
        print(f" Progress saved to {labels_file} ({classified_count}/{len(self.dataset)} classified)")
    
    def on_close(self, event):
        """Handle window close event."""
        print("\n Window closed. Saving progress...")
        self.save_progress()
    
    def show_completion_message(self):
        """Show completion message when all series are classified."""
        self.ax.clear()
        self.ax.text(0.5, 0.5, " All series classified!\n\nPress Q to exit",
                    ha='center', va='center', transform=self.ax.transAxes,
                    fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        self.ax.set_title("Classification Complete")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def find_first_unlabeled(self):
        """Find first unclassified sample based on labels."""
        if self.dataset.labels is None:
            self.current_index = 0
            print(f" Starting classification. Dataset has {len(self.dataset)} samples.")
            return
        
        for i in range(len(self.dataset)):
            if np.isnan(self.dataset.labels[i]):
                self.current_index = i
                classified_count = np.sum(~np.isnan(self.dataset.labels))
                print(f" Found {classified_count}/{len(self.dataset)} classified. "
                     f"Starting at sample {self.current_index + 1}.")
                return
        # All classified
        self.current_index = len(self.dataset)
        print(f" All {len(self.dataset)} samples are already classified.")

    def export_xy(self, include_ids: bool = False):
        """Export (X, y) exactly as shown during labeling.

        - X: list of processed values per sample
        - y: class index per sample
        - ids (optional): sample identifiers in the same order
        """
        X_list: List[List[float]] = []
        y_list: List[int] = []
        ids: List[str] = []

        classify_settings = self.settings.get("classify", {})

        for i in range(len(self.dataset)):
            label = self.dataset.labels[i]
            if np.isnan(label):
                continue  # Skip unlabeled samples

            y_values, x_values = self.prepare_series_data(i, classify_settings)
            
            # Data is already preprocessed, no normalization needed
            X_list.append(y_values.tolist())
            y_list.append(int(label))
            ids.append(self.dataset.get_sample_name(i))

        return X_list, y_list, (ids if include_ids else None)

    def save_xy_with_metadata(self, output_path: str) -> str:
        """Save X/y with a metadata snapshot of settings.

        The metadata includes:
        - dataset prefix and optional metadata
        - classify settings
        - timestamp and counts
        - class mapping summary
        """
        X, y, ids = self.export_xy(include_ids=True)

        metadata: Dict[str, Any] = {}
        import datetime
        metadata["created_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata["dataset"] = {
            "prefix": self.dataset.prefix,
            "description": self.dataset.get_description(),
            "samples_total": len(self.dataset),
        }

        # Settings snapshot
        classify_settings = self.settings.get("classify", {})
        metadata["settings"] = {
            "classify": {
                "num_classes": classify_settings.get("num_classes", self.num_classes),
            },
        }

        # Basic label stats
        if len(y) > 0:
            unique, counts = np.unique(y, return_counts=True)
            metadata["label_distribution"] = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
        else:
            metadata["label_distribution"] = {}
        metadata["num_samples"] = len(y)
        metadata["sequence_lengths"] = [len(x) for x in X] if X else []
        metadata["sequence_length_stats"] = {
            "min": min(metadata["sequence_lengths"]) if metadata["sequence_lengths"] else 0,
            "max": max(metadata["sequence_lengths"]) if metadata["sequence_lengths"] else 0,
            "avg": sum(metadata["sequence_lengths"]) / len(metadata["sequence_lengths"]) if metadata["sequence_lengths"] else 0
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        payload = {
            "X": X,  # X is already a list of lists
            "y": y,
            "ids": ids,
            "metadata": metadata,
        }

        # Save via fast JSON path for consistency
        import json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f" X/y exported to {output_path} — samples={metadata['num_samples']}, seq_len_range={metadata['sequence_length_stats']['min']}-{metadata['sequence_length_stats']['max']}")
        return output_path
    
    def _export_unlabeled_data(self, output_dir: str, timestamp: str) -> int:
        """Export unlabeled data (X only, no y labels).
        
        Args:
            output_dir: Directory to save the file
            timestamp: Timestamp for filename
            
        Returns:
            Number of unlabeled samples exported
        """
        X_list: List[List[float]] = []
        ids: List[str] = []
        
        classify_settings = self.settings.get("classify", {})
        
        # Collect unlabeled samples
        for i in range(len(self.dataset)):
            if not np.isnan(self.dataset.labels[i]):
                continue  # Skip labeled samples
            
            y_values, x_values = self.prepare_series_data(i, classify_settings)
            
            # Data is already preprocessed, no normalization needed
            X_list.append(y_values.tolist())
            ids.append(self.dataset.get_sample_name(i))
        
        if len(X_list) == 0:
            return 0
        
        # Create metadata for unlabeled data
        metadata: Dict[str, Any] = {}
        import datetime
        metadata["created_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata["dataset"] = {
            "prefix": self.dataset.prefix,
            "description": self.dataset.get_description(),
            "samples_total": len(self.dataset),
        }
        
        # Settings snapshot
        classify_settings = self.settings.get("classify", {})
        metadata["settings"] = {
            "classify": {
                "num_classes": classify_settings.get("num_classes", self.num_classes),
            },
        }
        
        metadata["num_samples"] = len(X_list)
        metadata["sequence_lengths"] = [len(x) for x in X_list] if X_list else []
        metadata["sequence_length_stats"] = {
            "min": min(metadata["sequence_lengths"]) if metadata["sequence_lengths"] else 0,
            "max": max(metadata["sequence_lengths"]) if metadata["sequence_lengths"] else 0,
            "avg": sum(metadata["sequence_lengths"]) / len(metadata["sequence_lengths"]) if metadata["sequence_lengths"] else 0
        }
        metadata["note"] = "Unlabeled data - no y labels available"
        
        # Save unlabeled data
        unlabeled_filename = f"classify_xy_unlabeled_{timestamp}.json"
        unlabeled_path = os.path.join(output_dir, unlabeled_filename)
        
        os.makedirs(os.path.dirname(unlabeled_path) or ".", exist_ok=True)
        payload = {
            "X": X_list,  # Only X data, no y labels
            "ids": ids,
            "metadata": metadata,
        }
        
        import json
        with open(unlabeled_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        
        print(f" Unlabeled X exported to {unlabeled_path} — samples={metadata['num_samples']}, seq_len_range={metadata['sequence_length_stats']['min']}-{metadata['sequence_length_stats']['max']}")
        return len(X_list)
    
    def export_xy_data(self):
        """Export X/y data with metadata (called by Export X/y button)."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        folder = self.settings.get("data", {}).get("folder", "")
        output_dir = folder if folder else "datasets/in_process_of_labeling"
        
        try:
            # Count labeled samples
            labeled_count = np.sum(~np.isnan(self.dataset.labels)) if self.dataset.labels is not None else 0
            
            # 1. Export labeled data (if any)
            if labeled_count > 0:
                labeled_filename = f"classify_xy_labeled_{timestamp}.json"
                labeled_path = os.path.join(output_dir, labeled_filename)
                self.save_xy_with_metadata(labeled_path)
                print(f" Labeled data exported: {labeled_count} samples")
            else:
                print("Warning: No labeled data to export")
            
            # 2. Export unlabeled data (if any)
            unlabeled_count = self._export_unlabeled_data(output_dir, timestamp)
            if unlabeled_count > 0:
                print(f" Unlabeled data exported: {unlabeled_count} samples")
            else:
                print(" No unlabeled data to export")
                
            total_exported = labeled_count + unlabeled_count
            print(f" Total exported: {total_exported} samples ({labeled_count} labeled + {unlabeled_count} unlabeled)")
            
        except Exception as e:
            print(f"Error: Export failed: {e}")
            import traceback
            traceback.print_exc()


