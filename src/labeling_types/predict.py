"""Prediction labeling type for time series."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor, Button, RectangleSelector
from typing import Dict, Any, List, Optional, Tuple
from src.labeling_types.base import BaseLabelingType
from src.dataset_loader import NumpyDataset
from src.similarity.base import BaseSimilarityFinder


class PredictLabeling(BaseLabelingType):
    """Labeling type for predicting future prices."""
    
    def __init__(self, dataset: NumpyDataset, settings: Dict[str, Any],
                 similarity_finder: BaseSimilarityFinder = None):
        """
        Initialize predict labeling.
        
        Args:
            dataset: NumPy dataset
            settings: Application settings
            similarity_finder: Similarity search instance
        """
        super().__init__(dataset, settings, similarity_finder)
        
        # Predict-specific settings
        predict_settings = settings.get("predict", {})
        self.num_prices = predict_settings.get("num_prices", 2)
        
        # Initialize labels array if not exists
        if self.dataset.labels is None:
            # Shape: (N, num_prices) - each row contains labels for one sample
            self.dataset.labels = np.full((len(self.dataset), self.num_prices), np.nan, dtype=np.float64)
        else:
            # Check if labels array needs to be resized for current num_prices
            if self.dataset.labels.shape[1] < self.num_prices:
                # Expand labels array to accommodate more prices
                old_shape = self.dataset.labels.shape
                new_labels = np.full((old_shape[0], self.num_prices), np.nan, dtype=np.float64)
                # Copy existing labels
                new_labels[:, :old_shape[1]] = self.dataset.labels
                self.dataset.labels = new_labels
                print(f" Expanded labels array from {old_shape[1]} to {self.num_prices} prices")
            elif self.dataset.labels.shape[1] > self.num_prices:
                # Shrink labels array if num_prices was decreased
                self.dataset.labels = self.dataset.labels[:, :self.num_prices]
                print(f" Shrunk labels array from {self.dataset.labels.shape[1]} to {self.num_prices} prices")
        
        # Prepare backup tracking state
        self._initialize_backup_tracking()

        # Click tracking for multi-price labeling
        self.click_count = 0
        
        # UI buttons storage
        self.predict_buttons = {}
        self.control_buttons = {}
        
        # Cursor line for price targeting
        self.cursor_line = None
        
        # Zoom region selection (for removing outliers)
        self.zoom_region: Optional[Tuple[float, float, float, float]] = None  # (x_min, x_max, y_min, y_max)
        self.rectangle_selector: Optional[RectangleSelector] = None
        self.zoom_mode_active = False
        
        # Find first unlabeled sample
        self.find_first_unlabeled()

    def find_first_unlabeled(self) -> None:
        """Find and move to first unlabeled sample for prediction."""
        if self.dataset.labels is None:
            self.current_index = 0
            print(f" Starting labeling. Dataset has {len(self.dataset)} samples.")
            return
        
        for i in range(len(self.dataset)):
            if not self._has_required_price_labels(i):
                self.current_index = i
                labeled_count = sum(1 for j in range(len(self.dataset)) if self._has_required_price_labels(j))
                print(f" Found {labeled_count}/{len(self.dataset)} samples with price labels. "
                     f"Starting at sample {self.current_index + 1}.")
                return
        
        # All samples have required price labels
        self.current_index = len(self.dataset)
        print(f" All {len(self.dataset)} samples have required price labels.")

    def _has_required_price_labels(self, index: int) -> bool:
        """Check if sample has the required price labels for current prediction mode."""
        if self.dataset.labels is None:
            return False
        
        labels = self.dataset.labels[index]
        # Check if all required prices are labeled (not NaN)
        valid_labels = ~np.isnan(labels[:self.num_prices])
        
        if self.num_prices == 1:
            return valid_labels[0] if len(valid_labels) > 0 else False
        else:
            return np.all(valid_labels)

    def _get_processed_data(self, index: int) -> np.ndarray:
        """Get processed data for similarity search (same as shown in plot)."""
        predict_settings = self.settings.get("predict", {})
        y_values, _ = self.prepare_series_data(index, predict_settings)
        # No normalization or transformation needed - data is already preprocessed
        return y_values
    
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
        """Create prediction and control buttons."""
        button_width = 0.12
        button_height = 0.04
        start_x = 0.1
        y_predict = 0.08  # First row
        y_control = 0.02  # Second row
        
        # First row: Reset and Zoom buttons
        ax_reset = plt.axes([start_x, y_predict, button_width, button_height])
        btn_reset = Button(ax_reset, 'Reset Labels', color='#DC143C')
        btn_reset.on_clicked(lambda event: self.reset_labels())
        self.predict_buttons['Reset Labels'] = btn_reset
        
        # Zoom region button
        ax_zoom = plt.axes([start_x + button_width + 0.02, y_predict, button_width, button_height])
        btn_zoom = Button(ax_zoom, 'Zoom Region', color='#9C27B0')  # Purple
        btn_zoom.on_clicked(lambda event: self.toggle_zoom_mode())
        self.predict_buttons['Zoom Region'] = btn_zoom
        
        # Reset zoom button
        ax_reset_zoom = plt.axes([start_x + 2 * (button_width + 0.02), y_predict, button_width, button_height])
        btn_reset_zoom = Button(ax_reset_zoom, 'Reset Zoom', color='#FF9800')  # Orange
        btn_reset_zoom.on_clicked(lambda event: self.reset_zoom())
        self.predict_buttons['Reset Zoom'] = btn_reset_zoom
        
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
        
        self.dataset.labels[self.current_index] = np.nan
        self.click_count = 0
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
        # Remove existing selector if any
        self._remove_rectangle_selector()
        
        # Create new RectangleSelector
        # Use interactive=True to allow dragging
        # Use 'data' coordinates for more intuitive selection
        self.rectangle_selector = RectangleSelector(
            self.ax,
            self._on_rectangle_select,
            useblit=True,
            button=[1],  # Only left mouse button
            minspanx=0,  # Minimum width in data units (0 = no minimum)
            minspany=0,  # Minimum height in data units (0 = no minimum)
            spancoords='data',  # Use data coordinates
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
        # Get coordinates
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return
        
        # Ensure x1 < x2 and y1 < y2
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)
        
        # Set zoom region
        self.zoom_region = (x_min, x_max, y_min, y_max)
        
        # Auto-disable zoom mode after selection
        self.zoom_mode_active = False
        self._remove_rectangle_selector()
        
        # Apply zoom and redraw
        self.show_data()
        print(f" Zoom region set: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}]")
        print(" Zoom mode auto-disabled - you can now label prices")
    
    def show_data(self):
        """Display current time series."""
        if self.current_index >= len(self.dataset):
            self.show_completion_message()
            return
        
        # Reset click count if sample already has required labels
        if self._has_required_price_labels(self.current_index):
            self.click_count = 0
        
        # Prepare data according to settings (returns y_values, x_values as numpy arrays)
        predict_settings = self.settings.get("predict", {})
        y_values, raw_x_values = self.prepare_series_data(self.current_index, predict_settings)
        x_values, is_datetime_axis = self._convert_x_axis_for_plot(raw_x_values)
        
        # Clear and plot
        self.ax.clear()
        self.cursor_line = None  # Reset cursor line
        
        # Get X limits for the main data
        # Ensure x_values is 1D and extract scalar values
        x_values_flat = np.asarray(x_values).flatten()
        y_values_flat = np.asarray(y_values).flatten()
        x_min_data = float(x_values_flat[0]) if len(x_values_flat) > 0 else 0
        x_max_data = float(x_values_flat[-1]) if len(x_values_flat) > 0 else 1
        
        # Get predicted prices for this sample (if available)
        predicted_price = None
        if self.dataset.predicted_prices is not None and self.current_index < len(self.dataset.predicted_prices):
            pred = self.dataset.predicted_prices[self.current_index]
            # Handle both scalar and array cases
            if np.isscalar(pred):
                predicted_price = float(pred)
            elif isinstance(pred, (np.ndarray, list)) and len(pred) > 0:
                predicted_price = float(pred[0])  # Take first predicted price if array
        
        # Get right padding setting
        labeling_settings = self.settings.get("labeling", {})
        right_padding = labeling_settings.get("right_padding", 0)
        num_points_from_end = labeling_settings.get("num_points_from_end", 0)
        
        # Separate last N points if highlighting is enabled
        if num_points_from_end > 0 and len(x_values_flat) > 0:
            # Calculate how many points to exclude from main plot
            n_points = min(num_points_from_end, len(x_values_flat))
            if n_points > 0 and n_points < len(x_values_flat):
                # Split data: main plot (without last N points) and highlighted points
                main_x = x_values_flat[:-n_points]
                main_y = y_values_flat[:-n_points]
                end_x = x_values_flat[-n_points:]
                end_y = y_values_flat[-n_points:]
                
                # Plot main data (without last N points)
                if len(main_x) >= 2:
                    self.ax.plot(main_x, main_y, marker='o', color='blue')
                else:
                    self.ax.scatter(main_x, main_y, color='blue')
                
                # Plot last N points in orange
                self.ax.scatter(end_x, end_y, color='orange', s=100, zorder=5, 
                              edgecolors='darkorange', linewidths=1.5, alpha=0.8)
            else:
                # If n_points >= len, show all points in orange only
                self.ax.scatter(x_values_flat, y_values_flat, color='orange', s=100, zorder=5, 
                              edgecolors='darkorange', linewidths=1.5, alpha=0.8)
        else:
            # Plot all data normally (blue)
            if len(x_values_flat) >= 2:
                self.ax.plot(x_values_flat, y_values_flat, marker='o', color='blue')
            else:
                self.ax.scatter(x_values_flat, y_values_flat, color='blue')
        
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
        
        # Show labeled values as full-width horizontal lines with different colors
        labels = self.dataset.labels[self.current_index]
        # Color map for different prices: Green, Teal, Red
        price_colors = ['#4CAF50', '#4ECDC4', '#FF6B6B']
        
        # Set Y-axis limits with margin first (needed for text positioning)
        all_y_values = list(y_values)
        for label_value in labels[:self.num_prices]:
            if not np.isnan(label_value):
                all_y_values.append(label_value)
        # Include predicted price in Y-axis limits if available
        if predicted_price is not None:
            all_y_values.append(predicted_price)
        
        if len(all_y_values) > 0:
            y_min = min(all_y_values)
            y_max = max(all_y_values)
            # Get Y-axis padding from settings
            labeling_settings = self.settings.get("labeling", {})
            margin_percent = labeling_settings.get("y_padding_percent", 15)
            margin = (y_max - y_min) * (margin_percent / 100.0) if y_max != y_min else abs(y_min) * (margin_percent / 100.0) if y_min != 0 else 1.0
            # Ensure margin is at least a small value if all values are the same
            if margin == 0:
                margin = max(abs(y_min), abs(y_max), 1.0) * 0.1
            self.ax.set_ylim(y_min - margin, y_max + margin)
        
        # Set X limits (padding already applied if enabled)
        # Apply zoom region if set, otherwise use default
        if self.zoom_region is not None:
            # Use zoom region boundaries
            self.ax.set_xlim(self.zoom_region[0], self.zoom_region[1])
            self.ax.set_ylim(self.zoom_region[2], self.zoom_region[3])
        else:
            # Default behavior
            if right_padding == 0:
                # No padding: show full data with small margins on sides
                x_range = x_max_data - x_min_data
                x_margin = x_range * 0.05 if x_range > 0 else 1.0  # 5% margin on each side
                self.ax.set_xlim(x_min_data - x_margin, x_max_data + x_margin)
            # ylim already set above
        
        # Draw predicted price at last point position (if available)
        if predicted_price is not None and len(x_values_flat) > 0:
            # Use the last point's X coordinate from the original data
            last_x = x_max_data
            # Plot predicted price in pink/magenta color
            self.ax.scatter([last_x], [predicted_price], color='magenta', s=250, 
                          zorder=6, edgecolors='darkmagenta', linewidths=2, 
                          alpha=0.9, marker='*', label=f'Predicted: {predicted_price:.3f}')
        
        # Now draw price lines and labels (after limits are set)
        if not np.all(np.isnan(labels[:self.num_prices])):
            xlim = self.ax.get_xlim()
            x_text_pos = xlim[1] - (xlim[1] - xlim[0]) * 0.02  # Position text near right edge
            
            for i, label_value in enumerate(labels[:self.num_prices]):
                if not np.isnan(label_value):
                    color = price_colors[i % len(price_colors)]
                    # Draw labeled price line across entire width
                    self.ax.axhline(y=label_value, color=color, linestyle='--',
                                  alpha=0.7, linewidth=2.5, 
                                  label=f'Price {i+1}: {label_value:.3f}')
                    
                    # Add text label with price number
                    self.ax.text(x_text_pos, label_value, f'{i+1}', 
                               color=color, fontsize=12, fontweight='bold',
                               horizontalalignment='right',
                               verticalalignment='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                       edgecolor=color, linewidth=1.5, alpha=0.8))
            
            self.ax.legend(loc='upper left', fontsize=9)
        
        # Cursor for precise clicking
        if not hasattr(self, 'cursor') or self.cursor is None:
            self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
        
        # Labels and title
        self.ax.set_xlabel("Date" if is_datetime_axis else "Timestamp")
        self.ax.set_ylabel("Value")
        sample_name = self.dataset.get_sample_name(self.current_index)
        title = f"Index: {self.current_index + 1}/{len(self.dataset)} - {sample_name}"
        self.ax.set_title(title)
        self.ax.grid(True)
        
        # Recreate RectangleSelector if zoom mode is active (after plot is drawn)
        if self.zoom_mode_active:
            self._setup_rectangle_selector()
        
        # Apply axis formatting after plotting
        self._format_x_axis(is_datetime_axis)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def go_forward(self):
        """Move to next time series."""
        if self.current_index < len(self.dataset) - 1:
            self.current_index += 1
            self.click_count = 0  # Reset click count when navigating
            # Reset zoom when navigating to next sample
            self.zoom_region = None
            self.zoom_mode_active = False
            self._remove_rectangle_selector()
            self.show_data()
            print(f"Moved forward to series {self.current_index + 1}/{len(self.dataset)}")
        else:
            print("Warning: This is the last series!")
    
    def go_backward(self):
        """Move to previous time series."""
        if self.current_index > 0:
            self.current_index -= 1
            self.click_count = 0  # Reset click count when navigating
            # Reset zoom when navigating to previous sample
            self.zoom_region = None
            self.zoom_mode_active = False
            self._remove_rectangle_selector()
            self.show_data()
            print(f"Moved backward to series {self.current_index + 1}/{len(self.dataset)}")
        else:
            print("Warning: This is the first series!")
    
    def handle_mouse_move(self, event):
        """Handle mouse movement to show cursor line."""
        if event.xdata is None or event.ydata is None:
            # Hide cursor line when mouse leaves the plot area
            if self.cursor_line is not None:
                self.cursor_line.set_ydata([float('nan'), float('nan')])
                self.fig.canvas.draw_idle()
            return
        
        # Get current x limits for the line
        xlim = self.ax.get_xlim()
        # Ensure y coordinate is exactly what the mouse points to
        y_pos = float(event.ydata)
        
        # Update cursor line position
        if self.cursor_line is None:
            # Create cursor line on first movement
            self.cursor_line, = self.ax.plot([xlim[0], xlim[1]], [y_pos, y_pos], 
                                           color='red', linestyle='-', alpha=0.5, linewidth=1,
                                           picker=5, transform=self.ax.transData)  # Explicitly use data transform
        else:
            # Update existing cursor line
            self.cursor_line.set_ydata([y_pos, y_pos])
            self.cursor_line.set_xdata([xlim[0], xlim[1]])
        
        self.fig.canvas.draw_idle()
    
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
        """Handle mouse clicks for price labeling."""
        # Only process clicks on the main plot axes, not on buttons or other UI elements
        if event.inaxes != self.ax:
            return
        
        # Only process left mouse button clicks
        if event.button != 1:
            return
        
        # Don't process clicks if zoom mode is active (RectangleSelector handles it)
        if self.zoom_mode_active:
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        # Check current state: how many prices are already labeled?
        labels = self.dataset.labels[self.current_index]
        
        # Ensure labels array has correct size for current num_prices setting
        if len(labels) < self.num_prices:
            # Expand labels array if needed (e.g., if num_prices was increased)
            # Create new array with correct size
            new_labels = np.full(self.num_prices, np.nan, dtype=np.float64)
            # Copy existing labels
            existing_size = min(len(labels), self.num_prices)
            new_labels[:existing_size] = labels[:existing_size]
            # Update the labels array for this sample
            self.dataset.labels[self.current_index] = new_labels
            labels = new_labels
        
        labeled_count = sum(1 for i in range(self.num_prices) if not np.isnan(labels[i]))
        
        # Determine which price to set (0-indexed)
        if labeled_count < self.num_prices:
            # Set the next unlabeled price
            price_index = labeled_count
            self.dataset.labels[self.current_index, price_index] = float(y)
            print(f" Labeled price_{price_index + 1} = {y:.3f}")
            self._maybe_handle_auto_backup()
            
            # Check if all prices are now labeled
            if labeled_count + 1 == self.num_prices:
                # All prices labeled, move to next sample
                self.click_count = 0
                self.current_index += 1
                # Reset zoom when automatically moving to next sample
                self.zoom_region = None
                self.zoom_mode_active = False
                self._remove_rectangle_selector()
                print(f" All {self.num_prices} prices labeled. Moving to next sample.")
            else:
                # Still need more prices
                print(f"Click for price_{price_index + 2}.")
            
            self.show_data()
        else:
            # All prices already labeled, but user clicked anyway
            print(f"Warning: All {self.num_prices} prices already labeled. Use Reset Labels button to change.")
    
    def show_similar_series(self):
        """Show similar labeled series using similarity finder."""
        if self.similarity_finder is None:
            print("Warning: Similarity finder not configured")
            return
        
        if self.current_index >= len(self.dataset):
            return
        
        # Prepare processed data (consistent with current view settings)
        # Query
        q_y = self._get_processed_data(self.current_index)
        
        # Labeled set - use only other samples that have price labels
        labeled_processed = []
        for i in range(len(self.dataset)):
            if i != self.current_index and self._has_required_price_labels(i):
                ly = self._get_processed_data(i)
                # Keep only same-length series
                if len(ly) == len(q_y) and len(ly) > 0:
                    labeled_processed.append((i, ly))
        
        if not labeled_processed:
            print("Warning: No labeled samples with matching length to compare")
            return
        
        # Find similar using processed-data API
        similar_results = self.similarity_finder.find_similar_with_processed_data(q_y, labeled_processed)
        
        # Visualize results
        if similar_results:
            # Convert results to format expected by similarity finder
            # Similarity finder expects (series, processed_data, distance) but we have (index, processed_data, distance)
            # We'll need to update similarity finder to work with indices, but for now create a wrapper
            similar_with_indices = [(idx, data, dist) for idx, data, dist in similar_results]
            self.similarity_finder.visualize_similar_with_processed_data(
                self.current_index, q_y, similar_with_indices,
                dataset=self.dataset, labeling_type='predict', num_prices=self.num_prices)
    
    def save_progress(self):
        """Save labeling progress to labels.npy file."""
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

        labeled_count = sum(1 for i in range(len(self.dataset)) if self._has_required_price_labels(i))
        print(f" Progress saved to {labels_file} ({labeled_count}/{len(self.dataset)} labeled)")
    
    def on_close(self, event):
        """Handle window close event."""
        print("\n Window closed. Saving progress...")
        self.save_progress()
    
    def show_completion_message(self):
        """Show completion message when all series are labeled."""
        self.ax.clear()
        self.ax.text(0.5, 0.5, " All series labeled!\n\nPress Q to exit",
                    ha='center', va='center', transform=self.ax.transAxes,
                    fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        self.ax.set_title("Labeling Complete")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


