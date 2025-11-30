"""Tabbed settings window for configuring the application."""
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import numpy as np
from src.settings.settings_manager import SettingsManager
from src.settings.last_metadata_path import save_last_metadata_path, load_last_metadata_path
from src.dataset_loader import load_numpy_dataset


class SettingsWindow:
    """Tabbed settings window with organized configuration sections."""
    
    def __init__(self):
        """Initialize settings window."""
        # Load base settings from settings.json
        self.settings = SettingsManager.load()
        
        self.window = None
        self.notebook = None
        
        # Storage for widget variables
        self.vars = {}
    
    def _load_settings_from_metadata(self, metadata: dict) -> None:
        """Load settings from dataset metadata if available."""
        if not metadata or "labeling_settings" not in metadata:
            return
        
        labeling_settings = metadata["labeling_settings"]
        
        # Update settings with values from metadata (dataset has priority)
        if "labeling_type" in labeling_settings:
            self.settings["labeling_type"] = labeling_settings["labeling_type"]
        
        for section in ["predict", "classify", "anomaly_detection", "similarity"]:
            if section in labeling_settings:
                # Dataset settings override settings.json
                if section not in self.settings:
                    self.settings[section] = {}
                self.settings[section].update(labeling_settings[section])
        
        print(" Settings loaded from dataset metadata")
    
    def show(self):
        """Show the settings window and wait for user to close it."""
        self.window = tk.Tk()
        self.window.title("Time Series Labeling - Settings")
        self.window.geometry("950x800")
        
        # Center window
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f"+{x}+{y}")
        
        # Create main frame with scrollbar
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel and keyboard scrolling to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _on_key_scroll(event):
            if event.keysym == 'Up':
                canvas.yview_scroll(-1, "units")
            elif event.keysym == 'Down':
                canvas.yview_scroll(1, "units")
            elif event.keysym == 'Prior':  # Page Up
                canvas.yview_scroll(-1, "pages")
            elif event.keysym == 'Next':   # Page Down
                canvas.yview_scroll(1, "pages")
        
        # Bind events
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Up>", _on_key_scroll)
        canvas.bind_all("<Down>", _on_key_scroll)
        canvas.bind_all("<Prior>", _on_key_scroll)
        canvas.bind_all("<Next>", _on_key_scroll)
        
        # Make canvas focusable for keyboard events
        canvas.focus_set()
        
        # Create notebook (tabbed interface) inside scrollable frame
        self.notebook = ttk.Notebook(scrollable_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self._create_general_tab()
        self._create_labeling_tab()
        self._create_similarity_tab()
        
        # Load last metadata file path if exists
        last_metadata_path = load_last_metadata_path()
        if last_metadata_path and 'metadata_file' in self.vars:
            self.vars['metadata_file'].set(last_metadata_path)
            # Load settings from metadata
            self._on_metadata_file_change()
        
        # Buttons at bottom
        btn_frame = tk.Frame(self.window)
        btn_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(btn_frame, text="Start Labeling", command=self._save_and_start,
                 bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), 
                 height=2).pack(fill='x')
        
        self.window.mainloop()
        
        return self.settings
    
    def _create_general_tab(self):
        """Create General settings tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="General")
        
        # Labeling type selection
        frame = self._create_labeled_frame(tab, "Labeling Type", 0)
        
        self.vars['labeling_type'] = tk.StringVar(value=self.settings.get('labeling_type', 'predict'))
        
        tk.Radiobutton(frame, text="Predict - Label future price values", 
                      variable=self.vars['labeling_type'], value='predict',
                      font=("Arial", 11), command=self._on_labeling_type_change).pack(anchor='w', pady=5)
        
        tk.Radiobutton(frame, text="Classify - Categorize time series into classes",
                      variable=self.vars['labeling_type'], value='classify',
                      font=("Arial", 11), command=self._on_labeling_type_change).pack(anchor='w', pady=5)
        
        tk.Radiobutton(frame, text="Anomaly Detection - Mark anomaly points in time series",
                      variable=self.vars['labeling_type'], value='anomaly_detection',
                      font=("Arial", 11), command=self._on_labeling_type_change).pack(anchor='w', pady=5)
        
        # Data source selection
        frame = self._create_labeled_frame(tab, "Data Source", 1)
        
        # Required files
        tk.Label(frame, text="Required files:", font=("Arial", 10, "bold")).pack(anchor='w', pady=(5, 10))
        
        # Prices file
        self._create_file_browser(frame, "prices_file", "Prices file (prices.npy):", 
                                  filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
                                  required=True)
        
        # Timestamps file
        self._create_file_browser(frame, "timestamps_file", "Timestamps file (timestamps.npy):", 
                                  filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
                                  required=True)
        
        # Optional files
        tk.Label(frame, text="Optional files:", font=("Arial", 10, "bold")).pack(anchor='w', pady=(15, 10))
        
        # IDs file
        self._create_file_browser(frame, "ids_file", "IDs file (ids.npy):", 
                                  filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
                                  required=False)
        
        # Cluster IDs file
        self._create_file_browser(frame, "cluster_ids_file", "Cluster IDs file (cluster_ids.npy):", 
                                  filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
                                  required=False)
        
        # Labels file
        self._create_file_browser(frame, "labels_file", "Labels file (labels.npy):", 
                                  filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
                                  required=False)
        
        # Metadata file
        self._create_file_browser(frame, "metadata_file", "Metadata file (metadata.json):", 
                                  filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                                  required=False)
        
        # Predicted prices to help file
        self._create_file_browser(frame, "predicted_prices_to_help_file", "Predicted prices to help (predicted_prices_to_help.npy):", 
                                  filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
                                  required=False)
        
        # Bind file changes to load dataset info
        for file_key in ['prices_file', 'timestamps_file', 'ids_file', 'cluster_ids_file', 'labels_file']:
            if file_key in self.vars:
                self.vars[file_key].trace_add('write', lambda *args, key=file_key: self._on_file_path_change(key))
        
        # Metadata file has separate handler to load settings
        if 'metadata_file' in self.vars:
            self.vars['metadata_file'].trace_add('write', lambda *args: self._on_metadata_file_change())
        
        # Dataset info display
        info_frame = self._create_labeled_frame(tab, "Dataset Information", 2)
        
        # Dataset summary (read-only)
        tk.Label(info_frame, text="Dataset Summary:", font=("Arial", 10, "bold")).pack(anchor='w', pady=(5, 2))
        
        self.vars['dataset_info'] = tk.Text(info_frame, height=6, width=70, font=("Arial", 9), 
                                           state='disabled', bg='#f0f0f0')
        self.vars['dataset_info'].pack(fill='x', pady=(0, 10))
        
        # Metadata JSON editor (editable)
        tk.Label(info_frame, text="Metadata (JSON):", font=("Arial", 10, "bold")).pack(anchor='w', pady=(5, 2))
        tk.Label(info_frame, text="Edit metadata JSON structure. Will be saved to metadata.json in the same folder as prices.npy", 
                font=("Arial", 8), fg="gray").pack(anchor='w', pady=(0, 2))
        
        self.vars['metadata'] = tk.Text(info_frame, height=12, width=70, font=("Courier", 9))
        self.vars['metadata'].pack(fill='both', expand=True, pady=(0, 10))
        
        # Show initial loading state
        self._show_loading_indicator("Select required files (prices.npy and timestamps.npy) to load dataset info...")
    
    def _create_labeling_tab(self):
        """Create Labeling Settings tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Labeling Settings")
        
        # Predict settings
        self.vars['predict_frame'] = self._create_labeled_frame(tab, "Predict Settings", 0)
        frame = self.vars['predict_frame']
        
        # Number of prices
        row_frame = tk.Frame(frame)
        row_frame.pack(fill='x', pady=5)
        tk.Label(row_frame, text="Number of prices to label:", font=("Arial", 10)).pack(side='left')
        self.vars['num_prices'] = tk.IntVar(value=self.settings.get('predict', {}).get('num_prices', 2))
        ttk.Combobox(row_frame, textvariable=self.vars['num_prices'], 
                    values=[1, 2, 3], state="readonly", width=10).pack(side='left', padx=10)
        
        # Classify settings
        self.vars['classify_frame'] = self._create_labeled_frame(tab, "Classify Settings", 1)
        frame = self.vars['classify_frame']
        
        # Number of classes
        row_frame = tk.Frame(frame)
        row_frame.pack(fill='x', pady=5)
        tk.Label(row_frame, text="Number of classes:", font=("Arial", 10)).pack(side='left')
        self.vars['num_classes'] = tk.IntVar(value=self.settings.get('classify', {}).get('num_classes', 5))
        tk.Spinbox(row_frame, from_=2, to=10, textvariable=self.vars['num_classes'],
                  width=10).pack(side='left', padx=10)
        
        # Anomaly detection settings (empty frame - no settings needed)
        self.vars['anomaly_detection_frame'] = self._create_labeled_frame(tab, "Anomaly Detection Settings", 2)
        frame = self.vars['anomaly_detection_frame']
        tk.Label(frame, text="No settings required for anomaly detection mode.", 
                font=("Arial", 10), fg="gray").pack(anchor='w', pady=10)
        tk.Label(frame, text="Click on the plot to mark/unmark anomaly points.", 
                font=("Arial", 9), fg="gray").pack(anchor='w', pady=5)
        
        # Common visualization settings
        self.vars['visualization_frame'] = self._create_labeled_frame(tab, "Visualization Settings", 3)
        frame = self.vars['visualization_frame']
        
        # Right padding (X-axis)
        row_frame = tk.Frame(frame)
        row_frame.pack(fill='x', pady=5)
        tk.Label(row_frame, text="Right padding (X-axis):", font=("Arial", 10)).pack(side='left')
        self.vars['right_padding'] = tk.IntVar(value=self.settings.get('labeling', {}).get('right_padding', 0))
        tk.Spinbox(row_frame, from_=0, to=100, textvariable=self.vars['right_padding'],
                  width=10).pack(side='left', padx=10)
        tk.Label(row_frame, text="(Points to add on right side, 0 = disabled)", 
                font=("Arial", 9), fg="gray").pack(side='left', padx=10)
        
        # Y-axis padding (top/bottom)
        row_frame = tk.Frame(frame)
        row_frame.pack(fill='x', pady=5)
        tk.Label(row_frame, text="Y-axis padding (top/bottom):", font=("Arial", 10)).pack(side='left')
        self.vars['y_padding_percent'] = tk.IntVar(value=self.settings.get('labeling', {}).get('y_padding_percent', 15))
        tk.Spinbox(row_frame, from_=0, to=100, textvariable=self.vars['y_padding_percent'],
                  width=10).pack(side='left', padx=10)
        tk.Label(row_frame, text="(%)", font=("Arial", 9), fg="gray").pack(side='left', padx=5)
        tk.Label(row_frame, text="(Percentage of data range to add above and below)", 
                font=("Arial", 9), fg="gray").pack(side='left', padx=10)
        
        # Number of points from end to highlight
        row_frame = tk.Frame(frame)
        row_frame.pack(fill='x', pady=5)
        tk.Label(row_frame, text="Highlight points from end:", font=("Arial", 10)).pack(side='left')
        self.vars['num_points_from_end'] = tk.IntVar(value=self.settings.get('labeling', {}).get('num_points_from_end', 0))
        tk.Spinbox(row_frame, from_=0, to=1000, textvariable=self.vars['num_points_from_end'],
                  width=10).pack(side='left', padx=10)
        tk.Label(row_frame, text="(Last N points highlighted in orange, 0 = disabled)", 
                font=("Arial", 9), fg="gray").pack(side='left', padx=10)
        
        # Update visibility based on current labeling type
        self._on_labeling_type_change()
    
    def _create_file_browser(self, parent, var_key, label_text, filetypes, required=True):
        """Create a file browser row with label, entry, and browse button."""
        row = tk.Frame(parent)
        row.pack(fill='x', pady=5)
        
        tk.Label(row, text=label_text, font=("Arial", 10)).pack(side='left', padx=(0, 10))
        
        saved_path = self.settings.get('data', {}).get(var_key, '')
        self.vars[var_key] = tk.StringVar(value=saved_path)
        
        entry = tk.Entry(row, textvariable=self.vars[var_key], width=60, font=("Arial", 10))
        entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        def browse_file():
            initial_dir = os.path.dirname(self.vars[var_key].get()) if self.vars[var_key].get() else os.getcwd()
            filename = filedialog.askopenfilename(
                title=f"Select {label_text}",
                initialdir=initial_dir,
                filetypes=filetypes
            )
            if filename:
                self.vars[var_key].set(filename)
                self._on_file_path_change(var_key)
        
        tk.Button(row, text="Browse...", command=browse_file, width=10).pack(side='left')
        
        if not required:
            tk.Label(row, text="(optional)", font=("Arial", 8), fg="gray").pack(side='left', padx=(5, 0))
    
    def _on_metadata_file_change(self):
        """Handle metadata file change - load settings from it if it contains labeling_settings."""
        metadata_file = self.vars.get('metadata_file', tk.StringVar()).get() if 'metadata_file' in self.vars else ""
        
        if not metadata_file or not os.path.exists(metadata_file):
            return
        
        try:
            import json
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Update metadata JSON editor
            metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False)
            self.vars['metadata'].delete("1.0", tk.END)
            self.vars['metadata'].insert(tk.END, metadata_json)
            
            # Check if metadata contains labeling_settings and load them
            if 'labeling_settings' in metadata:
                # Load settings from metadata
                self._load_settings_from_metadata(metadata)
                self._restore_data_settings_from_metadata(metadata)
                print(f" Settings loaded from metadata file: {metadata_file}")
            else:
                print(f" Metadata file loaded but no labeling_settings found: {metadata_file}")
                
        except Exception as e:
            print(f"Warning: Could not load metadata file: {e}")
    
    def _on_file_path_change(self, changed_key: str):
        """Handle file path change - try to load dataset info if required files are set."""
        prices_file = self.vars.get('prices_file', tk.StringVar()).get() if 'prices_file' in self.vars else ""
        timestamps_file = self.vars.get('timestamps_file', tk.StringVar()).get() if 'timestamps_file' in self.vars else ""
        
        # Only load if both required files are set
        if prices_file and timestamps_file and os.path.exists(prices_file) and os.path.exists(timestamps_file):
            self.window.after_idle(self._load_dataset_info_async)
    
    def _create_similarity_tab(self):
        """Create Similarity Search tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Similarity Search")
        
        frame = self._create_labeled_frame(tab, "Similarity Algorithm", 0)
        
        # Method selection
        row_frame = tk.Frame(frame)
        row_frame.pack(fill='x', pady=5)
        tk.Label(row_frame, text="Similarity method:", font=("Arial", 10)).pack(side='left')
        self.vars['similarity_method'] = tk.StringVar(
            value=self.settings.get('similarity', {}).get('method', 'soft_dtw'))
        ttk.Combobox(row_frame, textvariable=self.vars['similarity_method'],
                    values=['soft_dtw'], state="readonly", width=15).pack(side='left', padx=10)
        
        # Number of similar series
        row_frame = tk.Frame(frame)
        row_frame.pack(fill='x', pady=5)
        tk.Label(row_frame, text="Number of similar series to show:", font=("Arial", 10)).pack(side='left')
        self.vars['num_similar'] = tk.IntVar(value=self.settings.get('similarity', {}).get('num_similar', 4))
        tk.Spinbox(row_frame, from_=1, to=20, textvariable=self.vars['num_similar'],
                  width=10).pack(side='left', padx=10)
        
        # Gamma parameter
        row_frame = tk.Frame(frame)
        row_frame.pack(fill='x', pady=5)
        tk.Label(row_frame, text="Soft-DTW gamma parameter:", font=("Arial", 10)).pack(side='left')
        self.vars['gamma'] = tk.DoubleVar(value=self.settings.get('similarity', {}).get('gamma', 0.05))
        tk.Spinbox(row_frame, from_=0.01, to=1.0, increment=0.01, 
                  textvariable=self.vars['gamma'], width=10, format="%.2f").pack(side='left', padx=10)
        
        tk.Label(frame, text="Lower gamma = more strict alignment, Higher gamma = more flexible",
                font=("Arial", 9), fg="gray", justify='left').pack(anchor='w', padx=20, pady=(0, 10))
    
    def _create_labeled_frame(self, parent, label, row):
        """Create a labeled frame for grouping settings."""
        frame = tk.LabelFrame(parent, text=label, font=("Arial", 11, "bold"), padx=15, pady=15)
        frame.grid(row=row, column=0, padx=20, pady=15, sticky='ew')
        parent.grid_columnconfigure(0, weight=1)
        return frame
    
    def _on_labeling_type_change(self):
        """Handle labeling type change to show/hide relevant settings."""
        labeling_type = self.vars['labeling_type'].get()
        
        if 'predict_frame' in self.vars and 'classify_frame' in self.vars and 'anomaly_detection_frame' in self.vars:
            if labeling_type == 'predict':
                self.vars['predict_frame'].grid()
                self.vars['classify_frame'].grid_remove()
                self.vars['anomaly_detection_frame'].grid_remove()
            elif labeling_type == 'classify':
                self.vars['predict_frame'].grid_remove()
                self.vars['classify_frame'].grid()
                self.vars['anomaly_detection_frame'].grid_remove()
            else:  # anomaly_detection
                self.vars['predict_frame'].grid_remove()
                self.vars['classify_frame'].grid_remove()
                self.vars['anomaly_detection_frame'].grid()
        
        # Visualization settings are always visible
        if 'visualization_frame' in self.vars:
            self.vars['visualization_frame'].grid()
    
    
    def _show_loading_indicator(self, message="Loading dataset...", show_progress=False):
        """Show loading indicator in dataset info area."""
        self.vars['dataset_info'].config(state='normal')
        self.vars['dataset_info'].delete("1.0", tk.END)
        
        loading_text = f" {message}\n\nPlease wait while the dataset is being loaded and analyzed...\n\n"
        
        if show_progress:
            loading_text += " Processing large dataset - this may take a moment..."
        else:
            loading_text += " Tip: For large datasets, consider using smaller samples for faster loading."
        
        self.vars['dataset_info'].insert(tk.END, loading_text)
        self.vars['dataset_info'].config(state='disabled')
        
        # Force UI update
        self.window.update_idletasks()
    
    def _load_dataset_info_async(self):
        """Load dataset metadata without full dataset."""
        try:
            prices_file = self.vars.get('prices_file', tk.StringVar()).get() if 'prices_file' in self.vars else ""
            timestamps_file = self.vars.get('timestamps_file', tk.StringVar()).get() if 'timestamps_file' in self.vars else ""
            ids_file = self.vars.get('ids_file', tk.StringVar()).get() if 'ids_file' in self.vars else ""
            cluster_ids_file = self.vars.get('cluster_ids_file', tk.StringVar()).get() if 'cluster_ids_file' in self.vars else ""
            labels_file = self.vars.get('labels_file', tk.StringVar()).get() if 'labels_file' in self.vars else ""
            metadata_file = self.vars.get('metadata_file', tk.StringVar()).get() if 'metadata_file' in self.vars else ""
            predicted_prices_file = self.vars.get('predicted_prices_to_help_file', tk.StringVar()).get() if 'predicted_prices_to_help_file' in self.vars else ""
            
            if not prices_file or not timestamps_file:
                return
            
            self._show_loading_indicator("Loading dataset info...")
            self.window.update_idletasks()
            
            # Load dataset to get info
            dataset = load_numpy_dataset(
                prices_file=prices_file,
                timestamps_file=timestamps_file,
                ids_file=ids_file if ids_file else None,
                cluster_ids_file=cluster_ids_file if cluster_ids_file else None,
                labels_file=labels_file if labels_file else None,
                metadata_file=metadata_file if metadata_file else None,
                predicted_prices_file=predicted_prices_file if predicted_prices_file else None
            )
            
            # Build summary text
            summary_parts = [
                f"Samples: {len(dataset)}",
                f"Price shape: {dataset.prices.shape}",
            ]
            
            if dataset.cluster_ids is not None:
                summary_parts.append(f"Cluster IDs: Yes ({len(set(dataset.cluster_ids))} unique)")
            
            if dataset.ids is not None:
                summary_parts.append(f"IDs: Yes")
            
            if dataset.labels is not None:
                if dataset.labels.ndim == 1:
                    labeled_count = np.sum(~np.isnan(dataset.labels))
                else:
                    labeled_count = np.sum(~np.all(np.isnan(dataset.labels), axis=1))
                summary_parts.append(f"Labels: {labeled_count}/{len(dataset)} labeled")
            
            summary = "\n".join(summary_parts)
            
            # Update dataset info
            self.vars['dataset_info'].config(state='normal')
            self.vars['dataset_info'].delete("1.0", tk.END)
            self.vars['dataset_info'].insert(tk.END, summary)
            self.vars['dataset_info'].config(state='disabled')
            
            # Update metadata JSON editor
            import json
            if dataset.metadata:
                metadata_json = json.dumps(dataset.metadata, indent=2, ensure_ascii=False)
            else:
                metadata_json = "{}"
            
            self.vars['metadata'].delete("1.0", tk.END)
            self.vars['metadata'].insert(tk.END, metadata_json)
            
            # Don't auto-load settings from dataset metadata here
            # Settings are loaded only when user explicitly selects metadata_file
            # But we still show metadata in the editor
            
            # Force UI update
            self.window.update_idletasks()
            
        except FileNotFoundError as e:
            self._clear_dataset_info()
            self.vars['dataset_info'].config(state='normal')
            self.vars['dataset_info'].delete("1.0", tk.END)
            self.vars['dataset_info'].insert(tk.END, f"Error: Required files not found:\n{str(e)}")
            self.vars['dataset_info'].config(state='disabled')
        except Exception as e:
            self._clear_dataset_info()
            self.vars['dataset_info'].config(state='normal')
            self.vars['dataset_info'].delete("1.0", tk.END)
            self.vars['dataset_info'].insert(tk.END, f"Error: Error loading dataset: {str(e)}")
            self.vars['dataset_info'].config(state='disabled')
    
    def _clear_dataset_info(self):
        """Clear all dataset information fields."""
        # Clear dataset summary
        self.vars['dataset_info'].config(state='normal')
        self.vars['dataset_info'].delete("1.0", tk.END)
        self.vars['dataset_info'].config(state='disabled')
        
        # Clear metadata
        self.vars['metadata'].delete("1.0", tk.END)
        self.vars['metadata'].insert(tk.END, "{}")
    
    def _update_dataset_summary_lightweight(self, name, num_series, metadata):
        """Update dataset summary display with lightweight metadata only."""
        
        # Build summary text with available info
        summary = f"""Dataset: {name}
Total Series: {num_series}

Note: Full statistics will be calculated when starting labeling.
This preview shows only basic metadata to keep settings window responsive."""
        
        # Add metadata info if available
        if metadata and 'labeling_settings' in metadata:
            summary += "\n\nPrevious labeling settings detected in metadata."
        
        # Update display
        self.vars['dataset_info'].config(state='normal')
        self.vars['dataset_info'].delete("1.0", tk.END)
        self.vars['dataset_info'].insert(tk.END, summary)
        self.vars['dataset_info'].config(state='disabled')
    
    def _update_dataset_summary(self, dataset):
        """Update dataset summary display (deprecated - kept for compatibility)."""
        self._update_dataset_summary_lightweight(
            dataset.name, 
            len(dataset.series), 
            dataset.metadata
        )
    
    def _load_settings_from_metadata(self, metadata):
        """Load settings from dataset metadata if available."""
        if not metadata or 'labeling_settings' not in metadata:
            return
        
        try:
            saved_settings = metadata['labeling_settings']
            
            # Update UI with saved settings
            if 'labeling_type' in saved_settings:
                self.vars['labeling_type'].set(saved_settings['labeling_type'])
                self._on_labeling_type_change()
            
            if 'predict' in saved_settings:
                predict_settings = saved_settings['predict']
                if 'num_prices' in predict_settings:
                    self.vars['num_prices'].set(predict_settings['num_prices'])
            
            if 'classify' in saved_settings:
                classify_settings = saved_settings['classify']
                if 'num_classes' in classify_settings:
                    self.vars['num_classes'].set(classify_settings['num_classes'])
            
            if 'labeling' in saved_settings:
                labeling_settings = saved_settings['labeling']
                if 'right_padding' in labeling_settings:
                    self.vars['right_padding'].set(labeling_settings['right_padding'])
            
            if 'similarity' in saved_settings:
                similarity = saved_settings['similarity']
                if 'method' in similarity:
                    self.vars['similarity_method'].set(similarity['method'])
                if 'num_similar' in similarity:
                    self.vars['num_similar'].set(similarity['num_similar'])
                if 'gamma' in similarity:
                    self.vars['gamma'].set(similarity['gamma'])
            
            print(" Settings loaded from dataset metadata")
            
        except Exception as e:
            print(f"Warning: Could not load settings from metadata: {e}")
    
    def _restore_data_settings_from_metadata(self, metadata: dict):
        """Restore file paths and data settings from metadata."""
        try:
            if 'labeling_settings' in metadata and 'data' in metadata['labeling_settings']:
                data_settings = metadata['labeling_settings']['data']
                
                # Restore file paths if they exist in metadata
                if 'prices_file' in data_settings and data_settings['prices_file']:
                    if 'prices_file' in self.vars:
                        self.vars['prices_file'].set(data_settings['prices_file'])
                
                if 'timestamps_file' in data_settings and data_settings['timestamps_file']:
                    if 'timestamps_file' in self.vars:
                        self.vars['timestamps_file'].set(data_settings['timestamps_file'])
                
                if 'ids_file' in data_settings and data_settings.get('ids_file'):
                    if 'ids_file' in self.vars:
                        self.vars['ids_file'].set(data_settings['ids_file'])
                
                if 'cluster_ids_file' in data_settings and data_settings.get('cluster_ids_file'):
                    if 'cluster_ids_file' in self.vars:
                        self.vars['cluster_ids_file'].set(data_settings['cluster_ids_file'])
                
                if 'labels_file' in data_settings and data_settings.get('labels_file'):
                    if 'labels_file' in self.vars:
                        self.vars['labels_file'].set(data_settings['labels_file'])
                
                if 'metadata_file' in data_settings and data_settings.get('metadata_file'):
                    if 'metadata_file' in self.vars:
                        self.vars['metadata_file'].set(data_settings['metadata_file'])
                
                print(" File paths and data settings restored from metadata")
                
        except Exception as e:
            print(f"Warning: Could not restore data settings from metadata: {e}")
    
    def _save_settings_to_metadata(self, settings, dataset):
        """Save current settings to dataset metadata."""
        try:
            if dataset.metadata is None:
                dataset.metadata = {}
            
            # Save settings (excluding data section)
            settings_to_save = {
                'labeling_type': settings['labeling_type'],
                'predict': settings['predict'],
                'classify': settings['classify'],
                'data_preparation': settings['data_preparation'],
                'similarity': settings['similarity'],
                'saved_at': datetime.now().isoformat()
            }
            
            dataset.metadata['labeling_settings'] = settings_to_save
            print(" Settings saved to dataset metadata")
            
        except Exception as e:
            print(f"Warning: Could not save settings to metadata: {e}")
    
    def _save_and_start(self):
        """Save settings and close window."""
        try:
            # Validate inputs - get file paths
            prices_file = self.vars.get('prices_file', tk.StringVar()).get() if 'prices_file' in self.vars else ""
            timestamps_file = self.vars.get('timestamps_file', tk.StringVar()).get() if 'timestamps_file' in self.vars else ""
            ids_file = self.vars.get('ids_file', tk.StringVar()).get() if 'ids_file' in self.vars else ""
            cluster_ids_file = self.vars.get('cluster_ids_file', tk.StringVar()).get() if 'cluster_ids_file' in self.vars else ""
            labels_file = self.vars.get('labels_file', tk.StringVar()).get() if 'labels_file' in self.vars else ""
            metadata_file = self.vars.get('metadata_file', tk.StringVar()).get() if 'metadata_file' in self.vars else ""
            predicted_prices_file = self.vars.get('predicted_prices_to_help_file', tk.StringVar()).get() if 'predicted_prices_to_help_file' in self.vars else ""
            
            if not prices_file:
                messagebox.showerror("Error", "Please select prices.npy file")
                return
            
            if not timestamps_file:
                messagebox.showerror("Error", "Please select timestamps.npy file")
                return
            
            # Verify required files exist
            if not os.path.exists(prices_file):
                messagebox.showerror("Error", f"Required file not found: {prices_file}")
                return
            
            if not os.path.exists(timestamps_file):
                messagebox.showerror("Error", f"Required file not found: {timestamps_file}")
                return
            
            # Save metadata JSON to metadata.json in the same folder as prices.npy
            prices_dir = os.path.dirname(prices_file)
            metadata_output_path = os.path.join(prices_dir, "metadata.json")
            
            try:
                import json
                # Load existing metadata if it exists
                existing_metadata = {}
                if os.path.exists(metadata_output_path):
                    try:
                        with open(metadata_output_path, 'r', encoding='utf-8') as f:
                            existing_metadata = json.load(f)
                    except Exception:
                        pass  # If can't load, start fresh
                
                # Get metadata from editor or use existing
                metadata_text = self.vars['metadata'].get("1.0", tk.END).strip()
                if metadata_text:
                    try:
                        metadata_dict = json.loads(metadata_text)
                        # Merge with existing metadata to preserve other fields
                        existing_metadata.update(metadata_dict)
                    except json.JSONDecodeError:
                        # If invalid JSON, use existing metadata
                        metadata_dict = existing_metadata
                else:
                    metadata_dict = existing_metadata
                
                # Save all current settings to labeling_settings
                metadata_dict["labeling_settings"] = {
                    "labeling_type": self.vars['labeling_type'].get(),
                    "predict": {
                        "num_prices": self.vars['num_prices'].get()
                    },
                    "classify": {
                        "num_classes": self.vars['num_classes'].get()
                    },
                    "anomaly_detection": {},
                    "labeling": {
                        "right_padding": self.vars['right_padding'].get()
                    },
                    "similarity": {
                        "method": self.vars['similarity_method'].get(),
                        "num_similar": self.vars['num_similar'].get(),
                        "gamma": self.vars['gamma'].get()
                    },
                    "data": {
                        "prices_file": prices_file,
                        "timestamps_file": timestamps_file,
                        "ids_file": ids_file,
                        "cluster_ids_file": cluster_ids_file,
                        "labels_file": labels_file,
                        "metadata_file": metadata_output_path,
                        "predicted_prices_to_help_file": predicted_prices_file
                    },
                    "saved_at": datetime.now().isoformat()
                }
                
                # Save metadata.json
                os.makedirs(prices_dir, exist_ok=True)
                with open(metadata_output_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
                print(f" Metadata saved to {metadata_output_path}")
                
                # Save last metadata path
                save_last_metadata_path(metadata_output_path)
            except json.JSONDecodeError as e:
                messagebox.showerror("Error", f"Invalid JSON in metadata field: {str(e)}")
                return
            
            # Build settings dictionary
            self.settings = {
                "labeling_type": self.vars['labeling_type'].get(),
                "predict": {
                    "num_prices": self.vars['num_prices'].get()
                },
                "classify": {
                    "num_classes": self.vars['num_classes'].get()
                },
                "anomaly_detection": {},
                "labeling": {
                    "right_padding": self.vars['right_padding'].get(),
                    "y_padding_percent": self.vars['y_padding_percent'].get(),
                    "num_points_from_end": self.vars['num_points_from_end'].get()
                },
                "similarity": {
                    "method": self.vars['similarity_method'].get(),
                    "num_similar": self.vars['num_similar'].get(),
                    "gamma": self.vars['gamma'].get()
                },
                "data": {
                    "prices_file": prices_file,
                    "timestamps_file": timestamps_file,
                    "ids_file": ids_file,
                    "cluster_ids_file": cluster_ids_file,
                    "labels_file": labels_file,
                    "metadata_file": metadata_output_path,  # Always use the path in prices folder
                    "predicted_prices_to_help_file": predicted_prices_file
                }
            }
            
            # Save settings to settings.json
            SettingsManager.save(self.settings)
            
            print(" Settings prepared for labeling")
            
            self.window.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error preparing settings: {str(e)}")


def open_settings_window():
    """Open settings window and return configured settings."""
    window = SettingsWindow()
    return window.show()


