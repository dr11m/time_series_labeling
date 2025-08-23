import os
import tkinter as tk
from tkinter import ttk
from cfg.cfg_loader import cfg


def open_settings_window():
    def on_start():
        """Save selected settings and close the window."""
        cfg.NORMALIZE_VIEW = bool(normalize_var.get())
        cfg.SHOW_TIMESTAMPS_AS_DATES = bool(timestamps_var.get())
        cfg.SHOW_CURRENT_DATE = bool(current_date_var.get())
        cfg.CSV_DESCRIPTION = json_desc.get("1.0", tk.END).strip()  # Используем для JSON description
        selected = int(num_prices_var.get())
        cfg.NUM_PRICES = cfg.LabeledPriceAmount.ONE if selected == 1 else cfg.LabeledPriceAmount.TWO
        cfg.DATA_FILE = os.path.join("datasets", folder_var.get(), file_var.get())
        settings_win.destroy()

    # Create the main settings window
    settings_win = tk.Tk()
    settings_win.title("Settings and Instructions")
    settings_win.geometry("500x650")  # Set a larger window size
    settings_win.resizable(False, False)  # Prevent resizing

    # Center the window on the screen
    settings_win.update_idletasks()
    width = settings_win.winfo_width()
    height = settings_win.winfo_height()
    x = (settings_win.winfo_screenwidth() // 2) - (width // 2)
    y = (settings_win.winfo_screenheight() // 2) - (height // 2)
    settings_win.geometry(f"+{x}+{y}")

    # Define font styles
    font_main = ("Arial", 12)
    font_label = ("Arial", 11)

    # Instructions label
    instructions = tk.Label(
        settings_win,
        text=cfg.INSTRUCTIONS,
        justify="left",
        wraplength=480,
        font=("Arial", 10, "italic")
    )
    instructions.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="w")


    # Display settings section
    display_label = tk.Label(settings_win, text="Display Settings:", font=("Arial", 11, "bold"))
    display_label.grid(row=1, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")

    # Normalization checkbox
    normalize_var = tk.IntVar(value=1 if cfg.NORMALIZE_VIEW else 0)
    normalize_chk = tk.Checkbutton(settings_win, text="Display normalized view (0-1 scale)", variable=normalize_var, font=font_label)
    normalize_chk.grid(row=2, column=0, columnspan=2, padx=10, pady=2, sticky="w")

    # Timestamps as dates checkbox
    timestamps_var = tk.IntVar(value=1 if getattr(cfg, 'SHOW_TIMESTAMPS_AS_DATES', True) else 0)
    timestamps_chk = tk.Checkbutton(settings_win, text="Show timestamps as readable dates", variable=timestamps_var, font=font_label)
    timestamps_chk.grid(row=3, column=0, columnspan=2, padx=10, pady=2, sticky="w")

    # Current date checkbox
    current_date_var = tk.IntVar(value=1 if getattr(cfg, 'SHOW_CURRENT_DATE', True) else 0)
    current_date_chk = tk.Checkbutton(settings_win, text="Show current date on plot", variable=current_date_var, font=font_label)
    current_date_chk.grid(row=4, column=0, columnspan=2, padx=10, pady=2, sticky="w")

    # JSON description
    json_desc_label = tk.Label(settings_win, text="JSON Description (dataset description):", font=font_label)
    json_desc_label.grid(row=5, column=0, columnspan=2, padx=10, pady=(10, 0), sticky="w")

    json_desc = tk.Text(settings_win, height=3, width=55, font=font_main)
    json_desc.insert(tk.END, cfg.CSV_DESCRIPTION)  # Используем существующую переменную
    json_desc.grid(row=6, column=0, columnspan=2, padx=10, pady=5)

    # Number of prices dropdown
    num_prices_label = tk.Label(settings_win, text="Number of prices to label:", font=font_label)
    num_prices_label.grid(row=7, column=0, columnspan=2, padx=10, pady=(10, 0), sticky="w")

    num_prices_var = tk.StringVar(value=str(cfg.NUM_PRICES.value))
    num_prices_menu = ttk.Combobox(settings_win, textvariable=num_prices_var, values=["1", "2"], state="readonly", width=20)
    num_prices_menu.grid(row=8, column=0, columnspan=2, padx=10, pady=5)

    # Folder selection dropdown
    folder_label = tk.Label(settings_win, text="Select dataset folder:", font=font_label)
    folder_label.grid(row=9, column=0, columnspan=2, padx=10, pady=(10, 0), sticky="w")

    folder_options = ["in_process_of_labeling", "raw_data", "ready_to_work", "converted_json"]
    folder_var = tk.StringVar(value=folder_options[1])
    folder_menu = ttk.Combobox(settings_win, textvariable=folder_var, values=folder_options, state="readonly", width=25)
    folder_menu.grid(row=10, column=0, columnspan=2, padx=10, pady=5)

    # JSON file selection dropdown
    file_label = tk.Label(settings_win, text="Select JSON file:", font=font_label)
    file_label.grid(row=11, column=0, columnspan=2, padx=10, pady=(10, 0), sticky="w")

    file_var = tk.StringVar()
    file_menu = ttk.Combobox(settings_win, textvariable=file_var, state="readonly", width=25)
    file_menu.grid(row=12, column=0, columnspan=2, padx=10, pady=5)

    def update_file_list(*args):
        """Update the file list based on the selected folder."""
        folder = folder_var.get()
        path = os.path.join("datasets", folder)
        try:
            files = [f for f in os.listdir(path) if f.endswith(".json")]
        except FileNotFoundError:
            files = []
        file_menu["values"] = files
        file_var.set(files[0] if files else "")

    folder_var.trace("w", update_file_list)
    update_file_list()

    # Start button
    start_btn = tk.Button(settings_win, text="Save and Start Labeling", command=on_start, font=font_main, bg="#4CAF50", fg="white")
    start_btn.grid(row=13, column=0, columnspan=2, padx=10, pady=15)

    settings_win.mainloop()
