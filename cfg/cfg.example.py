from src.models import LabeledPriceAmount


# Data file paths
DATA_FILE = "datasets/raw_data/test_data_15_2608.csv"
OUTPUT_DIR = "datasets/in_process_of_labeling"

# Display settings
NORMALIZE_VIEW = True  # True: show normalized prices (0-1 scale), False: show raw prices
SHOW_TIMESTAMPS_AS_DATES = True  # True: show timestamps as readable dates, False: show as numbers
SHOW_CURRENT_DATE = True  # True: add current date as virtual point on the plot, False: hide it

# CSV description: will be written as the first line in the CSV (prefixed with '#')
CSV_DESCRIPTION = "This CSV contains labeled price time series data."

# Количество цен для разметки (используем enum)
NUM_PRICES = LabeledPriceAmount.TWO

# Настройки длины временных рядов
TARGET_SERIES_LENGTH = 10  # Желаемая длина временного ряда
SKIP_SHORTER_SERIES = True  # Пропускать ряды короче TARGET_SERIES_LENGTH
TAKE_LAST_N_VALUES = True  # Брать последние N значений если ряд длиннее TARGET_SERIES_LENGTH

# Plot and app settings
TITLE_TEMPLATE = "Index: {idx} - {name}"
INSTRUCTIONS = (
    "Click on the plot to label prices:\n"
    " - If labeling 1 price, click to set 'labeled_price_1'.\n"
    " - If labeling 2 prices, click once for 'labeled_price_1' and click again for 'labeled_price_2'.\n"
    " - Press 'd' to find similar time series using DTW (Euclidean distance).\n"
    " - Use arrow keys (← →) to navigate between series.\n"
    " - Press 'f' to filter series by length.\n\n"
    "Display settings:\n"
    " - Normalized (0-1 scale) or Raw prices\n"
    " - Show timestamps as dates or numbers\n"
    " - Show/hide current date on plot\n"
    "Adjust settings below if needed."
)
