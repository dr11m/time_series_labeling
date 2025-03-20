from src.models import LabeledPriceAmount


# Data file paths
DATA_FILE = "datasets/raw_data/test_data_15_2608.csv"
OUTPUT_DIR = "datasets/in_process_of_labeling"

NORMALIZE_VIEW = True  # Display normalized view: scale the time series between 0 and 1

# CSV description: will be written as the first line in the CSV (prefixed with '#')
CSV_DESCRIPTION = "This CSV contains labeled price time series data."

# Количество цен для разметки (используем enum)
NUM_PRICES = LabeledPriceAmount.TWO

# Plot and app settings
TITLE_TEMPLATE = "Index: {idx} - {name}"
INSTRUCTIONS = (
    "Click on the plot to label prices:\n"
    " - If labeling 1 price, click to set 'labeled_price_1'.\n"
    " - If labeling 2 prices, click once for 'labeled_price_1' and click again for 'labeled_price_2'.\n"
    " - Press 'd' to find similar time series using DTW (Euclidean distance).\n\n"
    "Adjust settings below if needed."
)
