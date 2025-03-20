import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import cfg.cfg as cfg
from src.models import SaleRow
from pydantic import ValidationError


def validate_df(df):
    try:
        validated_rows = [SaleRow(**row) for row in df.to_dict(orient='records')]
        print("All rows are valid")
        return validated_rows
    except ValidationError as e:
        print(f"Some rows are invalid: {e}")
        return None


def normalize_array(arr):
    arr = np.array(arr, dtype=float)
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val == 0:
        return arr
    return (arr - min_val) / (max_val - min_val)

def denormalize_value(norm_val, original_arr):
    arr = np.array(original_arr, dtype=float)
    min_val = arr.min()
    max_val = arr.max()
    return norm_val * (max_val - min_val) + min_val

def go_forward(main_app):
    if main_app.idx < len(main_app.df) - 1:
        main_app.idx += 1
        main_app.click_stage = 1
        main_app.show_data()
        print("Moved forward.")

def go_backward(main_app):
    if main_app.idx > 0:
        main_app.idx -= 1
        main_app.click_stage = 1
        main_app.show_data()
        print("Moved backward.")

def find_and_plot_distances(self):
    if self.idx >= len(self.df):
        return
    row = self.df.iloc[self.idx]
    val_ts = np.array([row[f'price_{i}'] for i in range(1, 11)], dtype=float)
    train_set = self.df[self.df['labeled_price_1'] != -1][[f'price_{i}' for i in range(1, 11)]].values.tolist()
    if len(train_set) < 10:
        print("Not enough labeled data!")
        return
    similar_rows = sorted(
        [(i, fastdtw(val_ts, ts, dist=lambda x, y: abs(x - y))[0]) for i, ts in enumerate(train_set)],
        key=lambda x: x[1]
    )[:4]
    
    fig = plt.figure(num="Similar Patterns", figsize=(12, 6))
    fig.suptitle(f"Similar patterns for index {self.idx}", fontsize=12)
    axes = fig.subplots(2, 2)
    
    for i, (index, distance) in enumerate(similar_rows):
        ax = axes[i // 2, i % 2]
        prices = train_set[index]
        if cfg.NORMALIZE_VIEW:
            prices_norm = normalize_array(prices)
            ax.plot(range(1, 11), prices_norm)
            predicted = normalize_array([self.df.iloc[index]['labeled_price_1']])[0]
            ax.axhline(y=predicted, color='green', linestyle='--')
        else:
            ax.plot(range(1, 11), prices)
            ax.axhline(y=self.df.iloc[index]['labeled_price_1'], color='green', linestyle='--')
        ax.set_title(f"Similar {i+1} (D: {distance:.2f})")
        ax.grid(True)
    
    plt.tight_layout()
    plt.show(block=False)  # don't block the main thread
