import datetime
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import warnings
from src.settings_window import open_settings_window
from src.controls_window import start_control_window
import src.app_utils as app_utils
import cfg.cfg as cfg


warnings.filterwarnings("ignore", category=FutureWarning)


class MainApp:
    def __init__(self, df):
        self.df = df
        self.idx = 0
        self.click_stage = 1
        self.first_click_y = None
        self.fig = plt.gcf()
        self.fig.canvas.mpl_connect('key_press_event', self.handle_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.handle_mouse_click)
        self.__post_init__()

    def __post_init__(self):
        # init columns if there are none
        for col in ['labeled_price_1', 'labeled_price_2']:
            if col not in self.df.columns:
                self.df[col] = -1

        while self.idx < len(df) and self.df.iloc[self.idx]['labeled_price_1'] != -1:
            self.idx += 1

    def show_data(self):
        if self.idx >= len(self.df):
            return
        row = self.df.iloc[self.idx]
        prices = [row[f'price_{i}'] for i in range(1, 11)]
        display_prices = app_utils.normalize_array(prices) if cfg.NORMALIZE_VIEW else prices

        plt.clf()
        plt.plot(range(1, 11), display_prices, marker='o')
        plt.scatter(range(1, 11), display_prices, color='green')
        if self.first_click_y is not None:
            plt.axhline(y=self.first_click_y, color='orange', linestyle='--')

        cursor = Cursor(plt.gca(), useblit=True, color='red', linewidth=1)
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title(cfg.TITLE_TEMPLATE.format(idx=self.idx, name=row.get('name', 'Unknown')))
        plt.grid(True)
        plt.show()


    def handle_key_press(self, event):
        if event.key == "right":
            self.go_forward()
        elif event.key == "left":
            self.go_backward()
        elif event.key == "d":
            self.find_and_plot_distances()

    def handle_mouse_click(self, event):
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        row = self.df.iloc[self.idx]
        prices = [row[f'price_{i}'] for i in range(1, 11)]
        y_original = app_utils.denormalize_value(y, prices) if cfg.NORMALIZE_VIEW else y

        if cfg.NUM_PRICES.value == 1:
            self.df.at[self.idx, 'labeled_price_1'] = round(y_original, 3)
            self.idx += 1
        else:
            if self.click_stage == 1:
                self.df.at[self.idx, 'labeled_price_1'] = round(y_original, 3)
                self.first_click_y = y
                self.click_stage = 2
            else:
                self.df.at[self.idx, 'labeled_price_2'] = round(y_original, 3)
                self.click_stage = 1
                self.first_click_y = None
                self.idx += 1

        self.save_csv_with_a_description()
        self.show_data()


    def save_csv_with_a_description(self):
        description = "# " + cfg.CSV_DESCRIPTION + "\n"
        current_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        output_csv_path = f"{cfg.OUTPUT_DIR}/labeling_in_progress_{current_timestamp}.csv"
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write(description)
            self.df.to_csv(f, index=False)
        print("Data saved!")

    def find_and_plot_distances(self):
        app_utils.find_and_plot_distances(self)

    def go_forward(self):
        app_utils.go_forward(self)

    def go_backward(self):
        app_utils.go_backward(self)


if __name__ == "__main__":
    open_settings_window()

    df = pd.read_csv(cfg.DATA_FILE, comment="#")
    app_utils.validate_df(df)

    main_app = MainApp(df)

    start_control_window(main_app)

    main_app.show_data()

