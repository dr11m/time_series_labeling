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
from cfg.cfg_loader import cfg
from src.data_formats import TimeSeriesDataset, TimeSeries, JSONAdapter, create_sample_dataset
import os

warnings.filterwarnings("ignore", category=FutureWarning)
current_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

class UniversalMainApp:
    def __init__(self, dataset: TimeSeriesDataset):
        self.dataset = dataset
        self.idx = 0
        self.click_stage = 1
        self.first_click_y = None
        self.fig = plt.gcf()
        self.fig.canvas.mpl_connect('key_press_event', self.handle_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.handle_mouse_click)
        self.__post_init__()

    def __post_init__(self):
        # Обрабатываем временные ряды согласно настройкам длины
        self.process_series_lengths()
        
        # Находим первый неразмеченный ряд
        while self.idx < len(self.dataset) and self.dataset.series[self.idx].is_labeled():
            self.idx += 1

    def show_data(self):
        if self.idx >= len(self.dataset):
            print("Все ряды размечены!")
            return
            
        series = self.dataset.series[self.idx]
        values = series.get_values()
        timestamps = series.get_timestamps()
        
        # Нормализуем для отображения если нужно
        display_values = app_utils.normalize_array(values) if cfg.NORMALIZE_VIEW else values
        
        # Подготавливаем оси X в зависимости от настроек
        if cfg.SHOW_TIMESTAMPS_AS_DATES and timestamps:
            # Показываем timestamps как даты
            try:
                x_points = []
                for ts in timestamps:
                    if isinstance(ts, (int, float)):
                        dt = datetime.datetime.fromtimestamp(ts)
                    else:
                        dt = pd.to_datetime(ts).to_pydatetime()
                    x_points.append(dt)
                x_label = "Date"
            except:
                # Если не удалось конвертировать, используем числа
                x_points = list(range(1, len(values) + 1))
                x_label = "Time Point"
        else:
            # Показываем как числа
            x_points = list(range(1, len(values) + 1))
            x_label = "Time Point"

        plt.clf()
        plt.plot(x_points, display_values, marker='o')
        plt.scatter(x_points, display_values, color='green')
        
        # Отображаем размеченные значения если они есть
        if series.labeled_values:
            for label_name, label_value in series.labeled_values.items():
                if cfg.NORMALIZE_VIEW:
                    # Нормализуем размеченное значение для отображения
                    display_label_value = app_utils.normalize_array([label_value])[0]
                else:
                    display_label_value = label_value
                
                # Рисуем горизонтальную линию для размеченного значения
                plt.axhline(y=display_label_value, color='orange', linestyle='--', alpha=0.8, 
                           label=f'Labeled {label_name}: {label_value:.3f}')
            
            # Добавляем легенду для размеченных значений
            plt.legend()
        
        # Добавляем текущую дату если включено
        if cfg.SHOW_CURRENT_DATE:
            current_time = datetime.datetime.now()
            if isinstance(x_points[0], datetime.datetime):
                # Если показываем даты, добавляем текущую дату
                plt.axvline(x=current_time, color='red', linestyle='--', alpha=0.7, label='Current Date')
                if not series.labeled_values:  # Добавляем легенду только если нет размеченных значений
                    plt.legend()
        
        if self.first_click_y is not None:
            plt.axhline(y=self.first_click_y, color='orange', linestyle='--')

        cursor = Cursor(plt.gca(), useblit=True, color='red', linewidth=1)
        plt.xlabel(x_label)
        plt.ylabel("Value")
        
        # Показываем информацию о ряде
        title = cfg.TITLE_TEMPLATE.format(
            idx=self.idx, 
            name=series.name or series.id,
            length=series.length()
        )
        plt.title(title)
        plt.grid(True)
        
        # Автоматическое форматирование меток оси X для дат
        if isinstance(x_points[0], datetime.datetime):
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator())
        
        plt.show()

    def handle_key_press(self, event):
        if event.key == "right":
            self.go_forward()
        elif event.key == "left":
            self.go_backward()
        elif event.key == "d":
            self.find_and_plot_distances()
        elif event.key == "f":
            self.filter_by_length()

    def handle_mouse_click(self, event):
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        series = self.dataset.series[self.idx]
        values = series.get_values()
        y_original = app_utils.denormalize_value(y, values) if cfg.NORMALIZE_VIEW else y

        if cfg.NUM_PRICES.value == 1:
            # Размечаем одну цену
            if series.labeled_values is None:
                series.labeled_values = {}
            series.labeled_values['price_1'] = round(y_original, 3)
            self.idx += 1
        else:
            # Размечаем две цены
            if series.labeled_values is None:
                series.labeled_values = {}
                
            if self.click_stage == 1:
                series.labeled_values['price_1'] = round(y_original, 3)
                self.first_click_y = y
                self.click_stage = 2
            else:
                series.labeled_values['price_2'] = round(y_original, 3)
                self.click_stage = 1
                self.first_click_y = None
                self.idx += 1

        self.save_data()
        self.show_data()

    def save_data(self):
        """Сохранить данные в JSON формате"""
        
        # Сохраняем в JSON
        json_path = f"{cfg.OUTPUT_DIR}/universal_labeling_{current_timestamp}.json"
        json_adapter = JSONAdapter()
        json_adapter.save_data(self.dataset, json_path)
        
        print(f"Data saved to {json_path}")

    def find_and_plot_distances(self):
        """Найти похожие паттерны"""
        if self.idx >= len(self.dataset):
            return
            
        current_series = self.dataset.series[self.idx]
        current_values = current_series.get_values()
        
        # Получаем размеченные ряды
        labeled_series = self.dataset.get_labeled_series()
        if len(labeled_series) < 2:
            print("Недостаточно размеченных данных!")
            return
            
        # Находим похожие ряды
        similar_series = []
        for labeled_ts in labeled_series:
            labeled_values = labeled_ts.get_values()
            distance = app_utils.calculate_dtw_distance(current_values, labeled_values)
            similar_series.append((labeled_ts, distance))
        
        # Сортируем по расстоянию
        similar_series.sort(key=lambda x: x[1])
        similar_series = similar_series[:4]  # Берем 4 самых похожих
        
        # Показываем графики
        fig = plt.figure(num="Similar Patterns", figsize=(12, 6))
        fig.suptitle(f"Similar patterns for series {current_series.name}", fontsize=12)
        axes = fig.subplots(2, 2)
        
        for i, (series, distance) in enumerate(similar_series):
            ax = axes[i // 2, i % 2]
            values = series.get_values()
            timestamps = series.get_timestamps()
            
            # Подготавливаем оси X в зависимости от настроек
            if cfg.SHOW_TIMESTAMPS_AS_DATES and timestamps:
                try:
                    x_points = []
                    for ts in timestamps:
                        if isinstance(ts, (int, float)):
                            dt = datetime.datetime.fromtimestamp(ts)
                        else:
                            dt = pd.to_datetime(ts).to_pydatetime()
                        x_points.append(dt)
                except:
                    x_points = list(range(1, len(values) + 1))
            else:
                x_points = list(range(1, len(values) + 1))
            
            if cfg.NORMALIZE_VIEW:
                values_norm = app_utils.normalize_array(values)
                ax.plot(x_points, values_norm)
            else:
                ax.plot(x_points, values)
            
            # Отображаем все размеченные значения
            if series.labeled_values:
                for label_name, label_value in series.labeled_values.items():
                    if cfg.NORMALIZE_VIEW:
                        display_label_value = app_utils.normalize_array([label_value])[0]
                    else:
                        display_label_value = label_value
                    
                    ax.axhline(y=display_label_value, color='blue', linestyle='--', alpha=0.8,
                               label=f'{label_name}: {label_value:.3f}')
                
                # Добавляем легенду для размеченных значений
                ax.legend(fontsize=8)
            
            # Добавляем текущую дату если включено
            if cfg.SHOW_CURRENT_DATE and isinstance(x_points[0], datetime.datetime):
                current_time = datetime.datetime.now()
                ax.axvline(x=current_time, color='red', linestyle='--', alpha=0.7)
                # Автоматическое форматирование меток оси X для дат
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator())
                    
            ax.set_title(f"Similar {i+1} (D: {distance:.2f})")
            ax.grid(True)
        
        plt.tight_layout()
        plt.show(block=False)

    def filter_by_length(self):
        """Фильтровать ряды по длине"""
        print("\nФильтрация по длине:")
        print("1. Минимальная длина")
        print("2. Максимальная длина") 
        print("3. Диапазон длин")
        print("4. Применить настройки из конфига")
        
        try:
            choice = input("Выберите опцию (1-4): ").strip()
            
            if choice == "1":
                min_len = int(input("Введите минимальную длину: "))
                self.dataset = self.dataset.filter_by_length(min_length=min_len)
                print(f"Осталось рядов: {len(self.dataset)}")
                
            elif choice == "2":
                max_len = int(input("Введите максимальную длину: "))
                self.dataset = self.dataset.filter_by_length(max_length=max_len)
                print(f"Осталось рядов: {len(self.dataset)}")
                
            elif choice == "3":
                min_len = int(input("Введите минимальную длину: "))
                max_len = int(input("Введите максимальную длину: "))
                self.dataset = self.dataset.filter_by_length(min_length=min_len, max_length=max_len)
                print(f"Осталось рядов: {len(self.dataset)}")
                
            elif choice == "4":
                # Применяем настройки из конфига
                self.process_series_lengths()
                print(f"Применены настройки из конфига. Осталось рядов: {len(self.dataset)}")
                
            # Сбрасываем индекс
            self.idx = 0
            self.__post_init__()
            
        except (ValueError, KeyboardInterrupt):
            print("Отменено")

    def go_forward(self):
        if self.idx < len(self.dataset) - 1:
            self.idx += 1
            self.click_stage = 1
            self.show_data()
            print("Moved forward.")

    def process_series_lengths(self):
        """Обработать временные ряды согласно настройкам длины"""
        if not hasattr(cfg, 'TARGET_SERIES_LENGTH'):
            return
            
        target_length = cfg.TARGET_SERIES_LENGTH
        skip_shorter = getattr(cfg, 'SKIP_SHORTER_SERIES', True)
        take_last_n = getattr(cfg, 'TAKE_LAST_N_VALUES', True)
        
        processed_series = []
        skipped_count = 0
        truncated_count = 0
        
        for series in self.dataset.series:
            current_length = series.length()
            
            # Пропускаем короткие ряды
            if skip_shorter and current_length < target_length:
                skipped_count += 1
                continue
            
            # Если ряд длиннее целевой длины, берем последние N значений
            if take_last_n and current_length > target_length:
                series.points = series.points[-target_length:]
                truncated_count += 1
            
            processed_series.append(series)
        
        # Обновляем dataset
        self.dataset.series = processed_series
        
        print(f"Обработка временных рядов завершена:")
        print(f"  - Целевая длина: {target_length}")
        print(f"  - Пропущено коротких рядов: {skipped_count}")
        print(f"  - Обрезано длинных рядов: {truncated_count}")
        print(f"  - Осталось рядов: {len(processed_series)}")

    def go_backward(self):
        if self.idx > 0:
            self.idx -= 1
            self.click_stage = 1
            self.show_data()
            print("Moved backward.")


def load_data_from_source(source_path: str) -> TimeSeriesDataset:
    """Загрузить данные из JSON файла"""
    if not source_path.endswith('.json'):
        raise ValueError(f"Only JSON format is supported: {source_path}")
    
    adapter = JSONAdapter()
    return adapter.load_data(source_path)


if __name__ == "__main__":
    open_settings_window()

    # Загружаем данные
    if hasattr(cfg, 'DATA_FILE') and os.path.exists(cfg.DATA_FILE):
        dataset = load_data_from_source(cfg.DATA_FILE)
    else:
        # Создаем демонстрационные данные
        dataset = create_sample_dataset()

    print(f"Loaded {len(dataset)} time series")
    print(f"Unlabeled: {len(dataset.get_unlabeled_series())}")
    print(f"Labeled: {len(dataset.get_labeled_series())}")

    main_app = UniversalMainApp(dataset)
    start_control_window(main_app)
    main_app.show_data()
