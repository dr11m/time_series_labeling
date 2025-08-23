"""
Универсальный формат данных для временных рядов
"""
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
import json
import pandas as pd
import numpy as np


class TimeSeriesPoint(BaseModel):
    """Одна точка временного ряда"""
    timestamp: Union[int, float, str]  # Unix timestamp или datetime string
    value: float
    metadata: Optional[Dict[str, Any]] = None  # Дополнительные данные


class TimeSeries(BaseModel):
    """Один временной ряд"""
    id: str  # Уникальный идентификатор
    name: Optional[str] = None  # Название/описание
    points: List[TimeSeriesPoint]  # Точки временного ряда
    metadata: Optional[Dict[str, Any]] = None  # Метаданные ряда
    labeled_values: Optional[Dict[str, float]] = None  # Размеченные значения
    
    def get_values(self) -> List[float]:
        """Получить список значений"""
        return [point.value for point in self.points]
    
    def get_timestamps(self) -> List[Union[int, float, str]]:
        """Получить список временных меток"""
        return [point.timestamp for point in self.points]
    
    def length(self) -> int:
        """Длина временного ряда"""
        return len(self.points)
    
    def is_labeled(self) -> bool:
        """Проверить, размечен ли ряд"""
        return self.labeled_values is not None and len(self.labeled_values) > 0


class TimeSeriesDataset(BaseModel):
    """Набор временных рядов"""
    name: str
    description: Optional[str] = None
    series: List[TimeSeries]
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    
    def __len__(self) -> int:
        return len(self.series)
    
    def get_unlabeled_series(self) -> List[TimeSeries]:
        """Получить неразмеченные ряды"""
        return [ts for ts in self.series if not ts.is_labeled()]
    
    def get_labeled_series(self) -> List[TimeSeries]:
        """Получить размеченные ряды"""
        return [ts for ts in self.series if ts.is_labeled()]
    
    def filter_by_length(self, min_length: int = 0, max_length: Optional[int] = None) -> 'TimeSeriesDataset':
        """Фильтровать ряды по длине"""
        filtered_series = []
        for ts in self.series:
            length = ts.length()
            if length >= min_length and (max_length is None or length <= max_length):
                filtered_series.append(ts)
        
        return TimeSeriesDataset(
            name=self.name,
            description=self.description,
            series=filtered_series,
            metadata=self.metadata,
            created_at=self.created_at
        )


class DataAdapter:
    """Базовый класс для адаптеров данных"""
    
    def load_data(self, source: str) -> TimeSeriesDataset:
        """Загрузить данные из источника"""
        raise NotImplementedError
    
    def save_data(self, dataset: TimeSeriesDataset, destination: str) -> None:
        """Сохранить данные в назначение"""
        raise NotImplementedError


class JSONAdapter(DataAdapter):
    """Адаптер для работы с JSON форматом"""
    
    def load_data(self, source: str) -> TimeSeriesDataset:
        """Загрузить данные из JSON файла"""
        with open(source, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return TimeSeriesDataset(**data)
    
    def save_data(self, dataset: TimeSeriesDataset, destination: str) -> None:
        """Сохранить данные в JSON файл"""
        with open(destination, 'w', encoding='utf-8') as f:
            json.dump(dataset.model_dump(), f, indent=2, default=str)


def create_sample_dataset() -> TimeSeriesDataset:
    """Создать пример набора данных для демонстрации"""
    import numpy as np
    
    test_series = []
    for i in range(5):
        # Создаем ряды разной длины
        length = np.random.randint(8, 15)
        points = []
        for j in range(length):
            points.append(TimeSeriesPoint(
                timestamp=1731535200 + j * 86400,  # Unix timestamp
                value=1.0 + np.random.random() * 2,  # Случайная цена
                metadata={"source": "demo"}
            ))
        
        series = TimeSeries(
            id=f"demo_series_{i}",
            name=f"Demo Series {i}",
            points=points,
            metadata={
                "category": "test",
                "created_by": "system"
            }
        )
        test_series.append(series)
    
    return TimeSeriesDataset(
        name="Sample Dataset",
        description="Пример набора данных для демонстрации",
        series=test_series,
        created_at=datetime.now()
    )
