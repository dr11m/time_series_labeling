from enum import Enum
from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import Optional
import math


class SaleRow(BaseModel):
    name: str
    buy_price: float
    price_1: float
    price_2: float
    price_3: float
    price_4: float
    price_5: float
    price_6: float
    price_7: float
    price_8: float
    price_9: float
    price_10: float
    price_11: Optional[float] = None
    price_12: Optional[float] = None
    price_13: Optional[float] = None
    price_14: Optional[float] = None
    price_15: Optional[float] = None
    ts_1: int
    ts_2: int
    ts_3: int
    ts_4: int
    ts_5: int
    ts_6: int
    ts_7: int
    ts_8: int
    ts_9: int
    ts_10: int
    ts_11: Optional[int] = None   
    ts_12: Optional[int] = None
    ts_13: Optional[int] = None
    ts_14: Optional[int] = None
    ts_15: Optional[int] = None
    sold_price: Optional[float] = None
    date_added: Optional[datetime] = None

    @field_validator('*', mode='before')
    def replace_nan(cls, v):
        if isinstance(v, float) and math.isnan(v):
            return None
        return v



class LabeledPriceAmount(Enum):
    ONE = 1
    TWO = 2