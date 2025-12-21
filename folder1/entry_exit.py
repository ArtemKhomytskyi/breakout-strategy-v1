from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PositionDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class ExitReason(str, Enum):
    SL = "SL"
    TP = "TP"


class TakeProfitMode(str, Enum):
    RR_BASED = "RR_BASED"       
    RANGE_BASED = "RANGE_BASED"  


class SameBarSlTpRule(str, Enum):
    WORST_CASE = "WORST_CASE"            
    OPEN_PROXIMITY = "OPEN_PROXIMITY"    
    LOWER_TIMEFRAME = "LOWER_TIMEFRAME"  


@dataclass(frozen=True)
class Bar:
    """
    Свеча (Bar): OHLC + опционально volume/time.
    """
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    time: Optional[str] = None


@dataclass(frozen=True)
class SwingLevels:
    """
    Снимок swing уровней на текущем баре.
    Названия соответствуют документу: lastSwingHighPrice / lastSwingLowPrice.
    """
    last_swing_high_price: Optional[float]
    last_swing_low_price: Optional[float]


@dataclass(frozen=True)
class BosSignal:
    """
    BOS-сигнал определяется по закрытию signal_candle (t).
    Исполнение входа будет на open следующей свечи (t+1).
    """
    direction: PositionDirection
    signal_candle_index: int


@dataclass(frozen=True)
class TradePlan:
    """
    План сделки: рассчитанные entry/sl/tp и индексы исполнения.
    """
    direction: PositionDirection
    entry_candle_index: int   # t+1
    entry_price: float
    sl_price: float
    tp_price: float


@dataclass(frozen=True)
class TradeExit:
    """
    Событие выхода по SL/TP.
    """
    exit_price: float
    exit_reason: ExitReason


def detect_bos_signal(
    *,
    bars: list[Bar],
    t: int,
    swing_levels: SwingLevels,
) -> Optional[BosSignal]:
    """
    BOS (V1) по закрытию свечи t:
      - BOS Long: Close[t] > lastSwingHighPrice
      - BOS Short: Close[t] < lastSwingLowPrice

    Возвращает сигнал на свече t (signal_candle), но вход выполняется на open[t+1].
    """
    if t < 0 or t >= len(bars):
        raise IndexError("Bar index out of range.")