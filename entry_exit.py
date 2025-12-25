from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

# ВАЖНО:
# stop_loss.py импортирует Bar, PositionDirection из entry_exit.
# Поэтому эти имена должны жить в этом модуле в таком виде.


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
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    time: Optional[str] = None


@dataclass(frozen=True)
class SwingLevels:
    last_swing_high_price: Optional[float]
    last_swing_low_price: Optional[float]


@dataclass(frozen=True)
class BosSignal:
    direction: PositionDirection
    signal_candle_index: int  # t


@dataclass(frozen=True)
class TradePlan:
    direction: PositionDirection
    signal_candle_index: int  # t (где BOS по close)
    entry_candle_index: int   # t+1 (где вход по open)
    entry_price: float
    sl_price: float
    tp_price: float


@dataclass(frozen=True)
class TradeExit:
    exit_price: float
    exit_reason: ExitReason


def detect_bos_signal(*, bars: list[Bar], t: int, swing_levels: SwingLevels) -> Optional[BosSignal]:
    """
    V1 BOS:
      - BOS Long: Close[t] > lastSwingHighPrice
      - BOS Short: Close[t] < lastSwingLowPrice

    Сигнал на close[t], но вход возможен только если существует t+1.
    """
    if t < 0 or t >= len(bars):
        raise IndexError("Bar index out of range.")

    if t + 1 >= len(bars):
        return None

    close_t = bars[t].close

    if swing_levels.last_swing_high_price is not None and close_t > swing_levels.last_swing_high_price:
        return BosSignal(direction=PositionDirection.LONG, signal_candle_index=t)

    if swing_levels.last_swing_low_price is not None and close_t < swing_levels.last_swing_low_price:
        return BosSignal(direction=PositionDirection.SHORT, signal_candle_index=t)

    return None


def calculate_take_profit_price(
    *,
    direction: PositionDirection,
    tp_mode: TakeProfitMode,
    entry_price: float,
    sl_price: float,
    tp_mult: float,
    swing_levels: SwingLevels,
) -> float:
    """
    TP V1:
      RR_BASED:
        R = |Entry - SL|
        Long:  TP = Entry + k*R
        Short: TP = Entry - k*R

      RANGE_BASED:
        range = lastSwingHigh - lastSwingLow
        Long:  TP = Entry + range
        Short: TP = Entry - range
    """
    if tp_mode == TakeProfitMode.RR_BASED:
        r = abs(entry_price - sl_price)
        if r <= 0:
            raise ValueError("RR_BASED: invalid R (entry_price must differ from sl_price).")
        if tp_mult <= 0:
            raise ValueError("RR_BASED: tp_mult must be > 0.")

        return entry_price + tp_mult * r if direction == PositionDirection.LONG else entry_price - tp_mult * r

    if tp_mode == TakeProfitMode.RANGE_BASED:
        hi = swing_levels.last_swing_high_price
        lo = swing_levels.last_swing_low_price
        if hi is None or lo is None:
            raise ValueError("RANGE_BASED: requires both last swing high and last swing low.")
        rng = hi - lo
        if rng <= 0:
            raise ValueError("RANGE_BASED: invalid range (swing high must be > swing low).")

        return entry_price + rng if direction == PositionDirection.LONG else entry_price - rng

    raise ValueError(f"Unsupported tp_mode: {tp_mode}")


def plan_trade_from_signal(
    *,
    bars: list[Bar],
    bos_signal: BosSignal,
    swing_levels: SwingLevels,
    stop_loss_manager,  # StopLossManager из твоего файла
    tp_mode: TakeProfitMode,
    tp_mult: float,
) -> TradePlan:
    """
    ГДЕ ENTRY:
      Entry (исполнение) = Open[t+1]

    Тут мы:
      - берём entry_price по open следующей свечи
      - фиксируем SL через stop_loss_manager.on_entry(...)
      - считаем TP
      - возвращаем TradePlan
    """
    t = bos_signal.signal_candle_index
    entry_candle_index = t + 1
    if entry_candle_index >= len(bars):
        raise ValueError("Cannot plan entry: t+1 bar does not exist.")

    entry_price = bars[entry_candle_index].open

    # Берём SL строго через твой StopLossManager :contentReference[oaicite:1]{index=1}
    # Передаём всё сразу: он сам проверит, что нужно для выбранного режима.
    sl_price = stop_loss_manager.on_entry(
        direction=bos_signal.direction,
        entry_price=entry_price,
        last_swing_high=swing_levels.last_swing_high_price,
        last_swing_low=swing_levels.last_swing_low_price,
        signal_bar=bars[t],
    )

    tp_price = calculate_take_profit_price(
        direction=bos_signal.direction,
        tp_mode=tp_mode,
        entry_price=entry_price,
        sl_price=sl_price,
        tp_mult=tp_mult,
        swing_levels=swing_levels,
    )

    return TradePlan(
        direction=bos_signal.direction,
        signal_candle_index=t,
        entry_candle_index=entry_candle_index,
        entry_price=entry_price,
        sl_price=sl_price,
        tp_price=tp_price,
    )


def check_exit_rules(
    *,
    bar: Bar,
    direction: PositionDirection,
    sl_price: float,
    tp_price: float,
    same_bar_rule: SameBarSlTpRule,
) -> Optional[TradeExit]:
    """
    Exit V1:
      LONG:  SL if Low <= SL, TP if High >= TP
      SHORT: SL if High >= SL, TP if Low <= TP

    Same bar SL+TP:
      - WORST_CASE: считаем SL первым
      - OPEN_PROXIMITY: кто ближе к open, тот первый
      - LOWER_TIMEFRAME: не реализуем тут
    """
    if direction == PositionDirection.LONG:
        sl_hit = bar.low <= sl_price
        tp_hit = bar.high >= tp_price
    else:
        sl_hit = bar.high >= sl_price
        tp_hit = bar.low <= tp_price

    if not sl_hit and not tp_hit:
        return None

    if sl_hit and not tp_hit:
        return TradeExit(exit_price=sl_price, exit_reason=ExitReason.SL)

    if tp_hit and not sl_hit:
        return TradeExit(exit_price=tp_price, exit_reason=ExitReason.TP)

    # both hit
    if same_bar_rule == SameBarSlTpRule.WORST_CASE:
        return TradeExit(exit_price=sl_price, exit_reason=ExitReason.SL)

    if same_bar_rule == SameBarSlTpRule.OPEN_PROXIMITY:
        sl_dist = abs(bar.open - sl_price)
        tp_dist = abs(bar.open - tp_price)
        if sl_dist <= tp_dist:
            return TradeExit(exit_price=sl_price, exit_reason=ExitReason.SL)
        return TradeExit(exit_price=tp_price, exit_reason=ExitReason.TP)

    if same_bar_rule == SameBarSlTpRule.LOWER_TIMEFRAME:
        raise NotImplementedError("LOWER_TIMEFRAME requires lower TF data in the backtest engine.")

    raise ValueError(f"Unsupported same_bar_rule: {same_bar_rule}")
