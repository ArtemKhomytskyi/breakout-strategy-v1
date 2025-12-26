from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from risk import RiskConfig, size_position

# IMPORTANT:
# If your stop_loss module imports Bar and PositionDirection from this file,
# these names must exist here and keep the same meaning.


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
    OHLC candlestick bar.
    Volume/time are optional and not required for V1 entry/exit rules.
    """
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    time: Optional[str] = None  # Replace with datetime if you prefer.


@dataclass(frozen=True)
class SwingLevels:
    """
    Snapshot of swing levels used by the strategy.

    Names match the V1 document:
      - last_swing_high_price == lastSwingHighPrice
      - last_swing_low_price == lastSwingLowPrice
    """
    last_swing_high_price: Optional[float] = None
    last_swing_low_price: Optional[float] = None


def has_level(level: Optional[float]) -> bool:
    """Level is usable only if it is not None/NaN."""
    return level is not None and not (isinstance(level, float) and math.isnan(level))


def bos_up(close: float, last_high: Optional[float]) -> bool:
    """Break of structure to the upside."""
    return has_level(last_high) and close > last_high  # type: ignore[arg-type]


def bos_down(close: float, last_low: Optional[float]) -> bool:
    """Break of structure to the downside."""
    return has_level(last_low) and close < last_low  # type: ignore[arg-type]


def update_last_swing_levels(
    swing_levels: SwingLevels,
    *,
    highlow_flag: Optional[float],
    level: Optional[float],
) -> SwingLevels:
    """
    Update last swing levels only when a confirmed swing is observed.

    Args:
        highlow_flag: 1 for swing high, -1 for swing low, anything else leaves levels unchanged.
        level: swing price level; ignored if invalid.
    """
    if not has_level(level):
        return swing_levels

    if highlow_flag == 1:
        return SwingLevels(last_swing_high_price=level, last_swing_low_price=swing_levels.last_swing_low_price)
    if highlow_flag == -1:
        return SwingLevels(last_swing_high_price=swing_levels.last_swing_high_price, last_swing_low_price=level)
    return swing_levels


@dataclass(frozen=True)
class BosSignal:
    """
    BOS signal detected on the CLOSE of the signal candle (t).

    Execution in V1 is deterministic:
      - signal is confirmed on Close[t]
      - entry is executed on Open[t+1]
    """
    direction: PositionDirection
    signal_candle_index: int  # t


@dataclass(frozen=True)
class TradePlan:
    """
    Trade plan created at entry time (V1 fixes SL/TP at entry).

    Contains:
      - signal_candle_index: t (where BOS was detected on close)
      - entry_candle_index: t+1 (where we execute on open)
      - entry_price, sl_price, tp_price, quantity
    """
    direction: PositionDirection
    signal_candle_index: int
    entry_candle_index: int
    entry_price: float
    sl_price: float
    tp_price: float
    quantity: float


@dataclass(frozen=True)
class TradeExit:
    """
    Exit event for a trade: price and reason (SL or TP).
    """
    exit_price: float
    exit_reason: ExitReason


def detect_bos_signal(*, bars: list[Bar], t: int, swing_levels: SwingLevels) -> Optional[BosSignal]:
    """
    V1 BOS definition:

      - BOS Long:  Close[t] > lastSwingHighPrice
      - BOS Short: Close[t] < lastSwingLowPrice

    Signal is evaluated on the CLOSE of bar t.
    Entry requires bar t+1 to exist (executed on Open[t+1]).
    """
    if t < 0 or t >= len(bars):
        raise IndexError("Bar index out of range.")

    # We need the next bar for execution on Open[t+1].
    if t + 1 >= len(bars):
        return None

    close_t = bars[t].close

    if not has_level(swing_levels.last_swing_high_price):
        assert not bos_up(close_t, swing_levels.last_swing_high_price)
    if not has_level(swing_levels.last_swing_low_price):
        assert not bos_down(close_t, swing_levels.last_swing_low_price)

    if bos_up(close_t, swing_levels.last_swing_high_price):
        return BosSignal(direction=PositionDirection.LONG, signal_candle_index=t)

    if bos_down(close_t, swing_levels.last_swing_low_price):
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
    V1 Take Profit modes:

    1) RR_BASED:
        R = abs(Entry - SL)
        Long:  TP = Entry + k * R
        Short: TP = Entry - k * R

    2) RANGE_BASED:
        range = lastSwingHighPrice - lastSwingLowPrice
        Long:  TP = Entry + range
        Short: TP = Entry - range
    """
    if tp_mode == TakeProfitMode.RR_BASED:
        r = abs(entry_price - sl_price)
        if r <= 0:
            raise ValueError("RR_BASED: invalid R (entry_price must differ from sl_price).")
        if tp_mult <= 0:
            raise ValueError("RR_BASED: tp_mult must be > 0.")

        if direction == PositionDirection.LONG:
            return entry_price + tp_mult * r
        return entry_price - tp_mult * r

    if tp_mode == TakeProfitMode.RANGE_BASED:
        hi = swing_levels.last_swing_high_price
        lo = swing_levels.last_swing_low_price
        if not (has_level(hi) and has_level(lo)):
            raise ValueError("RANGE_BASED: requires both last swing high and last swing low.")
        assert hi is not None and lo is not None
        rng = hi - lo
        if rng <= 0:
            raise ValueError("RANGE_BASED: invalid range (swing high must be > swing low).")

        if direction == PositionDirection.LONG:
            return entry_price + rng
        return entry_price - rng

    raise ValueError(f"Unsupported tp_mode: {tp_mode}")


def plan_trade_from_signal(
    *,
    bars: list[Bar],
    bos_signal: BosSignal,
    swing_levels: SwingLevels,
    stop_loss_manager,
    tp_mode: TakeProfitMode,
    tp_mult: float,
    risk_config: RiskConfig,
    buying_power_cash: Optional[float] = None,
    position_sizer=size_position,
) -> TradePlan:
    """
    WHERE ENTRY HAPPENS (V1):

      - Signal candle index = t (BOS confirmed on Close[t])
      - Entry candle index  = t+1
      - Entry price         = Open[t+1]

    This function:
      1) Takes entry_price from Open[t+1]
      2) Fixes SL using your StopLossManager.on_entry(...)
      3) Calculates position size (risk-based)
      4) Calculates TP (RR-based or Range-based)
      5) Returns a TradePlan with entry/sl/tp/qty fixed
    """
    t = bos_signal.signal_candle_index
    entry_candle_index = t + 1
    if entry_candle_index >= len(bars):
        raise ValueError("Cannot plan entry: next candle (t+1) does not exist.")

    entry_price = bars[entry_candle_index].open

    # Stop Loss is fixed at entry using your stop-loss module.
    sl_price = stop_loss_manager.on_entry(
        direction=bos_signal.direction,
        entry_price=entry_price,
        last_swing_high=swing_levels.last_swing_high_price,
        last_swing_low=swing_levels.last_swing_low_price,
        signal_bar=bars[t],
    )

    qty, refuse_reason = position_sizer(
        direction=bos_signal.direction,
        entry_price=entry_price,
        sl_price=sl_price,
        risk_config=risk_config,
        buying_power_cash=buying_power_cash,
    )
    if qty is None or qty <= 0:
        raise ValueError(
            f"Position sizing refused (reason={refuse_reason}, dir={bos_signal.direction}, entry={entry_price}, sl={sl_price})"
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
        quantity=qty,
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
    V1 exit rules:

    LONG:
      - SL hit if Low <= SL
      - TP hit if High >= TP

    SHORT:
      - SL hit if High >= SL
      - TP hit if Low <= TP

    If both SL and TP are hit within the same bar (OHLC-only ambiguity),
    we apply a deterministic tie-breaking rule:

      - WORST_CASE: assume SL first
      - OPEN_PROXIMITY: assume whichever level is closer to bar.open is hit first
      - LOWER_TIMEFRAME: not implemented in this module
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

    # Both hit in the same bar
    if same_bar_rule == SameBarSlTpRule.WORST_CASE:
        return TradeExit(exit_price=sl_price, exit_reason=ExitReason.SL)

    if same_bar_rule == SameBarSlTpRule.OPEN_PROXIMITY:
        sl_dist = abs(bar.open - sl_price)
        tp_dist = abs(bar.open - tp_price)
        if sl_dist <= tp_dist:
            return TradeExit(exit_price=sl_price, exit_reason=ExitReason.SL)
        return TradeExit(exit_price=tp_price, exit_reason=ExitReason.TP)

    if same_bar_rule == SameBarSlTpRule.LOWER_TIMEFRAME:
        raise NotImplementedError(
            "LOWER_TIMEFRAME requires lower timeframe data and must be handled in the backtest engine."
        )

    raise ValueError(f"Unsupported same_bar_rule: {same_bar_rule}")
