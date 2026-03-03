# region imports
from AlgorithmImports import *
# endregion

import math
import pandas as pd
from datetime import timedelta

from code.swing_high_low_detection import swing_highs_lows_online
from code.entry_exit import (
    Bar as SimBar,
    SwingLevels,
    PositionDirection,
    TakeProfitMode,
    SameBarSlTpRule,
    update_last_swing_levels,
    detect_bos_signal,
    plan_trade_from_signal,
    check_exit_rules,
)
from code.stop_loss import StopLossManager
from code.risk import RiskConfig, size_position


class BosBreakoutEth15m(QCAlgorithm):
    """
    ETHUSD 15-minute BOS strategy for QuantConnect.

    Uses project mechanics:
    - swing_highs_lows_online
    - detect_bos_signal
    - plan_trade_from_signal
    - StopLossManager
    - risk.size_position
    - check_exit_rules (same-bar handling)
    """

    def Initialize(self):
        self.SetTimeZone("UTC")
        self.SetStartDate(2021, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)

        self.symbol = self.AddCrypto("ETHUSD", Resolution.Minute).Symbol
        self.SetBenchmark(self.symbol)

        # Best parameters from your research
        self.sl_mode = "fixed"
        self.fixed_pct = 0.0100
        self.buffer_pct = 0.000

        self.tp_mode = TakeProfitMode.RR_BASED
        self.tp_mult = 3.0

        self.same_bar_rule = SameBarSlTpRule.WORST_CASE
        self.cooldown_bars = 0
        self.max_trades_per_day = None

        # Keep same risk sizing style from project
        self.risk_config = RiskConfig(
            risk_budget_cash=100.0,
            max_quantity=None,
            min_risk_per_unit=None,
            use_buying_power_cap=False,
        )
        self.qty_decimals = 6

        # Swing detection parameters
        self.N_candidates = [5, 10, 20]
        self.N_confirmation = 3
        self.min_move_threshold = 0.0
        self.min_bars_between_swings = 3
        self.required_warmup_bars = max(self.N_candidates) + self.N_confirmation + 20

        self.stop_loss_manager = StopLossManager(
            mode=self.sl_mode,
            fixed_pct=self.fixed_pct,
            buffer_pct=self.buffer_pct,
        )

        self.swing_levels = SwingLevels()
        self.bars_15m = []
        self.last_applied_swing_bar_index = -1

        self.bar_index = 0
        self.cooldown_until = -1
        self.current_day = None
        self.trades_today = 0

        # Trade state
        self.state = "FLAT"  # FLAT | ENTRY_SUBMITTED | OPEN | EXIT_SUBMITTED
        self.trade_plan = None
        self.entry_ticket = None
        self.exit_ticket = None
        self.active_qty = 0.0
        self.active_sl_price = None
        self.active_tp_price = None
        self.pending_exit_reason = None
        self.pending_exit_price = None
        self.entry_fill_price = None
        self.entry_fill_time = None

        # Stats
        self.stat_bos = 0
        self.stat_plan_ok = 0
        self.stat_plan_fail = 0
        self.stat_skip_qty0 = 0
        self.stat_entries = 0
        self.stat_exit = 0
        self.stat_sl = 0
        self.stat_tp = 0
        self.stat_entry_reject = 0
        self.plan_fail_log_count = 0
        self.total_trade_pnl = 0.0

        self.consolidator = TradeBarConsolidator(timedelta(minutes=15))
        self.consolidator.DataConsolidated += self.On15MinuteBar
        self.SubscriptionManager.AddConsolidator(self.symbol, self.consolidator)

        self.Debug(
            "READY | ETHUSD 15m | sl=fixed(1%), tp=RR(3.0), same_bar=WORST_CASE, cooldown=0, max_trades_day=None"
        )

    def OnData(self, data: Slice):
        pass

    def On15MinuteBar(self, sender, tb: TradeBar):
        sim_bar = SimBar(
            open=float(tb.Open),
            high=float(tb.High),
            low=float(tb.Low),
            close=float(tb.Close),
            volume=float(tb.Volume) if tb.Volume is not None else None,
            time=str(tb.EndTime),
        )

        self.bars_15m.append(sim_bar)
        self.bar_index += 1

        self.Plot("Price", "Close", sim_bar.close)

        day = tb.EndTime.date()
        if self.current_day is None or day != self.current_day:
            self.current_day = day
            self.trades_today = 0

        if len(self.bars_15m) < self.required_warmup_bars:
            return

        self._update_swings()
        self._plot_levels()
        self._heal_state_if_needed()

        if self.state == "OPEN":
            if self._try_exit_with_v1_rules(sim_bar):
                return

        if self.state != "FLAT":
            return

        if self.max_trades_per_day is not None and self.trades_today >= self.max_trades_per_day:
            return

        if self.cooldown_bars > 0 and self.bar_index < self.cooldown_until:
            return

        if self._has_open_orders():
            return

        # Streaming approach from your project: signal on t-1, entry planned on bar t open.
        signal_t = len(self.bars_15m) - 2
        if signal_t < 0:
            return

        bos_signal = detect_bos_signal(
            bars=self.bars_15m,
            t=signal_t,
            swing_levels=self.swing_levels,
        )
        if bos_signal is None:
            return

        self.stat_bos += 1
        self.stop_loss_manager.reset()

        try:
            plan = plan_trade_from_signal(
                bars=self.bars_15m,
                bos_signal=bos_signal,
                swing_levels=self.swing_levels,
                stop_loss_manager=self.stop_loss_manager,
                tp_mode=self.tp_mode,
                tp_mult=self.tp_mult,
                risk_config=self.risk_config,
                buying_power_cash=float(self.Portfolio.Cash),
                position_sizer=self._position_sizer_fractional,
            )
        except Exception as e:
            self.stop_loss_manager.reset()
            self.stat_plan_fail += 1

            err = str(e)
            if "sized quantity is zero" in err or "quantity is zero" in err:
                self.stat_skip_qty0 += 1

            if self.plan_fail_log_count < 25:
                self.Debug(f"PLAN FAIL {tb.EndTime} | {e}")
                self.plan_fail_log_count += 1
            return

        signed_qty = plan.quantity if plan.direction == PositionDirection.LONG else -plan.quantity
        signed_qty = self._round_quantity(float(signed_qty))
        if signed_qty == 0:
            self.stat_skip_qty0 += 1
            return

        sl_price = self._round_price(float(plan.sl_price))
        tp_price = self._round_price(float(plan.tp_price))

        if not self._is_valid_bracket(plan.direction, float(plan.entry_price), sl_price, tp_price):
            self.stat_plan_fail += 1
            return

        self.trade_plan = plan
        self.active_sl_price = sl_price
        self.active_tp_price = tp_price
        self.pending_exit_reason = None
        self.pending_exit_price = None
        self.entry_fill_price = None
        self.entry_fill_time = None

        self.state = "ENTRY_SUBMITTED"
        self.entry_ticket = self.MarketOrder(
            self.symbol,
            signed_qty,
            tag=f"ENTRY|{plan.direction.value}|sig={plan.signal_candle_index}",
        )
        self.stat_plan_ok += 1

    def OnOrderEvent(self, orderEvent: OrderEvent):
        if self.entry_ticket and orderEvent.OrderId == self.entry_ticket.OrderId:
            if orderEvent.Status in [OrderStatus.Filled, OrderStatus.PartiallyFilled]:
                filled_qty = float(self.entry_ticket.QuantityFilled)
                if filled_qty != 0:
                    self.active_qty = filled_qty
                    if self.entry_fill_time is None:
                        self.entry_fill_time = self.Time
                        if orderEvent.FillPrice > 0:
                            self.entry_fill_price = float(orderEvent.FillPrice)
                        else:
                            self.entry_fill_price = float(self.trade_plan.entry_price)

                        self.state = "OPEN"
                        self.trades_today += 1
                        self.stat_entries += 1
                        self.Debug(
                            f"ENTRY {self.Time} | Qty={filled_qty:.6f} "
                            f"Entry={self.entry_fill_price:.2f} "
                            f"SL={self.active_sl_price:.2f} TP={self.active_tp_price:.2f}"
                        )
                return

            if orderEvent.Status in [OrderStatus.Canceled, OrderStatus.Invalid]:
                self.stat_entry_reject += 1
                if self.plan_fail_log_count < 25:
                    self.Debug(f"ENTRY REJECTED {self.Time} | {orderEvent.Message}")
                    self.plan_fail_log_count += 1
                self._clear_trade_state(mark_cooldown=False)
                return

        if self.exit_ticket and orderEvent.OrderId == self.exit_ticket.OrderId:
            if orderEvent.Status in [OrderStatus.Filled, OrderStatus.PartiallyFilled]:
                if self.Portfolio[self.symbol].Invested:
                    return

                fill_price = float(orderEvent.FillPrice) if orderEvent.FillPrice > 0 else 0.0
                reason = self.pending_exit_reason if self.pending_exit_reason is not None else "UNKNOWN"

                if reason == "SL":
                    self.stat_sl += 1
                elif reason == "TP":
                    self.stat_tp += 1

                self.stat_exit += 1

                if self.entry_fill_price is not None and fill_price > 0 and self.active_qty != 0:
                    pnl = (fill_price - self.entry_fill_price) * self.active_qty
                    self.total_trade_pnl += pnl

                model_exit = f"{self.pending_exit_price:.2f}" if self.pending_exit_price is not None else "n/a"
                self.Debug(
                    f"EXIT {self.Time} | Reason={reason} Fill={fill_price:.2f} ModelExit={model_exit}"
                )
                self._clear_trade_state(mark_cooldown=True)
                return

            if orderEvent.Status in [OrderStatus.Canceled, OrderStatus.Invalid]:
                if self.Portfolio[self.symbol].Invested:
                    self.Liquidate(self.symbol, "Exit order failed")
                self._clear_trade_state(mark_cooldown=True)
                return

    def OnEndOfAlgorithm(self):
        self.Debug(
            f"DONE | Trades={self.stat_entries} Exits={self.stat_exit} SL={self.stat_sl} TP={self.stat_tp} "
            f"BOS={self.stat_bos} PlanOK={self.stat_plan_ok} PlanFail={self.stat_plan_fail} "
            f"Qty0Skips={self.stat_skip_qty0} EntryReject={self.stat_entry_reject} "
            f"ApproxPnL={self.total_trade_pnl:.2f}"
        )

    def _update_swings(self):
        lookback = max(self.N_candidates) + self.N_confirmation + 300
        start = max(0, len(self.bars_15m) - lookback)
        idx = list(range(start, len(self.bars_15m)))

        ohlc = pd.DataFrame(
            {
                "close": [self.bars_15m[i].close for i in idx],
                "high": [self.bars_15m[i].high for i in idx],
                "low": [self.bars_15m[i].low for i in idx],
            },
            index=idx,
        )

        swings = swing_highs_lows_online(
            ohlc,
            N_candidates=self.N_candidates,
            N_confirmation=self.N_confirmation,
            min_move_threshold=self.min_move_threshold,
            min_bars_between_swings=self.min_bars_between_swings,
        )

        confirmed = swings.dropna(subset=["HighLow", "Level"])
        if confirmed.empty:
            return

        for swing_idx, row in confirmed.iterrows():
            swing_i = int(swing_idx)
            if swing_i <= self.last_applied_swing_bar_index:
                continue

            hl = float(row["HighLow"])
            lvl = float(row["Level"])

            self.swing_levels = update_last_swing_levels(
                self.swing_levels,
                highlow_flag=hl,
                level=lvl,
            )
            self.last_applied_swing_bar_index = swing_i

    def _try_exit_with_v1_rules(self, bar: SimBar) -> bool:
        if self.trade_plan is None:
            return False

        if not self.Portfolio[self.symbol].Invested:
            return False

        if self.exit_ticket is not None and self._has_open_orders():
            return True

        exit_event = check_exit_rules(
            bar=bar,
            direction=self.trade_plan.direction,
            sl_price=float(self.active_sl_price),
            tp_price=float(self.active_tp_price),
            same_bar_rule=self.same_bar_rule,
        )
        if exit_event is None:
            return False

        qty_to_close = -float(self.Portfolio[self.symbol].Quantity)
        qty_to_close = self._round_quantity(qty_to_close)

        if qty_to_close == 0:
            self._clear_trade_state(mark_cooldown=True)
            return True

        self.pending_exit_reason = exit_event.exit_reason.value
        self.pending_exit_price = float(exit_event.exit_price)

        self.state = "EXIT_SUBMITTED"
        self.exit_ticket = self.MarketOrder(
            self.symbol,
            qty_to_close,
            tag=f"EXIT|{self.pending_exit_reason}",
        )
        return True

    def _position_sizer_fractional(self, **kwargs):
        qty, reason = size_position(
            **kwargs,
            round_func=lambda x: float(round(x, self.qty_decimals)),
        )
        if qty is not None and qty <= 0:
            return None, "qty <= 0 after rounding"
        return qty, reason

    def _is_valid_bracket(self, direction: PositionDirection, entry: float, sl: float, tp: float) -> bool:
        if entry <= 0 or sl <= 0 or tp <= 0:
            return False

        if direction == PositionDirection.LONG:
            return sl < entry < tp

        return tp < entry < sl

    def _round_price(self, price: float) -> float:
        security = self.Securities[self.symbol]
        tick = float(security.SymbolProperties.MinimumPriceVariation)
        if tick <= 0:
            return float(price)

        return float(round(price / tick) * tick)

    def _round_quantity(self, qty: float) -> float:
        if qty == 0:
            return 0.0

        security = self.Securities[self.symbol]
        lot = float(security.SymbolProperties.LotSize)

        abs_qty = abs(float(qty))
        if lot > 0:
            steps = math.floor(abs_qty / lot)
            abs_qty = steps * lot

        abs_qty = float(round(abs_qty, self.qty_decimals))
        if abs_qty <= 0:
            return 0.0

        return abs_qty if qty > 0 else -abs_qty

    def _plot_levels(self):
        hi = self.swing_levels.last_swing_high_price
        lo = self.swing_levels.last_swing_low_price

        if hi is not None:
            self.Plot("Levels", "SwingHigh", float(hi))
        if lo is not None:
            self.Plot("Levels", "SwingLow", float(lo))

        if self.state == "OPEN" and self.active_sl_price is not None and self.active_tp_price is not None:
            self.Plot("Levels", "SL", float(self.active_sl_price))
            self.Plot("Levels", "TP", float(self.active_tp_price))

    def _has_open_orders(self) -> bool:
        return len(self.Transactions.GetOpenOrders(self.symbol)) > 0

    def _heal_state_if_needed(self):
        if self.state == "OPEN":
            if not self.Portfolio[self.symbol].Invested and not self._has_open_orders():
                self._clear_trade_state(mark_cooldown=False)
                return

        if self.state == "ENTRY_SUBMITTED":
            if not self.Portfolio[self.symbol].Invested and not self._has_open_orders():
                self._clear_trade_state(mark_cooldown=False)
                return

        if self.state == "EXIT_SUBMITTED":
            if not self.Portfolio[self.symbol].Invested and not self._has_open_orders():
                self._clear_trade_state(mark_cooldown=True)

    def _clear_trade_state(self, mark_cooldown: bool):
        self.stop_loss_manager.reset()

        self.state = "FLAT"
        self.trade_plan = None
        self.entry_ticket = None
        self.exit_ticket = None
        self.active_qty = 0.0
        self.active_sl_price = None
        self.active_tp_price = None
        self.pending_exit_reason = None
        self.pending_exit_price = None
        self.entry_fill_price = None
        self.entry_fill_time = None

        if mark_cooldown and self.cooldown_bars > 0:
            self.cooldown_until = self.bar_index + self.cooldown_bars
