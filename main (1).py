# region imports
from AlgorithmImports import *
# endregion

import math
import pandas as pd
from datetime import timedelta

# --- Your workspaces ---
from swing_high_low_detection import swing_highs_lows_online

from entry_exit import (
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

from stop_loss import StopLossManager
from risk import RiskConfig, size_position


class BosBreakoutV1_15m_ETH(QCAlgorithm):
    """
    V1 BOS breakout on 15-minute bars, ETHUSD, workspace-driven.

    IMPORTANT: This version simulates trades internally (OHLC rules),
    and plots Sim/Equity. QC portfolio metrics will remain 0 because
    we do NOT submit real orders (by design for deterministic V1).
    """

    def Initialize(self):
        # --------------------
        # Backtest config (shorter period for sanity)
        # --------------------
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)

        # --------------------
        # Asset + 15m consolidator
        # --------------------
        self.symbol = self.AddCrypto("ETHUSD", Resolution.Minute).Symbol
        self.SetBenchmark(self.symbol)

        self.consolidator = TradeBarConsolidator(timedelta(minutes=15))
        self.consolidator.DataConsolidated += self.On15MinuteBar
        self.SubscriptionManager.AddConsolidator(self.symbol, self.consolidator)

        # --------------------
        # BEST CONFIG from your table screenshot (least-bad among shown)
        # SL: structural, fixed_pct=0.005, buffer_pct=0.002
        # TP: RANGE_BASED, tp_mult=1.0 (unused in RANGE_BASED)
        # same_bar_rule: WORST_CASE
        # --------------------
        self.sl_mode = "structural"
        self.fixed_pct = 0.005
        self.buffer_pct = 0.002

        self.tp_mode = TakeProfitMode.RANGE_BASED
        self.tp_mult = 1.0

        self.same_bar_rule = SameBarSlTpRule.WORST_CASE

        # Risk sizing
        self.risk_config = RiskConfig(
            risk_budget_cash=100.0,
            max_quantity=None,
            min_risk_per_unit=None,
            use_buying_power_cap=False
        )

        # Crypto fractional sizing
        self.qty_decimals = 6

        # Protections (off for now)
        self.cooldown_bars = 0
        self.max_trades_per_day = None

        # --------------------
        # Swing detection params
        # NOTE: your function uses a single last_swing_index across all N_candidates.
        # Keep as-is, but start with a moderate set.
        # --------------------
        self.N_candidates = [5, 10, 20]
        self.N_confirmation = 3
        self.min_move_threshold = 0.0
        self.min_bars_between_swings = 3

        self.required_warmup_bars = max(self.N_candidates) + self.N_confirmation + 20

        # --------------------
        # State
        # --------------------
        self.swing_levels = SwingLevels()
        # Last valid (hi, lo) pair for RANGE_BASED TP
        self.last_valid_range = None  # tuple[float, float] | None
        self.stop_loss_manager = StopLossManager(
            mode=self.sl_mode,
            fixed_pct=self.fixed_pct,
            buffer_pct=self.buffer_pct
        )

        self.bars_15m = []  # list[Bar]
        self.trade_plan = None
        self.state = "FLAT"  # FLAT/LONG/SHORT

        self._last_applied_swing_bar_index = -1

        self._bar_index = 0
        self._cooldown_until = -1
        self._current_day = None
        self._trades_today = 0

        # Stats
        self.stat_bos = 0
        self.stat_plan_ok = 0
        self.stat_plan_fail = 0
        self.stat_exit = 0
        self.stat_skip_qty0 = 0
        self.stat_fail_logged = 0

        # Sim equity + trades
        self.trades = []
        self.sim_equity = float(self.Portfolio.Cash)
        self.Plot("Sim", "Equity", self.sim_equity)

    def OnData(self, data: Slice):
        pass

    def _position_sizer_fractional(self, **kwargs):
        # Use your risk.size_position but allow fractional quantities
        qty, reason = size_position(
            **kwargs,
            round_func=lambda x: float(round(x, self.qty_decimals))
        )
        if qty is not None and qty <= 0:
            return None, "qty <= 0 after rounding"
        return qty, reason

    def On15MinuteBar(self, sender, tb: TradeBar):
        bar = SimBar(
            open=float(tb.Open),
            high=float(tb.High),
            low=float(tb.Low),
            close=float(tb.Close),
            volume=float(tb.Volume) if tb.Volume is not None else None,
            time=str(tb.EndTime)
        )

        self.bars_15m.append(bar)
        self._bar_index += 1
        t = len(self.bars_15m) - 1

        # Daily reset
        day = tb.EndTime.date()
        if self._current_day is None or day != self._current_day:
            self._current_day = day
            self._trades_today = 0

        # Progress log (so you don't guess)
        if self._bar_index % 2000 == 0:
            self.Debug(
                f"{tb.EndTime} | bars={self._bar_index} | "
                f"swing_hi={self.swing_levels.last_swing_high_price} swing_lo={self.swing_levels.last_swing_low_price} | "
                f"BOS={self.stat_bos} PlanOK={self.stat_plan_ok} PlanFail={self.stat_plan_fail} "
                f"Exits={self.stat_exit} Trades={len(self.trades)} SimEq={self.sim_equity:.2f}"
            )

        # Warmup
        if len(self.bars_15m) < self.required_warmup_bars:
            return

        # ----------------------------
        # 1) Update swings (rolling window)
        # ----------------------------
        lookback = max(self.N_candidates) + self.N_confirmation + 300
        start = max(0, len(self.bars_15m) - lookback)
        idx = list(range(start, len(self.bars_15m)))

        ohlc = pd.DataFrame(
            {
                "close": [self.bars_15m[i].close for i in idx],
                "high":  [self.bars_15m[i].high for i in idx],
                "low":   [self.bars_15m[i].low for i in idx],
            },
            index=idx
        )

        swings = swing_highs_lows_online(
            ohlc,
            N_candidates=self.N_candidates,
            N_confirmation=self.N_confirmation,
            min_move_threshold=self.min_move_threshold,
            min_bars_between_swings=self.min_bars_between_swings
        )

        confirmed = swings.dropna(subset=["HighLow", "Level"])
        if not confirmed.empty:
            for swing_idx, row in confirmed.iterrows():
                swing_i = int(swing_idx)
                if swing_i <= self._last_applied_swing_bar_index:
                    continue

            hl = float(row["HighLow"])
            lvl = float(row["Level"])

            # Update swing levels (your workspace function)
            self.swing_levels = update_last_swing_levels(
                self.swing_levels,
                highlow_flag=hl,
                level=lvl
            )

            # Maintain last_valid_range for RANGE_BASED:
            hi = self.swing_levels.last_swing_high_price
            lo = self.swing_levels.last_swing_low_price
            if hi is not None and lo is not None:
                if not (isinstance(hi, float) and math.isnan(hi)) and not (isinstance(lo, float) and math.isnan(lo)):
                    if hi > lo:
                        self.last_valid_range = (float(hi), float(lo))

            self._last_applied_swing_bar_index = swing_i


        # ----------------------------
        # 2) Exit
        # ----------------------------
        if self.state in ("LONG", "SHORT") and self.trade_plan is not None:
            exit_event = check_exit_rules(
                bar=bar,
                direction=self.trade_plan.direction,
                sl_price=self.trade_plan.sl_price,
                tp_price=self.trade_plan.tp_price,
                same_bar_rule=self.same_bar_rule
            )

            if exit_event is not None:
                entry_price = float(self.trade_plan.entry_price)
                exit_price = float(exit_event.exit_price)
                qty = float(self.trade_plan.quantity)

                pnl = (exit_price - entry_price) * qty if self.trade_plan.direction == PositionDirection.LONG \
                      else (entry_price - exit_price) * qty

                self.trades.append({
                    "entry_time": self.bars_15m[self.trade_plan.entry_candle_index].time,
                    "exit_time": bar.time,
                    "direction": self.trade_plan.direction.value,
                    "entry_price": entry_price,
                    "sl_price": float(self.trade_plan.sl_price),
                    "tp_price": float(self.trade_plan.tp_price),
                    "exit_price": exit_price,
                    "exit_reason": exit_event.exit_reason.value,
                    "qty": qty,
                    "pnl": float(pnl),
                    "signal_index": int(self.trade_plan.signal_candle_index),
                    "entry_index": int(self.trade_plan.entry_candle_index),
                    "exit_index": int(t),
                })

                self.stat_exit += 1

                self.sim_equity += float(pnl)
                self.Plot("Sim", "Equity", self.sim_equity)

                # Reset state
                self.state = "FLAT"
                self.trade_plan = None
                self.stop_loss_manager.reset()

                if self.cooldown_bars and self.cooldown_bars > 0:
                    self._cooldown_until = self._bar_index + self.cooldown_bars

                return

        # ----------------------------
        # 3) Entry (streaming fix: evaluate BOS on t-1, enter on Open[t])
        # ----------------------------
        if self.state != "FLAT":
            return

        if self.max_trades_per_day is not None and self._trades_today >= self.max_trades_per_day:
            return
        if self._bar_index < self._cooldown_until:
            return

        signal_t = t - 1
        if signal_t < 0:
            return

        bos_signal = detect_bos_signal(
            bars=self.bars_15m,
            t=signal_t,
            swing_levels=self.swing_levels
        )
        if bos_signal is None:
            return

        self.stat_bos += 1

        # RANGE_BASED requires a valid stored range
        if self.tp_mode == TakeProfitMode.RANGE_BASED:
            if self.last_valid_range is None:
                self.stat_plan_fail += 1
                if self.stat_fail_logged < 30:
                    hi = self.swing_levels.last_swing_high_price
                    lo = self.swing_levels.last_swing_low_price
                    self.Debug(f"PLAN SKIP {tb.EndTime} | RANGE_BASED no valid range yet (hi={hi}, lo={lo})")
                    self.stat_fail_logged += 1
                return

        # Build swing snapshot used for TP calculation inside plan_trade_from_signal
        effective_swing_levels = self.swing_levels
        if self.tp_mode == TakeProfitMode.RANGE_BASED and self.last_valid_range is not None:
            hi, lo = self.last_valid_range
            effective_swing_levels = SwingLevels(last_swing_high_price=hi, last_swing_low_price=lo)



        # StopLossManager safety: always reset before planning and on failure
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
                buying_power_cash=None,
                position_sizer=self._position_sizer_fractional
            )
        except Exception as e:
            self.stop_loss_manager.reset()
            self.stat_plan_fail += 1

            msg = str(e)
            if "sized quantity is zero" in msg or "quantity is zero" in msg:
                self.stat_skip_qty0 += 1

            # Log first 30 failures for diagnosis
            if self.stat_fail_logged < 30:
                self.Debug(f"PLAN FAIL {tb.EndTime} | {e}")
                self.stat_fail_logged += 1

            return

        # Accept plan
        self.trade_plan = plan
        self.state = "LONG" if plan.direction == PositionDirection.LONG else "SHORT"
        self._trades_today += 1
        self.stat_plan_ok += 1

        if self.cooldown_bars and self.cooldown_bars > 0:
            self._cooldown_until = self._bar_index + self.cooldown_bars

    def OnEndOfAlgorithm(self):
        n = len(self.trades)
        total_pnl = sum(t["pnl"] for t in self.trades) if n else 0.0

        self.Debug(
            f"DONE | Trades={n} | TotalPnL(sim)={total_pnl:.2f} | SimEquity={self.sim_equity:.2f} | "
            f"BOS={self.stat_bos} PlanOK={self.stat_plan_ok} PlanFail={self.stat_plan_fail} "
            f"Exits={self.stat_exit} Qty0Skips={self.stat_skip_qty0}"
        )

        for i, tr in enumerate(self.trades[:5]):
            self.Debug(f"Trade[{i}]: {tr}")
