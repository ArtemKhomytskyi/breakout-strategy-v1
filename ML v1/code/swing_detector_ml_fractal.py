from __future__ import annotations

"""Online ML swing detector driven by fractal-trained model artifacts.

This module bridges offline training outputs and bar-by-bar runtime usage.
The detector scores each incoming bar, stores the prediction until the
right-side confirmation delay has elapsed, and then emits confirmed swing
events through ``pop_confirmed()``.

Runtime contract:
- probabilities are produced at bar ``t``
- a swing candidate at ``t`` becomes confirmable only when ``t + right`` is seen
- external feature rows are preferred because they match training-time features
- a fallback OHLC-only feature builder exists for quick experiments, but it is
  only an approximation of the training pipeline
"""

import json
import math
import pickle
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConfirmedSwing:
    """Confirmed swing event.

    Attributes:
        index: integer bar index of the swing bar t (confirmed at t + right)
        highlow: +1 swing high, -1 swing low
        level: price level (high for swing high, low for swing low)
        p_high: P(swing_high) at bar t
        p_low:  P(swing_low)  at bar t
    """

    index: int
    highlow: int
    level: float
    p_high: float
    p_low: float


class SwingDetectorMLFractal:
    """ML swing detector trained on fractal labels (L/R).

    Online contract:
    - update(bar, features_row=None): update with a new bar.
      If features_row is provided (pd.Series or dict), uses it directly for inference.
      Otherwise falls back to computing simple features from OHLC (may NOT match training).
    - predict(): last (p_high, p_low)
    - pop_confirmed(): list of ConfirmedSwing confirmed since last call

    Notes:
    - Strict no-lookahead: swing at t is confirmed only when bar t+right is processed.
    - Uses feature_config.json to define feature order. If config includes segment_id
      but model expects fewer features, segment_id is auto-dropped at runtime.
    """

    def __init__(
        self,
        *,
        model_high_path: str,
        model_low_path: str,
        feature_config_path: str,
        thresholds_path: str,
        min_bars_between_swings: int = 3,
        eps: float = 1e-12,
    ):
        self.model_high = self._load_model(model_high_path)
        self.model_low = self._load_model(model_low_path)

        cfg = self._load_json(feature_config_path)
        feat_names = cfg.get("features") or cfg.get("feature_names") or cfg.get("columns")
        if not feat_names:
            raise ValueError(f"feature_config has no feature list: {feature_config_path}")
        self.feature_names: List[str] = list(feat_names)

        th = self._load_json(thresholds_path)
        self.threshold_high = float(th["threshold_high"])
        self.threshold_low = float(th["threshold_low"])
        self.left = int(th.get("left", 10))
        self.right = int(th.get("right", 10))

        self.min_bars_between_swings = int(min_bars_between_swings)
        self.eps = float(eps)

        self._align_feature_names_to_model()

        self._opens: Deque[float] = deque(maxlen=200)
        self._highs: Deque[float] = deque(maxlen=200)
        self._lows: Deque[float] = deque(maxlen=200)
        self._closes: Deque[float] = deque(maxlen=200)

        self._bar_index = -1
        self._last_p_high = float("nan")
        self._last_p_low = float("nan")
        # Buffer the last right+1 bars so the oldest entry becomes confirmable
        # exactly when the current bar closes the confirmation window.
        self._buffer: Deque[Dict[str, float]] = deque(maxlen=self.right + 1)

        self._last_confirmed_swing_index = -10**9
        self._confirmed_queue: Deque[ConfirmedSwing] = deque()

        self.last_swing_high_price: Optional[float] = None
        self.last_swing_low_price: Optional[float] = None

    def reset(self) -> None:
        """Reset state (useful for notebook experiments)."""
        self._opens.clear()
        self._highs.clear()
        self._lows.clear()
        self._closes.clear()
        self._buffer.clear()
        self._confirmed_queue.clear()
        self._bar_index = -1
        self._last_p_high = float("nan")
        self._last_p_low = float("nan")
        self._last_confirmed_swing_index = -10**9
        self.last_swing_high_price = None
        self.last_swing_low_price = None

    def update(self, bar: Any, *, features_row: Optional[Any] = None) -> None:
        """Process a new bar.

        bar must have: open, high, low, close.
        features_row: pd.Series or dict with at least self.feature_names (recommended).
        """
        self._bar_index += 1

        o = float(bar.open)
        h = float(bar.high)
        l = float(bar.low)
        c = float(bar.close)

        self._opens.append(o)
        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)

        if features_row is not None:
            x = self._build_from_features_row(features_row)
        else:
            if not self._has_min_history_for_fallback():
                self._last_p_high = float("nan")
                self._last_p_low = float("nan")
                self._buffer.append(
                    {"p_high": float("nan"), "p_low": float("nan"), "high": h, "low": l}
                )
                return
            x = self._build_from_ohlc_fallback()

        self._last_p_high = self._predict_proba_safe(self.model_high, x, self.feature_names)
        self._last_p_low = self._predict_proba_safe(self.model_low, x, self.feature_names)

        self._buffer.append({"p_high": self._last_p_high, "p_low": self._last_p_low, "high": h, "low": l})

        if len(self._buffer) < (self.right + 1):
            return

        swing_index = self._bar_index - self.right
        # The leftmost buffered item corresponds to the bar that has just
        # matured past the right-side confirmation delay.
        candidate = self._buffer[0]

        self._try_confirm(
            swing_index=swing_index,
            p_high=float(candidate["p_high"]),
            p_low=float(candidate["p_low"]),
            high=float(candidate["high"]),
            low=float(candidate["low"]),
        )

    def predict(self) -> Tuple[float, float]:
        return self._last_p_high, self._last_p_low

    def pop_confirmed(self) -> List[ConfirmedSwing]:
        out = list(self._confirmed_queue)
        self._confirmed_queue.clear()
        return out

    @staticmethod
    def _load_json(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_model(path: str):
        """Prefer joblib for sklearn persistence, fallback to pickle."""
        try:
            import joblib

            return joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f)

    def _align_feature_names_to_model(self) -> None:
        def get_n_features(model) -> Optional[int]:
            if hasattr(model, "n_features_in_"):
                return int(model.n_features_in_)  # type: ignore
            if hasattr(model, "estimator") and hasattr(model.estimator, "n_features_in_"):
                return int(model.estimator.n_features_in_)  # type: ignore
            return None

        n_hi = get_n_features(self.model_high)
        n_lo = get_n_features(self.model_low)
        n_exp = None
        if n_hi is not None and n_lo is not None:
            n_exp = min(n_hi, n_lo)
        elif n_hi is not None:
            n_exp = n_hi
        elif n_lo is not None:
            n_exp = n_lo

        if n_exp is None:
            return

        if len(self.feature_names) == n_exp:
            return

        # Some research artifacts keep segment_id in the saved feature config
        # even when the trained sklearn model was fit without it.
        if "segment_id" in self.feature_names and len(self.feature_names) - 1 == n_exp:
            self.feature_names = [f for f in self.feature_names if f != "segment_id"]
            return

        raise ValueError(
            f"Feature count mismatch: config has {len(self.feature_names)} features but model expects {n_exp}."
        )

    def _build_from_features_row(self, features_row: Any) -> np.ndarray:
        if isinstance(features_row, pd.Series):
            values = [float(features_row[name]) for name in self.feature_names]
        elif isinstance(features_row, dict):
            values = [float(features_row[name]) for name in self.feature_names]
        else:
            values = [float(getattr(features_row, name)) for name in self.feature_names]

        x = np.array(values, dtype=float).reshape(1, -1)
        x = np.where(np.isfinite(x), x, 0.0)
        return x

    def _has_min_history_for_fallback(self) -> bool:
        return len(self._closes) >= 51 and len(self._highs) >= 20 and len(self._lows) >= 20

    def _build_from_ohlc_fallback(self) -> np.ndarray:
        """Fallback OHLC feature calc. Warning: may not match training pipeline."""
        o = self._opens[-1]
        h = self._highs[-1]
        l = self._lows[-1]
        c = self._closes[-1]

        prev_c = self._closes[-2]
        c_3 = self._closes[-4]

        ret_1 = (c / max(prev_c, self.eps)) - 1.0
        ret_3 = (c / max(c_3, self.eps)) - 1.0

        body = (c - o) / max(c, self.eps)
        upper_wick = (h - max(o, c)) / max(c, self.eps)
        lower_wick = (min(o, c) - l) / max(c, self.eps)

        highs = list(self._highs)
        lows = list(self._lows)
        roll_max_20 = max(highs[-20:])
        roll_min_20 = min(lows[-20:])

        dist_to_roll_max_20 = (c - roll_max_20) / max(c, self.eps)
        dist_to_roll_min_20 = (c - roll_min_20) / max(c, self.eps)

        closes = np.array(self._closes, dtype=float)
        rets = (closes[1:] / np.maximum(closes[:-1], self.eps)) - 1.0
        vol_50 = float(np.std(rets[-50:], ddof=0))

        # Mirror the broad shape of the engineered training features so the
        # detector can still be exercised when an external feature row is
        # unavailable in a notebook or replay test.
        values_map: Dict[str, float] = {
            "ret_1": float(ret_1),
            "ret_3": float(ret_3),
            "body": float(body),
            "upper_wick": float(upper_wick),
            "lower_wick": float(lower_wick),
            "dist_to_roll_max_20": float(dist_to_roll_max_20),
            "dist_to_roll_min_20": float(dist_to_roll_min_20),
            "vol_50": float(vol_50),
            "segment_id": 0.0,
        }

        values = [values_map[name] for name in self.feature_names]
        x = np.array(values, dtype=float).reshape(1, -1)
        x = np.where(np.isfinite(x), x, 0.0)
        return x

    @staticmethod
    def _predict_proba_safe(model, x: np.ndarray, feature_names: List[str]) -> float:
        if not hasattr(model, "predict_proba"):
            raise ValueError("Model has no predict_proba(). Expected sklearn-like classifier.")
        X = pd.DataFrame(x, columns=feature_names)
        p = model.predict_proba(X)
        return float(p[0, 1])

    def _try_confirm(self, *, swing_index: int, p_high: float, p_low: float, high: float, low: float) -> None:
        if swing_index - self._last_confirmed_swing_index < self.min_bars_between_swings:
            return

        is_high = math.isfinite(p_high) and p_high >= self.threshold_high
        is_low = math.isfinite(p_low) and p_low >= self.threshold_low

        if not is_high and not is_low:
            return

        # If both sides clear threshold on the same bar, keep only the side
        # with the larger margin above its threshold.
        if is_high and is_low:
            if (p_high - self.threshold_high) >= (p_low - self.threshold_low):
                is_low = False
            else:
                is_high = False

        if is_high:
            self.last_swing_high_price = float(high)
            self._last_confirmed_swing_index = swing_index
            self._confirmed_queue.append(
                ConfirmedSwing(
                    index=swing_index,
                    highlow=1,
                    level=float(high),
                    p_high=float(p_high),
                    p_low=float(p_low),
                )
            )
            return

        if is_low:
            self.last_swing_low_price = float(low)
            self._last_confirmed_swing_index = swing_index
            self._confirmed_queue.append(
                ConfirmedSwing(
                    index=swing_index,
                    highlow=-1,
                    level=float(low),
                    p_high=float(p_high),
                    p_low=float(p_low),
                )
            )
