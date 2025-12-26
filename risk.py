from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from entry_exit import PositionDirection


@dataclass(frozen=True)
class RiskConfig:
    """
    Risk configuration for sizing positions.

    Attributes:
        risk_budget_cash: how much cash you are willing to lose on the trade.
        max_quantity: optional hard cap on quantity (float to allow fractional assets).
        min_risk_per_unit: optional minimal stop distance; if stop is tighter, skip the trade.
        use_buying_power_cap: if True, cap budget by provided buying_power_cash in size_position.
    """
    risk_budget_cash: float
    max_quantity: Optional[float] = None
    min_risk_per_unit: Optional[float] = None
    use_buying_power_cap: bool = False


def size_position(
    *,
    direction: "PositionDirection",
    entry_price: float,
    sl_price: float,
    risk_config: RiskConfig,
    buying_power_cash: Optional[float] = None,
    round_func: Callable[[float], float] = math.floor,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Compute position quantity based on risk budget and stop distance.

    Hard failures -> ValueError (broken data).
    Soft refusals -> (None, reason).
    """
    if entry_price <= 0:
        raise ValueError(f"entry_price must be > 0, got {entry_price}")
    if sl_price <= 0:
        raise ValueError(f"sl_price must be > 0, got {sl_price}")
    if risk_config.risk_budget_cash <= 0:
        raise ValueError(f"risk_budget_cash must be > 0, got {risk_config.risk_budget_cash}")

    risk_per_unit = abs(entry_price - sl_price)
    if risk_per_unit <= 0:
        raise ValueError("risk_per_unit must be > 0 (entry and SL cannot coincide).")

    if risk_config.min_risk_per_unit is not None and risk_per_unit < risk_config.min_risk_per_unit:
        return None, "risk_per_unit below min_risk_per_unit"

    budget = risk_config.risk_budget_cash
    if risk_config.use_buying_power_cap and buying_power_cash is not None:
        budget = min(budget, buying_power_cash)

    if budget <= 0:
        return None, "no budget available"

    raw_qty = budget / risk_per_unit
    qty = round_func(raw_qty)

    if risk_config.max_quantity is not None:
        qty = min(qty, risk_config.max_quantity)

    if qty <= 0:
        return None, "sized quantity is zero after caps"

    return float(qty), None
