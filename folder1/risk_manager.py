# FILE: risk_manager.py
# region imports
from AlgorithmImports import *
# endregion
import math

class RiskManager:
    """
    Calculates position size based on CURRENT capital and risk percentage.
    """
    def __init__(self, risk_per_trade_pct: float = 0.01):
        """
        Updated Init: NO total_capital here anymore.
        Args:
            risk_per_trade_pct: How much we are willing to lose per trade (0.01 = 1%).
        """
        self.risk_per_trade_pct = risk_per_trade_pct

    def calculate_quantity(self, entry_price: float, sl_price: float, current_total_capital: float) -> int:
        """
        Updated Calculation: We ask for current_total_capital HERE.
        """
        # 1. Calculate the dollar amount we are willing to lose based on CURRENT money
        risk_dollars = current_total_capital * self.risk_per_trade_pct
        
        # 2. Calculate risk per share
        risk_per_share = abs(entry_price - sl_price)
        
        if risk_per_share <= 0:
            return 0
            
        # 3. Calculate quantity
        quantity = math.floor(risk_dollars / risk_per_share)
        
        return int(quantity)