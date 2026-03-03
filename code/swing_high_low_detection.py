# region imports
from AlgorithmImports import *
# endregion
from collections import deque
import pandas as pd
import numpy as np

def swing_highs_lows_online(
    ohlc: pd.DataFrame,
    N_candidates: list = [5, 10, 20, 50],
    N_confirmation: int = 3,
    min_move_threshold: float = 0.0,
    min_bars_between_swings: int = 3
) -> pd.DataFrame:
    """
    Online swing high/low detection with confirmation (no repainting).

    Parameters:
    -----------
    ohlc : pd.DataFrame with 'close', 'high', 'low'
    N_candidates : list of int - windows to detect candidate swings
    N_confirmation : int - number of future bars to confirm candidate
    min_move_threshold : float - minimal move in % to accept swing
    min_bars_between_swings : int - minimal bars between consecutive swings

    Returns:
    --------
    pd.DataFrame with columns:
        'HighLow' : 1 for swing high, -1 for swing low, NaN otherwise
        'Level'   : price level of the swing
    """
    
    swings = pd.DataFrame(index=ohlc.index, columns=["HighLow", "Level"], dtype=float)
    last_swing_index = -min_bars_between_swings - 1  # initialize for spacing
    
    # For each candidate window size
    for N in N_candidates:
        window = N
        closes = ohlc['close'].values
        highs = ohlc['high'].values
        lows = ohlc['low'].values
        
        # deque to hold future bars for confirmation
        future_window = deque(maxlen=N_confirmation)
        
        for i in range(len(ohlc)):
            # update future_window
            future_window.append(closes[i])
            
            # only start confirming when we have enough bars
            if len(future_window) < N_confirmation:
                continue
            
            # candidate index to check
            idx = i - N_confirmation
            if idx < 0 or (idx - last_swing_index) < min_bars_between_swings:
                continue
            
            # get window for candidate
            left = max(0, idx - window)
            right = idx + 1  # do not include future bars beyond candidate
            candidate_close = closes[idx]
            window_values = closes[left:right]
            
            # check swing high
            if candidate_close == max(window_values):
                # optional min_move_threshold check
                if min_move_threshold == 0 or (candidate_close - min(window_values)) / candidate_close >= min_move_threshold:
                    swings.at[ohlc.index[idx], "HighLow"] = 1
                    swings.at[ohlc.index[idx], "Level"] = highs[idx]
                    last_swing_index = idx
            
            # check swing low
            elif candidate_close == min(window_values):
                if min_move_threshold == 0 or (max(window_values) - candidate_close) / candidate_close >= min_move_threshold:
                    swings.at[ohlc.index[idx], "HighLow"] = -1
                    swings.at[ohlc.index[idx], "Level"] = lows[idx]
                    last_swing_index = idx
                    
    return swings