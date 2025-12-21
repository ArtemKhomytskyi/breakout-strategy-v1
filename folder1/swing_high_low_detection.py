# FILE: swing_high_low_detection.py
from AlgorithmImports import *
import pandas as pd
import numpy as np

# This CLASS wrapper is what was missing
class SwingDetector:
    def __init__(self, period=20):
        self.period = period
        self.history = pd.DataFrame(columns=['open', 'high', 'low', 'close'])
        self.last_high = 0
        self.last_low = 0

    def update(self, bar):
        # 1. Save new data
        new_row = pd.DataFrame({
            'open': [bar.Open], 'high': [bar.High], 
            'low': [bar.Low], 'close': [bar.Close]
        }, index=[bar.Time])
        self.history = pd.concat([self.history, new_row]).tail(100)
        
        # 2. Run logic (Simplified for stability)
        if len(self.history) >= self.period:
            closes = self.history['close'].values
            if len(closes) < 5: return
            
            # Simple Swing Logic to get you started
            # (Checks if 3 bars ago was a local high/low)
            candidate = closes[-4]
            if candidate == max(closes[-7:-1]): # Swing High
                self.last_high = self.history['high'].values[-4]
            elif candidate == min(closes[-7:-1]): # Swing Low
                self.last_low = self.history['low'].values[-4]

    def get_last_high(self):
        return self.last_high

    def get_last_low(self):
        return self.last_low