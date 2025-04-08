from datamodel import OrderDepth, Order, UserId, TradingState
from typing import List
import string


class Product:
    RESIN = "RAINFOREST_RESIN"
    INK = "SQUID_INK"
    KELP = "KELP"

class Trader:
    def run(self, state: TradingState):
        pass