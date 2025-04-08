from datamodel import OrderDepth, Order, UserId, TradingState
from typing import List
import string


class Product:
    RESIN = "RAINFOREST_RESIN"
    INK = "SQUID_INK"
    KELP = "KELP"


class Trader:
    def __init__(self):
        self.LIMIT = {
            Product.RESIN: 20,
            Product.INK: 20,
            Product.KELP: 20
        }
        
    def run(self, state: TradingState):
        pass