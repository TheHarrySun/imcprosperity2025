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
            Product.RESIN: 50,
            Product.INK: 50,
            Product.KELP: 50
        }
        
    def take_best_orders(self, product: str, fair_value: int, take_width: float, orders: List[Order], order_depth: OrderDepth):
        prod_limit = self.LIMIT[product]
        
        
    def run(self, state: TradingState):
        pass