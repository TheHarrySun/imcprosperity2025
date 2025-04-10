from datamodel import OrderDepth, Order, UserId, TradingState
from typing import List, Tuple
import string
import math

class Product:
    RESIN = "RAINFOREST_RESIN"
    INK = "SQUID_INK"
    KELP = "KELP"

PARAMS = {
    Product.RESIN: {
        "fair_value": 10000,
        "take_width": 1
    },
    Product.INK: {
        "take_width": 2,
        "prevent_adverse": True,
        "adverse_volume": 15
    },
    Product.KELP: {
        "take_width": 2,
        "prevent_adverse": True,
        "adverse_volume": 15
    }
}

class Trader:
    def __init__(self):
        self.LIMIT = {
            Product.RESIN: 50,
            Product.INK: 50,
            Product.KELP: 50
        }
        
    def take_best_orders(
        self, 
        product: str, 
        fair_value: int, 
        take_width: float, 
        orders: List[Order], 
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0
    ) -> Tuple[int, int]:
        prod_limit = self.LIMIT[product]
        
        if len(order_depth.sell_orders) != 0:
            while (True):
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_amt = -1 * order_depth.sell_orders[best_ask]
                if (prevent_adverse):
                    if best_ask <= fair_value - take_width:
                        quantity = min(adverse_volume, min(best_ask_amt, prod_limit - position))
                        if quantity > 0:
                            orders.append(Order(product, best_ask, quantity))
                            buy_order_volume += quantity
                            order_depth.sell_orders[best_ask] += quantity
                            if order_depth.sell_orders[best_ask] == 0:
                                del order_depth.sell_orders[best_ask]
                            if buy_order_volume > adverse_volume:
                                break
                        else:
                            break
                    else: 
                        break                    
                else:
                    if (best_ask <= fair_value - take_width):
                        quantity = min(best_ask_amt, prod_limit - position)
                        if quantity > 0:
                            orders.append(Order(product, best_ask, quantity))
                            buy_order_volume += quantity
                            order_depth.sell_orders[best_ask] += quantity
                            if order_depth.sell_orders[best_ask] == 0:
                                del order_depth.sell_orders[best_ask]
                        else:
                            break
                    else:
                        break
            
        if len(order_depth.buy_orders) != 0:
            while (True):
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_amt = order_depth.buy_orders[best_bid]
                if prevent_adverse:
                    if best_bid >= fair_value + take_width:
                        quantity = min(adverse_volume, min(best_bid_amt, prod_limit + position))
                        if quantity > 0:
                            orders.append(Order(product, best_bid, -1 * quantity))
                            sell_order_volume += quantity
                            order_depth.buy_orders[best_bid] += quantity
                            if order_depth.buy_orders[best_bid] == 0:
                                del order_depth.buy_orders[best_bid]
                            if sell_order_volume > adverse_volume:
                                break
                        else:
                            break 
                    else:
                        break   
                        

                else:
                    if best_bid >= fair_value + take_width:
                        quantity = min(best_bid_amt, prod_limit + position)

                        if quantity > 0:
                            orders.append(Order(product, best_bid, -1 * quantity))
                            sell_order_volume += quantity
                            order_depth.buy_orders[best_bid] -= quantity
                            if order_depth.buy_orders[best_bid] == 0:
                                del order_depth.buy_orders[best_bid]
                        else:
                            break
                    else:
                        break
        
        return buy_order_volume, sell_order_volume
        
    def market_make(
        self, 
        product: str, 
        orders: List[Order], 
        bid: int, 
        ask: int, 
        position: int, 
        buy_order_volume: int, 
        sell_order_volume: int
    ) -> Tuple[int, int]:
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if (buy_quantity > 0):
            orders.append(Order(product, bid, buy_quantity))
        
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, ask, -sell_quantity))
            
        return buy_order_volume, sell_order_volume
    
    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int
    ) -> Tuple[int, int]:
        
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.floor(fair_value)
        
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        
        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
                
        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_ask]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        
        return buy_order_volume, sell_order_volume
    
    def resin_orders(
        self, 
        order_depth: OrderDepth, 
        fair_value: int, 
        width: int, 
        position: int
    ) -> List[Order]:
        
        orders: List[Order] = []
        
        buy_order_volume = 0
        sell_order_volume = 0
        product = Product.RESIN
        
        fair_ask = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        fair_bid = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])
        buy_order_volume, sell_order_volume = self.take_best_orders(product, fair_value, width, orders, order_depth, position, buy_order_volume, sell_order_volume)
        buy_order_volume, sell_order_volume = self.clear_position_order(product, fair_value, orders, order_depth, position, buy_order_volume, sell_order_volume)
        buy_order_volume, sell_order_volume = self.market_make(product, orders, fair_bid, fair_ask, position, buy_order_volume, sell_order_volume)
        
        return orders
    
    def fair_value(self, product: str, order_depth: OrderDepth) -> float:
        order_depth = order_depth[product]
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= PARAMS[product]["adverse_volume"]]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= PARAMS[product]["adverse_volume"]]
            
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid
            
            mmid_price = (mm_ask + mm_bid) / 2
            return mmid_price
        return None

    def ink_orders(
        self, 
        order_depth: OrderDepth, 
        fair_value: int, 
        width: int,
        position: int
    ) -> List[Order]:
        
        orders: List[Order] = []
        
        buy_order_volume = 0
        sell_order_volume = 0
        product = Product.INK
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
        bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2
        
        buy_order_volume, sell_order_volume = self.take_best_orders(product, fair_value, width, orders, order_depth, position, buy_order_volume, sell_order_volume, PARAMS[product]["prevent_adverse"], PARAMS[product]["adverse_volume"])
        buy_order_volume, sell_order_volume = self.clear_position_order(product, fair_value, orders, order_depth, position, buy_order_volume, sell_order_volume)
        buy_order_volume, sell_order_volume = self.market_make(product, orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume)
        
        return orders
    
    def kelp_orders(
        self, 
        order_depth: OrderDepth, 
        fair_value: int, 
        width: int,
        position: int
    ) -> List[Order]:
        
        orders: List[Order] = []
        
        buy_order_volume = 0
        sell_order_volume = 0
        product = Product.KELP
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
        bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2
        
        buy_order_volume, sell_order_volume = self.take_best_orders(product, fair_value, width, orders, order_depth, position, buy_order_volume, sell_order_volume, PARAMS[product]["prevent_adverse"], PARAMS[product]["adverse_volume"])
        buy_order_volume, sell_order_volume = self.clear_position_order(product, fair_value, orders, order_depth, position, buy_order_volume, sell_order_volume)
        buy_order_volume, sell_order_volume = self.market_make(product, orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume)
        
        return orders
    
    def run(self, state: TradingState):
        result = {}
        
        if Product.RESIN in state.order_depths:
            resin_position = state.position[Product.RESIN] if Product.RESIN in state.position else 0
            resin_orders = self.resin_orders(state.order_depths[Product.RESIN], PARAMS[Product.RESIN]["fair_value"], PARAMS[Product.RESIN]["take_width"], resin_position)
            result[Product.RESIN] = resin_orders
            
        if Product.INK in state.order_depths:
            ink_position = state.position[Product.INK] if Product.INK in state.position else 0
            ink_fair_value = self.fair_value(Product.INK, state.order_depths)
            ink_orders = self.ink_orders(state.order_depths[Product.INK], ink_fair_value, PARAMS[Product.INK]["take_width"], ink_position)
            result[Product.INK] = ink_orders
            
        if Product.KELP in state.order_depths:
            kelp_position = state.position[Product.KELP] if Product.KELP in state.position else 0
            kelp_fair_value = self.fair_value(Product.KELP, state.order_depths)
            kelp_orders = self.kelp_orders(state.order_depths[Product.KELP], kelp_fair_value, PARAMS[Product.KELP]["take_width"], kelp_position)
            result[Product.KELP] = kelp_orders
            
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        
        return result, conversions, traderData