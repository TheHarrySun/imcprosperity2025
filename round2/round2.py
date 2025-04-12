from typing import List, Tuple
import string
import math
import numpy as np
import json
import jsonpickle
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()







class Product:
    RESIN = "RAINFOREST_RESIN"
    INK = "SQUID_INK"
    KELP = "KELP"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PIC1 = "PICNIC_BASKET1"
    PIC2 = "PICNIC_BASKET2"

PARAMS = {
    Product.RESIN: {
        "fair_value": 10000,
        "take_width": 1
    },
    Product.INK: {
        # "take_width": 0.5,
        "prevent_adverse": True,
        "adverse_volume": 20
    },
    Product.KELP: {
        "take_width": 1,
        "prevent_adverse": True,
        "adverse_volume": 20
    },
    Product.PIC1: {
        "num_croissants": 6,
        "num_jams": 3,
        "num_djembes": 1
    },
    Product.PIC2: {
        "num_croissants": 4,
        "num_jams": 2,
        "num_djembes": 0
    },
    Product.CROISSANTS: {
        "take_width": 1,
        "prevent_adverse": True,
        "adverse_volume": 25
    },
    Product.JAMS: {
        "take_width": 1,
        "prevent_adverse": True,
        "adverse_volume": 25
    },
    Product.DJEMBES: {
        "take_width": 1,
        "prevent_adverse": True,
        "adverse_volume": 25
    }
}

class Trader:
    def __init__(self):
        self.LIMIT = {
            Product.RESIN: 50,
            Product.INK: 50,
            Product.KELP: 50,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.PIC1: 60,
            Product.PIC2: 100
        }
        
    def take_best_orders(
        self, 
        product: str, 
        fair_value: float, 
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
                            orders.append(Order(product,  int(round(best_ask)), quantity))
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
                            orders.append(Order(product, int(round(best_ask)), quantity))
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
                            orders.append(Order(product, int(round(best_bid)), -1 * quantity))
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
                            orders.append(Order(product, int(round(best_bid)), -1 * quantity))
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
        bid: float, 
        ask: float, 
        position: int, 
        buy_order_volume: int, 
        sell_order_volume: int
    ) -> Tuple[int, int]:
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if (buy_quantity > 0):
            orders.append(Order(product, int(round(bid)), buy_quantity))
        
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, int(round(ask)), -sell_quantity))
            
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
                orders.append(Order(product, int(round(fair_for_ask)), -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
                
        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_ask]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, int(round(fair_for_bid)), abs(sent_quantity)))
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
        if (fair_value == None):
            return orders
        buy_order_volume = 0
        sell_order_volume = 0
        product = Product.RESIN
        
        fair_ask = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        fair_ask = fair_value + 1 if len(fair_ask) == 0 else min(fair_ask)
        fair_bid = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        fair_bid = fair_value - 1 if len(fair_bid) == 0 else min(fair_bid)
        buy_order_volume, sell_order_volume = self.take_best_orders(product, fair_value, width, orders, order_depth, position, buy_order_volume, sell_order_volume)
        buy_order_volume, sell_order_volume = self.clear_position_order(product, fair_value, orders, order_depth, position, buy_order_volume, sell_order_volume)
        buy_order_volume, sell_order_volume = self.market_make(product, orders, fair_bid, fair_ask, position, buy_order_volume, sell_order_volume)
        
        return orders
    
    def mm_fair_value(self, product: str, order_depth: OrderDepth) -> float:
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
    
    def ink_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= PARAMS[Product.INK]["adverse_volume"]]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= PARAMS[Product.INK]["adverse_volume"]]
            
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid
            
            if mm_ask == None or mm_bid == None:
                if traderObject.get("ink_last_price", None) == None:
                    mmid_price = (best_ask + best_bid) / 2
                else:
                    mmid_price = traderObject["ink_last_price"]
            else:
                mmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("ink_last_price", None) != None:
                last_price = traderObject["ink_last_price"]
                last_returns = (mmid_price - last_price) / last_price
                pred_returns = last_returns * (-0.229)
                fair = mmid_price + (mmid_price * pred_returns)
            else:
                fair = mmid_price
            traderObject["ink_last_price"] = mmid_price
            return fair
        return None
    # tried using the weighted fair values and it didn't seem to perform as well as mid price
    
    def weight_fair_value(self, order_depth: OrderDepth) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            
            ask_vol = abs(order_depth.sell_orders[best_ask])
            bid_vol = abs(order_depth.buy_orders[best_bid])
            volume = ask_vol + bid_vol
            
            vwap = (best_bid * ask_vol + best_ask * bid_vol) / volume
            return vwap
        return None
     

    def ink_orders(
        self, 
        order_depth: OrderDepth, 
        fair_value: int, 
        position: int
    ) -> List[Order]:
        
        orders: List[Order] = []
        if (fair_value == None):
            return orders
        buy_order_volume = 0
        sell_order_volume = 0
        product = Product.INK
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        baaf = min(aaf) if len(aaf) > 0 else fair_value + 1
        bbbf = max(bbf) if len(bbf) > 0 else fair_value - 1
        
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        volatility = abs(best_ask - best_bid)
        dynamic_take_width = max(0.5, volatility * 0.25)
        
        buy_order_volume, sell_order_volume = self.take_best_orders(product, fair_value, dynamic_take_width, orders, order_depth, position, buy_order_volume, sell_order_volume, PARAMS[product]["prevent_adverse"], PARAMS[product]["adverse_volume"])
        buy_order_volume, sell_order_volume = self.clear_position_order(product, fair_value, orders, order_depth, position, buy_order_volume, sell_order_volume)
        buy_order_volume, sell_order_volume = self.market_make(product, orders, bbbf + 0.5, baaf - 0.5, position, buy_order_volume, sell_order_volume)
        
        return orders
    
    def kelp_orders(
        self, 
        order_depth: OrderDepth, 
        fair_value: int, 
        position: int
    ) -> List[Order]:
        
        orders: List[Order] = []
        if (fair_value == None):
            return orders
        buy_order_volume = 0
        sell_order_volume = 0
        product = Product.KELP
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
        bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2
        
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        volatility = abs(best_ask - best_bid)
        dynamic_take_width = max(1, volatility * 0.25)
        
        buy_order_volume, sell_order_volume = self.take_best_orders(product, fair_value, PARAMS[Product.KELP]["take_width"], orders, order_depth, position, buy_order_volume, sell_order_volume, PARAMS[product]["prevent_adverse"], PARAMS[product]["adverse_volume"])
        buy_order_volume, sell_order_volume = self.clear_position_order(product, fair_value, orders, order_depth, position, buy_order_volume, sell_order_volume)
        buy_order_volume, sell_order_volume = self.market_make(product, orders, bbbf + 0.5, baaf - 0.5, position, buy_order_volume, sell_order_volume)
        
        return orders
    
    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        result = {}
        
        if Product.RESIN in state.order_depths:
            resin_position = state.position[Product.RESIN] if Product.RESIN in state.position else 0
            resin_orders = self.resin_orders(state.order_depths[Product.RESIN], PARAMS[Product.RESIN]["fair_value"], PARAMS[Product.RESIN]["take_width"], resin_position)
            result[Product.RESIN] = resin_orders
                    
        
        if Product.INK in state.order_depths:
            ink_position = state.position[Product.INK] if Product.INK in state.position else 0
            ink_fair_value = self.mm_fair_value(Product.INK, state.order_depths[Product.INK])
            # ink_fair_value = self.weight_fair_value(state.order_depths[Product.INK])            

            ink_orders = self.ink_orders(state.order_depths[Product.INK], ink_fair_value, ink_position)
            result[Product.INK] = ink_orders
        
        if Product.KELP in state.order_depths:
            kelp_position = state.position[Product.KELP] if Product.KELP in state.position else 0
            kelp_fair_value = self.mm_fair_value(Product.KELP, state.order_depths[Product.KELP])
            # kelp_fair_value = self.weight_fair_value(Product.KELP, state.order_depths)
            # kelp_fair_value = self.ink_fair_value(state.order_depths[Product.KELP])

            
            kelp_orders = self.kelp_orders(state.order_depths[Product.KELP], kelp_fair_value, kelp_position)
            result[Product.KELP] = kelp_orders
            
                        
        traderData = jsonpickle.encode(traderObject) # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        
        logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData