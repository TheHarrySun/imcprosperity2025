from typing import List, Tuple, Dict
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
    SYNTHETIC = "SYNTHEIC"
    SPREAD1 = "SPREAD1"
    SPREAD2 = "SPREAD2"
    ROCK = "VOLCANIC_ROCK"
    COUPON_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    COUPON_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    COUPON_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    COUPON_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    COUPON_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

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
        Product.CROISSANTS: 6,
        Product.JAMS: 3,
        Product.DJEMBES: 1
    },
    Product.PIC2: {
        Product.CROISSANTS: 4,
        Product.JAMS: 2,
        Product.DJEMBES: 0
    },
    Product.CROISSANTS: {
        "take_width": 1,
        "prevent_adverse": True,
        "adverse_volume": 100
    },
    Product.JAMS: {
        "take_width": 1,
        "prevent_adverse": True,
        "adverse_volume": 100
    },
    Product.DJEMBES: {
        "take_width": 1,
        "prevent_adverse": True,
        "adverse_volume": 50
    },
    Product.SPREAD1: {
        "default_spread_mean": 58,
        "spread_std_window": 45,
        "zscore_threshold": 1.5,
        "target_position": 55
    },
    Product.SPREAD2: {
        "default_spread_mean": 40,
        "spread_std_window": 15,
        "zscore_threshold": 1.5,
        "target_position": 60
    },
    Product.ROCK: {
        "delta_hedge_threshold": 20
    },
    Product.COUPON_9500: {
        "mean_vol": 0.02372146,
        "strike": 9500,
        "starting_time_to_expiry": 3 / 7,
        "std_window": 6,
        "zscore_threshold": 20,
        "edge_threshold": 1.5
    },
    Product.COUPON_9750: {
        "mean_vol": 0.0241247,
        "strike": 9750,
        "starting_time_to_expiry": 3 / 7,
        "std_window": 6,
        "zscore_threshold": 20,
        "edge_threshold": 1.5
    },
    Product.COUPON_10000: {
        "mean_vol": 0.02190607,
        "strike": 10000,
        "starting_time_to_expiry": 2 / 7,
        "std_window": 6,
        "zscore_threshold": 20,
        "edge_threshold": 1.5
    },
    Product.COUPON_10250: {
        "mean_vol": 0.0204700,
        "strike": 10250,
        "starting_time_to_expiry": 2 / 7,
        "std_window": 6,
        "zscore_threshold": 20,
        "edge_threshold": 1.5
    },
    Product.COUPON_10500: {
        "mean_vol": 0.020823,
        "strike": 10500,
        "starting_time_to_expiry": 2 / 7,
        "std_window": 6,
        "zscore_threshold": 20,
        "edge_threshold": 1.5
    }
}
# 12k -> 0.5


from math import log, sqrt, exp
from statistics import NormalDist

class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        # print(f"d1: {d1}")
        # print(f"vol: {volatility}")
        # print(f"spot: {spot}")
        # print(f"strike: {strike}")
        # print(f"time: {time_to_expiry}")
        return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility


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
            Product.PIC2: 100,
            Product.ROCK: 400,
            Product.COUPON_9500: 200,
            Product.COUPON_9750: 200,
            Product.COUPON_10000: 200,
            Product.COUPON_10250: 200, 
            Product.COUPON_10500: 200
        }
        self.prev_coeffs = None
        
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
    
    def croissants_orders(
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
        product = Product.CROISSANTS
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value]
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value]
        baaf = min(aaf) if len(aaf) > 0 else fair_value + 1
        bbbf = max(bbf) if len(bbf) > 0 else fair_value - 1
        
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        volatility = abs(best_ask - best_bid)
        dynamic_take_width = max(1, volatility * 0.25)
        
        buy_order_volume, sell_order_volume = self.take_best_orders(product, fair_value, PARAMS[Product.CROISSANTS]["take_width"], orders, order_depth, position, buy_order_volume, sell_order_volume, PARAMS[product]["prevent_adverse"], PARAMS[product]["adverse_volume"])
        buy_order_volume, sell_order_volume = self.clear_position_order(product, fair_value, orders, order_depth, position, buy_order_volume, sell_order_volume)
        buy_order_volume, sell_order_volume = self.market_make(product, orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume)
        
        return orders
    
    def jams_orders(
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
        product = Product.JAMS
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value]
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value]
        baaf = min(aaf) if len(aaf) > 0 else fair_value + 1
        bbbf = max(bbf) if len(bbf) > 0 else fair_value - 1
        
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        volatility = abs(best_ask - best_bid)
        dynamic_take_width = max(1, volatility * 0.25)
        
        buy_order_volume, sell_order_volume = self.take_best_orders(product, fair_value, PARAMS[Product.JAMS]["take_width"], orders, order_depth, position, buy_order_volume, sell_order_volume, PARAMS[product]["prevent_adverse"], PARAMS[product]["adverse_volume"])
        buy_order_volume, sell_order_volume = self.clear_position_order(product, fair_value, orders, order_depth, position, buy_order_volume, sell_order_volume)
        buy_order_volume, sell_order_volume = self.market_make(product, orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume)
        
        return orders
    
    def djembes_orders(
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
        product = Product.DJEMBES
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
        bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2
        
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        volatility = abs(best_ask - best_bid)
        dynamic_take_width = max(1, volatility * 0.25)
        
        buy_order_volume, sell_order_volume = self.take_best_orders(product, fair_value, PARAMS[Product.DJEMBES]["take_width"], orders, order_depth, position, buy_order_volume, sell_order_volume, PARAMS[product]["prevent_adverse"], PARAMS[product]["adverse_volume"])
        buy_order_volume, sell_order_volume = self.clear_position_order(product, fair_value, orders, order_depth, position, buy_order_volume, sell_order_volume)
        buy_order_volume, sell_order_volume = self.market_make(product, orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume)
        
        return orders
    
    
    
    
    
    def get_synthetic_basket_order_depth(
        self, basket_type: str, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # constants
        CROISSANTS_PER_BASKET = PARAMS[basket_type][Product.CROISSANTS]
        JAMS_PER_BASKET = PARAMS[basket_type][Product.JAMS]
        DJEMBES_PER_BASKET = PARAMS[basket_type][Product.DJEMBES]
        
        synthetic_order_prices = OrderDepth()
        
        crois_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders else 0
        )
        
        crois_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float ("inf")
        )
        
        jams_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        
        jams_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float ("inf")
        )
        
        djembes_best_bid = (
            max(order_depths[Product.DJEMBES].buy_orders.keys())
            if order_depths[Product.DJEMBES].buy_orders
            else 0
        )
        
        djembes_best_ask = (
            min(order_depths[Product.DJEMBES].sell_orders.keys())
            if order_depths[Product.DJEMBES].sell_orders
            else float ("inf")
        )
        
        implied_bid = crois_best_bid * CROISSANTS_PER_BASKET + jams_best_bid * JAMS_PER_BASKET + djembes_best_bid * DJEMBES_PER_BASKET
        implied_ask = crois_best_ask * CROISSANTS_PER_BASKET + jams_best_ask * JAMS_PER_BASKET + djembes_best_ask * DJEMBES_PER_BASKET
        
        if implied_bid > 0:
            crois_bid_volume = order_depths[Product.CROISSANTS].buy_orders[crois_best_bid] // CROISSANTS_PER_BASKET
            jams_bid_volume = order_depths[Product.JAMS].buy_orders[jams_best_bid] // JAMS_PER_BASKET
            djembes_bid_volume = order_depths[Product.DJEMBES].buy_orders[djembes_best_bid] // DJEMBES_PER_BASKET if DJEMBES_PER_BASKET != 0 else float('inf')
            
            implied_bid_volume = min(crois_bid_volume, jams_bid_volume, djembes_bid_volume)
            synthetic_order_prices.buy_orders[implied_bid] = implied_bid_volume
        
        if implied_ask < float('inf'):
            crois_ask_volume = order_depths[Product.CROISSANTS].sell_orders[crois_best_ask] // CROISSANTS_PER_BASKET
            jams_ask_volume = order_depths[Product.JAMS].sell_orders[jams_best_ask] // JAMS_PER_BASKET
            djembe_ask_volume = order_depths[Product.DJEMBES].sell_orders[djembes_best_ask] // DJEMBES_PER_BASKET if DJEMBES_PER_BASKET != 0 else 0
            
            implied_ask_volume = max(crois_ask_volume, jams_ask_volume, djembe_ask_volume)
            synthetic_order_prices.sell_orders[implied_ask] = implied_ask_volume
            
        return synthetic_order_prices
    
    def convert_synthetic_basket_orders(
        self, basket_type: str, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
            Product.DJEMBES: []
        }
        
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(basket_type, order_depths)
        best_bid = max(synthetic_basket_order_depth.buy_orders.keys()) if synthetic_basket_order_depth.buy_orders else 0
        best_ask = min(synthetic_basket_order_depth.sell_orders.keys()) if synthetic_basket_order_depth.sell_orders else float('inf')
        
        for order in synthetic_orders:
            price = order.price
            quantity = order.quantity
            
            if quantity > 0 and price >= best_ask:
                crois_price = min(order_depths[Product.CROISSANTS].sell_orders.keys())
                jams_price = min(order_depths[Product.JAMS].sell_orders.keys())
                djembes_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
                
            elif quantity < 0 and price <= best_bid:
                crois_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                jams_price = max(order_depths[Product.JAMS].buy_orders.keys())
                djembes_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                continue
            
            crois_order = Order(Product.CROISSANTS, crois_price, quantity * PARAMS[basket_type][Product.CROISSANTS])
            jams_order = Order(Product.JAMS, jams_price, quantity * PARAMS[basket_type][Product.JAMS])
            djembes_order = Order(Product.DJEMBES, djembes_price, quantity * PARAMS[basket_type][Product.DJEMBES])
            
            component_orders[Product.CROISSANTS].append(crois_order)
            component_orders[Product.JAMS].append(jams_order)
            component_orders[Product.DJEMBES].append(djembes_order)
        
        return component_orders
    
    def execute_spread_orders(
        self, 
        basket_type: str,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth]
    ):
        if target_position == basket_position:
            return None
        
        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[basket_type]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(basket_type, order_depths)
        
        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])
            
            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_order_depth.buy_orders[synthetic_bid_price])
            
            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            
            basket_orders = [Order(basket_type, basket_ask_price, execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)]
            
            aggregate_orders = self.convert_synthetic_basket_orders(basket_type, synthetic_orders, order_depths)
            aggregate_orders[basket_type] = basket_orders
            return aggregate_orders
        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])
            
            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_order_depth.sell_orders[synthetic_ask_price])
            
            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            
            basket_orders=  [Order(basket_type, basket_bid_price, -execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)]
            
            aggregate_orders = self.convert_synthetic_basket_orders(basket_type, synthetic_orders, order_depths)
            aggregate_orders[basket_type] = basket_orders
            return aggregate_orders
        
    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        if (best_ask_vol + best_bid_vol == 0):
            return (best_bid + best_ask) / 2
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_ask_vol + best_bid_vol)
    
    def spread_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        basket_type: str,
        basket_position: int,
        spread_data: Dict[str, Any]        
    ):
        if basket_type not in order_depths.keys():
            return None
        
        basket_order_depth = order_depths[basket_type]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(basket_type, order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)
        spread_type = ""
        if (basket_type == Product.PIC1):
            spread_type = Product.SPREAD1
        elif (basket_type == Product.PIC2):
            spread_type = Product.SPREAD2
        
        if (len(spread_data["spread_history"]) < PARAMS[spread_type]["spread_std_window"]):
            return None
        elif len(spread_data["spread_history"]) > PARAMS[spread_type]["spread_std_window"]:
            spread_data["spread_history"].pop(0)
        
        zscore = 0
        spread_std = np.std(spread_data["spread_history"])
        spread_mean = np.mean(spread_data["spread_history"])
        if (basket_type == Product.PIC1):
            zscore = (spread - PARAMS[spread_type]["default_spread_mean"]) / spread_std
        elif (basket_type == Product.PIC2):
            zscore = (spread - spread_mean) / spread_std
                
        if zscore >= PARAMS[spread_type]["zscore_threshold"]:
            if basket_position != -PARAMS[spread_type]["target_position"]:
                return self.execute_spread_orders(basket_type, -PARAMS[spread_type]["target_position"], basket_position, order_depths)
        
        if zscore <= -PARAMS[spread_type]["zscore_threshold"]:
            if basket_position != PARAMS[spread_type]["target_position"]:
                return self.execute_spread_orders(basket_type, PARAMS[spread_type]["target_position"], basket_position, order_depths)
                
        spread_data["prev_zscore"] = zscore
        return None
    
    def get_coupon_mid_price(self, coupon_order_depth: OrderDepth, traderData: Dict[str, Any]):
        if (len(coupon_order_depth.buy_orders) > 0 and len(coupon_order_depth.sell_orders) > 0):
            best_bid = max(coupon_order_depth.buy_orders.keys())
            best_ask = min(coupon_order_depth.sell_orders.keys())
            traderData["prev_coupon_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return traderData["prev_coupon_price"]
        
    def coupon_orders(self, type: str, coupon_order_depth: OrderDepth, coupon_position: int, traderData: Dict[str, Any], volatility: float) -> List[Order]:
        traderData["past_coupon_vol"].append(volatility)
        
        if len(traderData["past_coupon_vol"]) < PARAMS[type]["std_window"]:
            return None, None
        
        if len(traderData["past_coupon_vol"]) > PARAMS[type]['std_window']:
            traderData["past_coupon_vol"].pop(0)
            
        vol_z_score = (volatility - PARAMS[type]["mean_vol"]) / np.std(traderData["past_coupon_vol"])
        
        if vol_z_score >= PARAMS[type]["zscore_threshold"]:
            if coupon_position != -self.LIMIT[type]:
                target_coupon_position = -self.LIMIT[type]
                if len(coupon_order_depth.buy_orders) > 0:
                    best_bid = max(coupon_order_depth.buy_orders.keys())
                    target_quantity = abs(target_coupon_position - coupon_position)
                    quantity = min(target_quantity, abs(coupon_order_depth.buy_orders[best_bid]))
                    quote_quantity = target_quantity - quantity
                    
                    if quote_quantity == 0:
                        return [Order(type, best_bid, -quantity)], []
                    else:
                        return [Order(type, best_bid, -quantity)], [Order(type, best_bid, -quote_quantity)]
                    
        elif vol_z_score <= PARAMS[type]["zscore_threshold"]:
            if coupon_position != self.LIMIT[type]:
                target_coupon_position = self.LIMIT[type]
                if len(coupon_order_depth.sell_orders) > 0:
                    best_ask = min(coupon_order_depth.sell_orders.keys())
                    target_quantity = abs(target_coupon_position - coupon_position)
                    quantity = min(target_quantity, abs(coupon_order_depth.sell_orders[best_ask]))
                    quote_quantity = target_quantity - quantity
                    if (quote_quantity == 0):
                        return [Order(type, best_ask, quantity)], []
                    else:
                        return [Order(type, best_ask, quantity)], [Order(type, best_ask, quote_quantity)]
                    
        return None, None
    
    def rock_hedge_orders(self, rock_order_depth: OrderDepth, coupon_order_depth: OrderDepth, coupon_orders: List[Order], rock_position: int, coupon_position: int, delta: float):
        if coupon_orders == None or len(coupon_orders) == 0:
            coupon_position_after_trade = coupon_position
        else:
            coupon_position_after_trade = coupon_position + sum(order.quantity for order in coupon_orders)
            
        target_rock_position = -delta * coupon_position_after_trade
        
        if target_rock_position == rock_position:
            return None
        
        target_rock_quantity = target_rock_position - rock_position
        
        orders: List[Order] = []
        if target_rock_quantity > 0:
            best_ask = min(rock_order_depth.sell_orders.keys())
            quantity = min(abs(target_rock_quantity), self.LIMIT[Product.ROCK] - rock_position)
            if quantity > 0:
                orders.append(Order(Product.ROCK, best_ask, round(quantity)))
        elif target_rock_quantity < 0:
            best_bid = max(rock_order_depth.buy_orders.keys())
            quantity = min(abs(target_rock_quantity), self.LIMIT[Product.ROCK] + rock_position)
            if quantity > 0:
                orders.append(Order(Product.ROCK, best_bid, -round(quantity)))
                
        return orders
    
    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        result = {}
        conversions = 0
        
        
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
            
        if Product.SPREAD1 not in traderObject:
            traderObject[Product.SPREAD1] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0
            }
        if Product.SPREAD2 not in traderObject:
            traderObject[Product.SPREAD2] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0
            }

        pic2_position = state.position[Product.PIC2] if Product.PIC2 in state.position else 0        
        pic1_position = state.position[Product.PIC1] if Product.PIC1 in state.position else 0

        pic1_spread_orders = self.spread_orders(state.order_depths, Product.PIC1, pic1_position, traderObject[Product.SPREAD1])
        pic2_spread_orders = self.spread_orders(state.order_depths, Product.PIC2, pic2_position, traderObject[Product.SPREAD2])       
        result[Product.CROISSANTS] = []
        result[Product.JAMS] = []
        result[Product.DJEMBES] = [] 
        if pic1_spread_orders != None:
            result[Product.CROISSANTS].extend(pic1_spread_orders[Product.CROISSANTS])
            result[Product.JAMS].extend(pic1_spread_orders[Product.JAMS])
            result[Product.DJEMBES].extend(pic1_spread_orders[Product.DJEMBES])
            result[Product.PIC1] = pic1_spread_orders[Product.PIC1]
        if pic2_spread_orders != None:
            result[Product.CROISSANTS].extend(pic2_spread_orders[Product.CROISSANTS])
            result[Product.JAMS].extend(pic2_spread_orders[Product.JAMS])
            result[Product.DJEMBES].extend(pic2_spread_orders[Product.DJEMBES])
            result[Product.PIC2] = pic2_spread_orders[Product.PIC2]

        '''
        if Product.CROISSANTS in state.order_depths:
            croissant_position = state.position[Product.CROISSANTS] if Product.CROISSANTS in state.position else 0
            croissant_fair_value = self.mm_fair_value(Product.CROISSANTS, state.order_depths[Product.CROISSANTS])
            croissant_orders = self.croissants_orders(state.order_depths[Product.CROISSANTS], croissant_fair_value, croissant_position)
            result[Product.CROISSANTS].extend(croissant_orders)
        
        if Product.JAMS in state.order_depths:
            jam_position = state.position[Product.JAMS] if Product.JAMS in state.position else 0
            jam_fair_value = self.mm_fair_value(Product.JAMS, state.order_depths[Product.JAMS])
            jam_orders= self.jams_orders(state.order_depths[Product.JAMS], jam_fair_value, jam_position)
            result[Product.JAMS].extend(jam_orders)
            
        if Product.DJEMBES in state.order_depths:
            djembes_position = state.position[Product.DJEMBES] if Product.DJEMBES in state.position else 0
            djembes_fair_value = self.mm_fair_value(Product.DJEMBES, state.order_depths[Product.DJEMBES])
            djembes_orders= self.djembes_orders(state.order_depths[Product.DJEMBES], djembes_fair_value, djembes_position)
            result[Product.DJEMBES].extend(djembes_orders)
        '''         
        
        '''
        vouchers = [Product.COUPON_9500, Product.COUPON_9750, Product.COUPON_10000, Product.COUPON_10250, Product.COUPON_10500]
        
        for coupon in vouchers:
            if coupon not in traderObject:
                traderObject[coupon] = {
                    "prev_coupon_price": 0,
                    "past_coupon_vol": []
                }
            
            if coupon in state.order_depths:
                coupon_position = state.position[coupon] if coupon in state.position else 0
                rock_position = state.position[Product.ROCK] if Product.ROCK in state.position else 0
                
                rock_order_depth = state.order_depths[Product.ROCK]
                coupon_order_depth = state.order_depths[coupon]
                
                rock_mid_price = (min(rock_order_depth.buy_orders.keys()) + max(rock_order_depth.sell_orders.keys())) / 2
                coupon_mid_price = self.get_coupon_mid_price(coupon_order_depth, traderObject[coupon])
                tte = PARAMS[coupon]["starting_time_to_expiry"] - (state.timestamp) / 1000000 / 250
                
                volatility = BlackScholes.implied_volatility(coupon_mid_price, rock_mid_price, PARAMS[coupon]["strike"], tte)
                delta = BlackScholes.delta(rock_mid_price, PARAMS[coupon]["strike"], tte, volatility)
                
                coupon_take_orders, coupon_make_orders = self.coupon_orders(coupon, coupon_order_depth, coupon_position, traderObject[coupon], volatility)
                
                rock_orders = self.rock_hedge_orders(rock_order_depth, coupon_order_depth, coupon_take_orders, rock_position, coupon_position, delta)
                
                if coupon_take_orders != None or coupon_make_orders != None:
                    result[coupon] = coupon_take_orders + coupon_make_orders
                    
                if rock_orders != None:
                    if (Product.ROCK not in result):
                        result[Product.ROCK] = rock_orders
                    else:
                        result[Product.ROCK] += rock_orders
        '''
        
        vouchers = [Product.COUPON_9500, Product.COUPON_9750, Product.COUPON_10000, Product.COUPON_10250, Product.COUPON_10500]
        tte = PARAMS[Product.COUPON_9500]["starting_time_to_expiry"] - (state.timestamp) / 1000000 / 250
        m_list = []
        iv_list = []
        voucher_data = {}
        
        for coupon in vouchers:
            if coupon not in traderObject:
                traderObject[coupon] = {
                    "prev_coupon_price": 0,
                    "past_coupon_vol": []
                }
        
        rock_orders = []
        rock_order_depth = state.order_depths[Product.ROCK]
        for coupon in vouchers:
            coupon_order_depth = state.order_depths[coupon]
            if not coupon_order_depth or not coupon_order_depth.buy_orders or not coupon_order_depth.sell_orders:
                continue
            
            coupon_mid_price = self.get_coupon_mid_price(coupon_order_depth, traderObject[coupon])
            
            bid = max(coupon_order_depth.buy_orders.keys())
            ask = min(coupon_order_depth.sell_orders.keys())
            
            rock_mid_price = (min(rock_order_depth.buy_orders.keys()) + max(rock_order_depth.sell_orders.keys())) / 2
            m = math.log(PARAMS[coupon]["strike"] / rock_mid_price) / math.sqrt(tte)
            iv = BlackScholes.implied_volatility(coupon_mid_price, rock_mid_price, PARAMS[coupon]["strike"], tte)
            
            if math.isnan(iv):
                continue
            m_list.append(m)
            iv_list.append(iv)
            voucher_data[coupon] = {
                "K": PARAMS[coupon]["strike"],
                "m": m,
                "iv": iv,
                "bid": bid,
                "ask": ask,
                "mid": coupon_mid_price
            }
        if len(m_list) >= 3:
            coeffs = list(self._fit_parabola(m_list, iv_list))
            if self.prev_coeffs is None:
                self.prev_coeffs = coeffs
            else:
                self.prev_coeffs = [0.7 * prev + 0.3 * new for prev, new in zip(self.prev_coeffs, coeffs)]
            coeffs = self.prev_coeffs
            net_delta = 0
            
            for coupon, data in voucher_data.items():
                coupon_orders = []
                m = data["m"]
                iv_fit = coeffs[0] * m**2 + coeffs[1] * m + coeffs[2]
                fair_price = BlackScholes.black_scholes_call(rock_mid_price, data["K"], tte, iv_fit)
                delta = BlackScholes.delta(rock_mid_price, data["K"], tte, iv_fit)
                data["delta"] = delta
                
                price_diff = data["mid"] - fair_price
                if (price_diff > PARAMS[coupon]["edge_threshold"]):
                    if (coupon in state.position):
                        size = min(self.LIMIT[coupon], abs(state.position[coupon] + self.LIMIT[coupon]))
                    else:
                        size = self.LIMIT[coupon]
                    coupon_orders.append(Order(coupon, data["bid"], size))
                    net_delta -= delta * size
                elif price_diff < -PARAMS[coupon]["edge_threshold"]:
                    if (coupon in state.position):
                        size = min(self.LIMIT[coupon], abs(state.position[coupon] - self.LIMIT[coupon]))
                    else:
                        size = self.LIMIT[coupon]
                    coupon_orders.append(Order(coupon, data["ask"], -size))
                    net_delta += delta * size
                
                if coupon_orders:
                    result[coupon] = coupon_orders
                    
            if rock_order_depth and abs(net_delta) > PARAMS[Product.ROCK]["delta_hedge_threshold"]:
                hedge_size = round(-net_delta)
                if hedge_size > 0 and rock_order_depth.sell_orders:
                    best_ask = min(rock_order_depth.sell_orders.keys())
                    hedge_size = min(hedge_size, abs(rock_order_depth.sell_orders[best_ask]))
                    rock_orders.append(Order(Product.ROCK, best_ask, hedge_size))
                    rock_order_depth.sell_orders[best_ask] += hedge_size
                    if rock_order_depth.sell_orders[best_ask] == 0:
                        del rock_order_depth.sell_orders[best_ask]
                    
                elif hedge_size < 0 and rock_order_depth.buy_orders:
                    best_bid = max(rock_order_depth.buy_orders.keys())
                    hedge_size = min(abs(hedge_size), abs(rock_order_depth.buy_orders[best_bid]))
                    rock_orders.append(Order(Product.ROCK, best_bid, hedge_size))
                    rock_order_depth.buy_orders[best_bid] -= hedge_size
                    if rock_order_depth.buy_orders[best_bid] == 0:
                        del rock_order_depth.buy_orders[best_bid]
                
                
        if rock_orders:
            result[Product.ROCK] = rock_orders
                              
        
        traderData = jsonpickle.encode(traderObject) # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
                
        logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData
    
    def _fit_parabola(self, x_list, y_list):
        X = [[x ** 2, x, 1] for x in x_list]
        Y = y_list
        Xt = np.transpose(X)
        XtX = np.dot(Xt, X)
        XtY = np.dot(Xt, Y)
        coeffs = np.linalg.solve(XtX, XtY)
        return coeffs