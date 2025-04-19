from typing import List, Tuple, Dict
import string
import math
import numpy as np
import json
import jsonpickle
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, ConversionObservation


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
    MACARON = "MAGNIFICENT_MACARONS"

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
        "default_spread_mean": 30,
        "spread_std_window": 45,
        "zscore_threshold": 1.5,
        "target_position": 70
    },
    Product.ROCK: {
        "trade_threshold": 1
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
        "starting_time_to_expiry": 3 / 7,
        "std_window": 6,
        "zscore_threshold": 20,
        "edge_threshold": 1.5
    },
    Product.COUPON_10250: {
        "mean_vol": 0.0204700,
        "strike": 10250,
        "starting_time_to_expiry": 3 / 7,
        "std_window": 6,
        "zscore_threshold": 20,
        "edge_threshold": 1.5
    },
    Product.COUPON_10500: {
        "mean_vol": 0.020823,
        "strike": 10500,
        "starting_time_to_expiry": 3 / 7,
        "std_window": 6,
        "zscore_threshold": 20,
        "edge_threshold": 1.5
    },
    Product.MACARON: {
        "make_edge": 2,
        "make_min_edge": 1,
        "make_probability": 0.7,
        "init_make_edge": 2,
        "min_edge": 0.5,
        "volume_avg_timestamp": 10,
        "volume_bar": 100,
        "dec_edge_discount": 0.9,
        "step_size": 0.5,
        "csi" : 29,
        "sunlight_weight": 0.9260,
        "sugar_weight": 10.1435,
        "intercept": -1407.3115,
        "normal_edge": 0.5,
        "panic_edge": 0.5,
        "conversion_threshold": 0.8,
        "std_window": 3,
        "adverse_volume": 30
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
            Product.COUPON_10500: 200,
            Product.MACARON: 75
        }
        self.CONVLIMIT = {
            Product.MACARON: 10
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
        if (basket_type == Product.PIC1 or basket_type == Product.PIC2):
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
    
    def clear_basket_position(self, basket_type: str, position: int, order_depth: OrderDepth):
        if abs(position) == 0:
            return []
        
        if position > 0:
            price = max(order_depth.buy_orders.keys())
        else:
            price = min(order_depth.sell_orders.keys())
        return [Order(basket_type, price, -position)]
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
    
    def macaron_implied_bid_ask(self, observation: ConversionObservation) -> Tuple[float, float]:
        bidprice = observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1
        askprice = observation.askPrice + observation.importTariff + observation.transportFees
        return bidprice, askprice
        
    def macaron_arb_clear(self, position: int) -> int:
        conversions = -position
        return conversions
    
    def macaron_adap_edge(
        self, 
        timestamp: int,
        curr_edge: float,
        position: int,
        traderObject: dict
    ) -> float:
        if timestamp == 0:
            traderObject[Product.MACARON]["curr_edge"] = PARAMS[Product.MACARON]["init_make_edge"]
            return PARAMS[Product.MACARON]["init_make_edge"]
        
        traderObject[Product.MACARON]["volume_history"].append(abs(position))
        if len(traderObject[Product.MACARON]["volume_history"]) > PARAMS[Product.MACARON]["volume_avg_timestamp"]:
            traderObject[Product.MACARON]["volume_history"].pop(0)
        
        if len(traderObject[Product.MACARON]["volume_history"]) < PARAMS[Product.MACARON]["volume_avg_timestamp"]:
            return curr_edge
        elif not traderObject[Product.MACARON]["optimized"]:
            volume_avg = np.mean(traderObject[Product.MACARON]["volume_history"])
            
            if volume_avg >= PARAMS[Product.MACARON]["volume_bar"]:
                traderObject[Product.MACARON]["volume_history"] = []
                traderObject[Product.MACARON]["curr_edge"] = curr_edge + PARAMS[Product.MACARON]["step_size"]
                return curr_edge + PARAMS[Product.MACARON]["step_size"]
            
            elif PARAMS[Product.MACARON]["dec_edge_discount"] * PARAMS[Product.MACARON]["volume_bar"] * (curr_edge - PARAMS[Product.MACARON]["step_size"]) > volume_avg * curr_edge:
                if curr_edge - PARAMS[Product.MACARON]["step_size"] > PARAMS[Product.MACARON]["min_edge"]:
                    traderObject[Product.MACARON]["volume_history"] = []
                    traderObject[Product.MACARON]["curr_edge"] = curr_edge - PARAMS[Product.MACARON]["step_size"]
                    traderObject[Product.MACARON]["optimized"] = True
                    return curr_edge - PARAMS[Product.MACARON]["step_size"]
                else:
                    traderObject[Product.MACARON]["curr_edge"] = PARAMS[Product.MACARON]["min_edge"]
                    return PARAMS[Product.MACARON]["min_edge"]
                
        traderObject[Product.MACARON]["curr_edge"] = curr_edge
        return curr_edge
    
    def macaron_arb_take(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        adap_edge: float,
        position: int
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.MACARON]
        buy_order_volume = 0
        sell_order_volume = 0
        
        implied_bid, implied_ask = self.macaron_implied_bid_ask(observation)
        
        buy_quantity = position_limit - position
        sell_quantity = position_limit + position
        
        ask = implied_ask + adap_edge
        
        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - 1.6
        
        if aggressive_ask > implied_ask:
            ask = aggressive_ask
            
        edge = (ask - implied_ask) * PARAMS[Product.MACARON]["make_probability"]
        
        for price in sorted(list(order_depth.sell_orders.keys())):
            if price > implied_bid - edge:
                break
            
            if price < implied_bid - edge:
                quantity = min(abs(order_depth.sell_orders[price]), buy_quantity)
                if quantity > 0:
                    orders.append(Order(Product.MACARON, round(price), quantity))
                    buy_order_volume += quantity
                    
        for price in sorted(list(order_depth.buy_orders.keys()), reverse=True):
            if price < implied_ask + edge:
                break
            
            if price > implied_ask + edge:
                quantity = min(abs(order_depth.buy_orders[price]), sell_quantity)
                if quantity > 0:
                    orders.append(Order(Product.MACARON, round(price), -quantity))
                    sell_order_volume += quantity
        
        return orders, buy_order_volume, sell_order_volume
    
    def macaron_arb_make(
        self, 
        order_depth: OrderDepth,
        observation: ConversionObservation,
        position: int,
        edge: float,
        buy_order_volume: int,
        sell_order_volume: int
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.MACARON]
        
        implied_bid, implied_ask = self.macaron_implied_bid_ask(observation)
        
        bid = implied_bid - edge
        ask = implied_ask + edge
        
        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - 1.6
        
        if aggressive_ask >= implied_ask + PARAMS[Product.MACARON]["min_edge"]:
            ask = aggressive_ask
        
        filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
        filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 10]
        
        if len(filtered_ask) > 0 and ask > filtered_ask[0]:
            if filtered_ask[0] - 1 > implied_ask:
                ask = filtered_ask[0] - 1
            else:
                ask = implied_ask + edge
            
        if len(filtered_bid) > 0 and bid < filtered_bid[0]:
            if filtered_bid[0] + 1 < implied_bid:
                bid = filtered_bid[0] + 1
            else:
                bid = implied_bid - edge
                
        best_bid = min(order_depth.buy_orders.keys())
        best_ask = max(order_depth.sell_orders.keys())
        
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.MACARON, round(bid), buy_quantity))
        
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.MACARON, round(ask), -sell_quantity))
            
        return orders, buy_order_volume, sell_order_volume
            
    def macaron_fair_value(self, obs: ConversionObservation, traderObject: Dict[str, int]) -> float:
        sunlight = obs.sunlightIndex
        sugar = obs.sugarPrice
        
        csi = PARAMS[Product.MACARON]["csi"]
        sunlight_weight = PARAMS[Product.MACARON]["sunlight_weight"]
        sugar_weight = PARAMS[Product.MACARON]["sugar_weight"]
        
        
        if sunlight < csi:
            fair_price = sunlight * sunlight_weight + sugar * sugar_weight + PARAMS[Product.MACARON]["intercept"]
            return fair_price
        else:
            bid = obs.bidPrice - obs.exportTariff - obs.transportFees
            ask = obs.askPrice + obs.importTariff + obs.transportFees
            traderObject["prev_fairprices"].append((bid + ask) / 2)
            if len(traderObject["prev_fairprices"]) < PARAMS[Product.MACARON]["std_window"]:
                return (bid + ask) / 2
            else:
                return np.array(traderObject["prev_fairprices"]).mean()
        
    def macaron_regime_trade(
        self, 
        order_depth: OrderDepth,
        observation: ConversionObservation,
        position: int,
        traderObject: dict
    ) -> Tuple[List[Order], float, int]:
        orders = []
        sunlight = observation.sunlightIndex
        position_limit = self.LIMIT[Product.MACARON]
        
        best_bid = max(order_depth.buy_orders.keys(), default = None)
        best_ask = min(order_depth.sell_orders.keys(), default = None)
        
        buy_order_volume = 0
        sell_order_volume = 0
        
        csi = PARAMS[Product.MACARON]["csi"]
        fair_value = self.macaron_fair_value(observation, traderObject)
        if sunlight < csi:
            edge = PARAMS[Product.MACARON]["panic_edge"]
        else:
            edge = PARAMS[Product.MACARON]["normal_edge"]
            
        if best_ask is not None and best_ask < fair_value - edge:
            qty = min(position_limit - position, abs(order_depth.sell_orders[best_ask]), PARAMS[Product.MACARON]["adverse_volume"])
            if qty > 0:
                orders.append(Order(Product.MACARON, best_ask, qty))
                buy_order_volume += qty
        if best_bid is not None and best_bid > fair_value + edge:
            qty = min(position_limit + position, abs(order_depth.buy_orders[best_bid]), PARAMS[Product.MACARON]["adverse_volume"])
            if qty > 0:
                orders.append(Order(Product.MACARON, best_bid, -qty))
                sell_order_volume += qty
                
        make_bid = round(fair_value - edge)
        make_ask = round(fair_value + edge)
            
        temp_position = position + buy_order_volume - sell_order_volume
        if temp_position < position_limit:
            qty = min(position_limit - temp_position, PARAMS[Product.MACARON]["adverse_volume"])
            orders.append(Order(Product.MACARON, make_bid, qty))
            buy_order_volume += qty
        if temp_position > -position_limit:
            qty = max(-(position + position_limit), -PARAMS[Product.MACARON]["adverse_volume"])
            orders.append(Order(Product.MACARON, make_ask, qty))
            sell_order_volume += qty
            
        return orders, fair_value
    
    def macaron_conversion_decision(
        self,
        position: int,
        observation: ConversionObservation,
        fair_value: float
    ) -> int:
        if position == 0:
            return 0
        
        if position > 0:
            conv_price = observation.bidPrice - observation.exportTariff - observation.transportFees
            profit = conv_price - fair_value
        else:
            conv_price = observation.askPrice + observation.importTariff + observation.transportFees
            profit = fair_value - conv_price
        
        if profit > PARAMS[Product.MACARON]["conversion_threshold"]:
            if position > 0:
                return -min(abs(position), self.CONVLIMIT[Product.MACARON])
            else:
                return min(abs(position), self.CONVLIMIT[Product.MACARON])
        else:
            return 0
    
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
        if not pic1_spread_orders:
            if traderObject[Product.SPREAD1]["prev_zscore"] < 0.5 and traderObject[Product.SPREAD1]["prev_zscore"] > -0.5:
                result[Product.PIC1] = self.clear_basket_position(Product.PIC1, pic1_position, state.order_depths[Product.PIC1])
        if not pic2_spread_orders:
            if traderObject[Product.SPREAD2]["prev_zscore"] < 0.5 and traderObject[Product.SPREAD2]["prev_zscore"] > -0.5:
                result[Product.PIC2] = self.clear_basket_position(Product.PIC2, pic2_position, state.order_depths[Product.PIC2])
        
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

        
        vouchers = [Product.COUPON_9500, Product.COUPON_9750, Product.COUPON_10000, Product.COUPON_10250, Product.COUPON_10500]
        rock_order_depth = state.order_depths[Product.ROCK]
        rock_mid_price = (max(rock_order_depth.buy_orders.keys()) + min(rock_order_depth.sell_orders.keys())) / 2            
        rock_estimates = []
        for coupon in vouchers:
            coupon_depth = state.order_depths[coupon]
            if not coupon_depth or not coupon_depth.buy_orders or not coupon_depth.sell_orders:
                continue
            
            best_bid = max(coupon_depth.buy_orders.keys())
            best_ask = min(coupon_depth.sell_orders.keys())
            
            mid = (best_bid + best_ask) / 2
            intrinsic = max(0, rock_mid_price - PARAMS[coupon]["strike"])
            position = 0
            if coupon in state.position:
                position = state.position[coupon]
            
            if mid < intrinsic - PARAMS[coupon]["edge_threshold"]:
                size = min(self.LIMIT[coupon], max(0, self.LIMIT[coupon] - position))
                if coupon not in result:
                    result[coupon] = [Order(coupon, best_ask, size)]
                else:
                    result[coupon].append([Order(coupon, best_ask, size)])
            elif mid > intrinsic + PARAMS[coupon]["edge_threshold"]:
                size = min(self.LIMIT[coupon], max(0, position + self.LIMIT[coupon]))
                if coupon not in result:
                    result[coupon] = [Order(coupon, best_bid, -size)]
                else:
                    result[coupon].append([Order(coupon, best_bid, -size)])
                    
            rock_estimates.append(PARAMS[coupon]["strike"] + mid)
        
        if len(rock_estimates) >= 3:
            estimated_rock = sum(rock_estimates) / len(rock_estimates)
            diff = estimated_rock - rock_mid_price
            rock_orders = []
            best_rock_ask = min(rock_order_depth.sell_orders.keys())
            best_rock_bid = max(rock_order_depth.buy_orders.keys())
            rock_position = state.position[Product.ROCK] if Product.ROCK in state.position else 0
            if diff > PARAMS[Product.ROCK]["trade_threshold"]:
                size = min(50, self.LIMIT[Product.ROCK] - rock_position, rock_order_depth.sell_orders[best_rock_ask])
                if size > 0:
                    rock_orders.append(Order(Product.ROCK, best_rock_ask, size))
            elif diff < -PARAMS[Product.ROCK]["trade_threshold"]:
                size = min(50, self.LIMIT[Product.ROCK] + rock_position, abs(rock_order_depth.buy_orders[best_rock_bid]))
                if size > 0:
                    rock_orders.append(Order(Product.ROCK, best_rock_bid, -size))
                
            if rock_orders:
                result[Product.ROCK] = rock_orders
        
                
        if Product.MACARON not in traderObject:
            traderObject[Product.MACARON] = {"prev_fairprices": []}
        
        if Product.MACARON in state.order_depths:
            macaron_position = state.position[Product.MACARON] if Product.MACARON in state.position else 0
            obs = state.observations.conversionObservations[Product.MACARON]
            order_depth = state.order_depths[Product.MACARON]
            
            orders, fair_value = self.macaron_regime_trade(order_depth, obs, macaron_position, traderObject[Product.MACARON])
            result[Product.MACARON] = orders
            conversions = self.macaron_conversion_decision(macaron_position, obs, fair_value)

                    
        if len(traderObject[Product.MACARON]["prev_fairprices"]) > PARAMS[Product.MACARON]["std_window"]:
            traderObject[Product.MACARON]["prev_fairprices"].pop(0)
        
        traderData = jsonpickle.encode(traderObject) # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
                
        logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData
    