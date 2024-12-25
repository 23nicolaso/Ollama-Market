
import random
import time
import math
import threading
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import queue
from langchain_ollama import OllamaLLM

# =========================== #
# Created by Nicolas Ollivier #
# Last Updated: 25-09-2024    #
# =========================== #

'''
Welcome to the optimized codebase for OLLAMA Market Simulator!                                          
This codebase simulates a market in a controlled environment, allowing for efficient execution and scalability.
'''

model = OllamaLLM(model="llama3.1")

# =================================================== Data Structures ================================================================== #
class MarketData:
    def __init__(self):
        self.accounts = {}  # Stores Account objects, indexed by accountID
        self.assets = ["Simula 500", "Rivala ETF", "Allia ETF", "Factoria ETF", "Gold"]
        self.initial_prices = {asset: 100 if asset != "Gold" else 4000 for asset in self.assets}
        self.last_prices = self.initial_prices.copy()
        self.spreads_by_market = {"Simula 500": 0.02, "Rivala ETF": 0.05, "Allia ETF": 0.02, "Factoria ETF": 0.08, "Gold": 0.05}
        self.average_annual_return = {"Simula 500": 0.3, "Rivala ETF": 0.15, "Allia ETF": 0.12, "Factoria ETF": 0.09, "Gold": 0.05}
        self.economic_health = {asset: 1 for asset in self.assets}
        self.simulation_age = 0
        self.price_history = {asset: deque([price], maxlen=1000) for asset, price in self.last_prices.items()}
        self.markets = {}
        self.recentHeadlines = deque(maxlen=10)
        self.news_queue = queue.Queue()
        self.chat_queue = queue.Queue()
        self.model_lock = threading.Lock()
        self.current_asset = "Simula 500"

market_data = MarketData()

# ============================================ Helper Functions ============================================ #
def update_price_history(asset, price):
    market_data.price_history[asset].append(price)

def make_markets():
    for asset in market_data.assets:
        market_data.markets[asset] = OrderBook(asset, market_data.last_prices[asset])

def estimate_underlying_value(asset):
    estimated_price = market_data.initial_prices[asset] * (1 + market_data.average_annual_return[asset] / 252 / 24) ** market_data.simulation_age
    economic_adjustment = market_data.economic_health[asset] * 10 * market_data.spreads_by_market[asset]
    adjusted_price = max(0, estimated_price + economic_adjustment)
    return adjusted_price

def redeem_contracts(asset, quantity):
    market_data.economic_health[asset] += quantity * market_data.spreads_by_market[asset] * 0.1

# ========================================================================================================== #

# ============================================ Class Definitions ============================================ #
class Account:
    def __init__(self, accountID, CASH):
        self.accountID = accountID
        self.positions = {"CASH": CASH}

    def add_position(self, asset, quantity):
        self.positions[asset] = self.positions.get(asset, 0) + quantity

    def get_position(self, asset):
        return self.positions.get(asset, 0)

    def get_cash(self):
        return self.positions.get("CASH", 0)

    def get_value(self):
        total = self.get_cash()
        for asset, qty in self.positions.items():
            if asset != "CASH":
                total += qty * market_data.last_prices.get(asset, 0)
        return total

    def trade_at_price(self, asset, price, quantity, direction):
        self.add_position(asset, quantity * direction)
        self.add_position("CASH", -price * quantity * direction)

    def redeem_underlying_value(self, asset, quantity):
        self.add_position(asset, -quantity)
        self.add_position("CASH", quantity * estimate_underlying_value(asset))
        redeem_contracts(asset, quantity)


class OrderBook:
    class OrderLevel:
        def __init__(self, price, quantity, asset, direction):
            self.asset = asset
            self.direction = 1 if direction == 1 else -1
            self.price = price
            self.netQuantity = quantity
            self.orders = deque()

        def add_order(self, quantity, accountID):
            self.netQuantity += quantity
            creation_time = time.time()
            self.orders.append((quantity, accountID, creation_time))

        def fulfill_all(self):
            while self.orders:
                quantity, accountID, _ = self.orders.popleft()
                account = market_data.accounts.get(accountID)
                if account:
                    account.trade_at_price(self.asset, self.price, quantity, self.direction)
            self.netQuantity = 0
            market_data.last_prices[self.asset] = self.price

        def fulfill_quantity(self, quantity_to_fill):
            filled = 0
            while quantity_to_fill > 0 and self.orders:
                order_qty, accountID, _ = self.orders[0]
                account = market_data.accounts.get(accountID)
                if account:
                    trade_qty = min(order_qty, quantity_to_fill)
                    account.trade_at_price(self.asset, self.price, trade_qty, self.direction)
                    filled += trade_qty
                    quantity_to_fill -= trade_qty
                    if trade_qty == order_qty:
                        self.orders.popleft()
                    else:
                        self.orders[0] = (order_qty - trade_qty, accountID, time.time())
            self.netQuantity -= filled
            market_data.last_prices[self.asset] = self.price
            return quantity_to_fill

        def cancel_orders_from_id(self, accountID):
            self.orders = deque([order for order in self.orders if order[1] != accountID])
            self.netQuantity = sum(order[0] for order in self.orders)

    def __init__(self, asset, initial_price):
        self.asset = asset
        self.bids = {}
        self.asks = {}
        self.urgent_buys = deque()
        self.urgent_sells = deque()
        market_data.last_prices[self.asset] = initial_price

    def add_order(self, direction, price, quantity, order_type, accountID):
        book = self.bids if direction == "buy" else self.asks
        if order_type == "limit":
            if price in book:
                book[price].add_order(quantity, accountID)
            else:
                book[price] = self.OrderLevel(price, quantity, self.asset, 1 if direction == "buy" else -1)
                book[price].add_order(quantity, accountID)
            self.match_books()
        else:
            if direction == "buy":
                self.urgent_buys.append((quantity, accountID))
            else:
                self.urgent_sells.append((quantity, accountID))
            self.fill_urgent_orders()

    def get_best_bid(self):
        return self.bids[max(self.bids.keys())] if self.bids else None

    def get_best_ask(self):
        return self.asks[min(self.asks.keys())] if self.asks else None

    def match_books(self):
        while True:
            best_bid = self.get_best_bid()
            best_ask = self.get_best_ask()
            if not best_bid or not best_ask or best_bid.price < best_ask.price:
                break
            quantity_to_fill = min(best_bid.netQuantity, best_ask.netQuantity)
            best_bid.fulfill_quantity(quantity_to_fill)
            best_ask.fulfill_quantity(quantity_to_fill)
            if best_bid.netQuantity == 0:
                del self.bids[best_bid.price]
            if best_ask.netQuantity == 0:
                del self.asks[best_ask.price]

    def fill_urgent_orders(self):
        while self.urgent_buys and self.urgent_sells:
            buy_qty, buy_id = self.urgent_buys.popleft()
            sell_qty, sell_id = self.urgent_sells.popleft()
            filled_qty = min(buy_qty, sell_qty)
            account_buy = market_data.accounts.get(buy_id)
            account_sell = market_data.accounts.get(sell_id)
            if account_buy and account_sell:
                mid_price = (self.get_best_bid().price + self.get_best_ask().price) / 2 if self.get_best_bid() and self.get_best_ask() else market_data.last_prices[self.asset]
                account_buy.add_position(self.asset, filled_qty)
                account_buy.add_position("CASH", -mid_price * filled_qty)
                account_sell.add_position(self.asset, -filled_qty)
                account_sell.add_position("CASH", mid_price * filled_qty)
                market_data.last_prices[self.asset] = mid_price

        # Handle remaining urgent buys
        while self.urgent_buys:
            buy_qty, buy_id = self.urgent_buys.popleft()
            account_buy = market_data.accounts.get(buy_id)
            best_ask = self.get_best_ask()
            if account_buy and best_ask:
                fill_qty = min(buy_qty, best_ask.netQuantity)
                best_ask.fulfill_quantity(fill_qty)
                account_buy.add_position(self.asset, fill_qty)
                account_buy.add_position("CASH", -best_ask.price * fill_qty)
                market_data.last_prices[self.asset] = best_ask.price
                if best_ask.netQuantity == 0:
                    del self.asks[best_ask.price]
            else:
                break

        # Handle remaining urgent sells
        while self.urgent_sells:
            sell_qty, sell_id = self.urgent_sells.popleft()
            account_sell = market_data.accounts.get(sell_id)
            best_bid = self.get_best_bid()
            if account_sell and best_bid:
                fill_qty = min(sell_qty, best_bid.netQuantity)
                best_bid.fulfill_quantity(fill_qty)
                account_sell.add_position(self.asset, -fill_qty)
                account_sell.add_position("CASH", best_bid.price * fill_qty)
                market_data.last_prices[self.asset] = best_bid.price
                if best_bid.netQuantity == 0:
                    del self.bids[best_bid.price]
            else:
                break

    def cancel_orders_by_account(self, accountID):
        for bid in list(self.bids.values()):
            bid.cancel_orders_from_id(accountID)
        for ask in list(self.asks.values()):
            ask.cancel_orders_from_id(accountID)

    def clear_far_orders(self):
        threshold = 5
        last_price = market_data.last_prices.get(self.asset, 0)
        far_bid_prices = [price for price in self.bids if price < last_price - threshold]
        for price in far_bid_prices:
            del self.bids[price]
        far_ask_prices = [price for price in self.asks if price > last_price + threshold]
        for price in far_ask_prices:
            del self.asks[price]

    def display_price(self):
        print(f"{self.asset} price: {round(market_data.last_prices[self.asset], 2)}")

    def get_last_price(self):
        return market_data.last_prices.get(self.asset, None)


class MarketAgent:
    def __init__(self, accountID, cash):
        self.account = Account(accountID, cash)
        market_data.accounts[accountID] = self.account

    def place_order(self, orderBook, direction, price, quantity, orderType):
        orderBook.add_order(direction, price, quantity, orderType, self.account.accountID)

    def redeem_underlying_value(self, asset, quantity):
        self.account.redeem_underlying_value(asset, quantity)

    def display_account(self):
        print(f"Account {self.account.accountID} has {self.account.get_cash()} cash and positions: {self.account.positions}")


class MarketMaker(MarketAgent):
    def __init__(self, accountID, cash, spreads):
        super().__init__(accountID, cash)
        self.spreads = spreads

    def wipe_orders(self, orderBook):
        value = estimate_underlying_value(orderBook.asset)
        for bid in list(orderBook.bids.values()):
            if value - bid.price >= self.spreads[orderBook.asset] * 10:
                bid.cancel_orders_from_id("MARKET MAKER")
        for ask in list(orderBook.asks.values()):
            if ask.price - value >= self.spreads[orderBook.asset] * 10:
                ask.cancel_orders_from_id("MARKET MAKER")

    def make_market(self, orderBook):
        mid_price = orderBook.get_last_price()
        self.wipe_orders(orderBook)
        bid_price = round(mid_price - (self.spreads[orderBook.asset] / 2), 2)
        ask_price = round(mid_price + (self.spreads[orderBook.asset] / 2), 2)
        self.place_order(orderBook, "buy", bid_price, 10000, "limit")
        self.place_order(orderBook, "sell", ask_price, 10000, "limit")

    def provide_liquidity(self, orderBook):
        total_buy = sum(qty for qty, aid in orderBook.urgent_buys if aid != "MARKET MAKER")
        total_sell = sum(qty for qty, aid in orderBook.urgent_sells if aid != "MARKET MAKER")
        if total_buy > 0:
            price_change = self.spreads[orderBook.asset] * total_buy / 100
            self.place_order(orderBook, "sell", market_data.last_prices[orderBook.asset] + round(price_change, 2), total_buy, "limit")
            self.redeem_underlying_value(orderBook.asset, total_buy)
        if total_sell > 0:
            price_change = self.spreads[orderBook.asset] * total_sell / 100
            self.place_order(orderBook, "buy", market_data.last_prices[orderBook.asset] - round(price_change, 2), total_sell, "limit")
            self.redeem_underlying_value(orderBook.asset, -total_sell)

    def arbitrage_fair_value(self, orderBook):
        fair_value = estimate_underlying_value(orderBook.asset)
        current_price = orderBook.get_last_price()
        spread = self.spreads[orderBook.asset] * 10
        if current_price < fair_value - spread:
            self.place_order(orderBook, "buy", fair_value - self.spreads[orderBook.asset], 100, "limit")
            self.redeem_underlying_value(orderBook.asset, -100)
        elif current_price > fair_value + spread:
            self.place_order(orderBook, "sell", fair_value + self.spreads[orderBook.asset], 100, "limit")
            self.redeem_underlying_value(orderBook.asset, 100)
        self.provide_liquidity(orderBook)


class ExecutionalTrader(MarketAgent):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        self.intended_orders = {}
        self.conditional_orders = {}

    def get_conditional_orders(self, orderBook, direction):
        return sum(order["quantity"] for order in self.conditional_orders.get(orderBook.asset, []) if order["direction"] == direction)

    def place_conditional_order(self, orderBook, direction, price, quantity, priceCondition, conditionalDirection):
        if orderBook.asset not in self.conditional_orders:
            self.conditional_orders[orderBook.asset] = []
        self.conditional_orders[orderBook.asset].append({
            "direction": direction,
            "price": price,
            "quantity": quantity,
            "priceCondition": priceCondition,
            "conditionalDirection": conditionalDirection
        })

    def check_conditional_orders(self, asset):
        orders = self.conditional_orders.get(asset, [])
        for order in orders[:]:
            current_price = market_data.last_prices.get(asset, 0)
            condition_met = False
            if order["conditionalDirection"] == "above" and current_price >= order["priceCondition"]:
                condition_met = True
            elif order["conditionalDirection"] == "below" and current_price <= order["priceCondition"]:
                condition_met = True
            if condition_met:
                self.place_order(market_data.markets[asset], order["direction"], order["price"], order["quantity"], "market")
                orders.remove(order)

    def execute_trade_in_legs(self, orderBook, direction, price, quantity):
        self.intended_orders[orderBook.asset] = {"direction": direction, "price": price, "quantity": quantity}

    def remove_old_intended_orders(self):
        self.intended_orders.clear()

    def partial_execute_market(self, orderBook):
        if random.random() < 0.05 and orderBook.asset in self.intended_orders:
            order = self.intended_orders[orderBook.asset]
            if order["quantity"] > 0:
                qty_to_fill = random.randint(1, max(order["quantity"] // 2, 1))
                self.place_order(orderBook, order["direction"], order["price"], qty_to_fill, "market")
                order["quantity"] -= qty_to_fill
                if order["quantity"] <= 0:
                    del self.intended_orders[orderBook.asset]

    def update_orders_in_legs(self, orderBook):
        best_bid = orderBook.get_best_bid()
        best_ask = orderBook.get_best_ask()
        try:
            intended = self.intended_orders[orderBook.asset]
            if intended["direction"] == "buy" and (best_ask and intended["price"] >= best_ask.price):
                fill_qty = min(intended["quantity"], best_ask.netQuantity)
                self.place_order(orderBook, "buy", 0, fill_qty, "market")
                intended["quantity"] -= fill_qty
            elif intended["direction"] == "sell" and (best_bid and intended["price"] <= best_bid.price):
                fill_qty = min(intended["quantity"], best_bid.netQuantity)
                self.place_order(orderBook, "sell", 0, fill_qty, "market")
                intended["quantity"] -= fill_qty
        except KeyError:
            pass


class HedgeFund(ExecutionalTrader):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)


class RetailTrader(MarketAgent):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        self.sentiment_score = {asset: 0.5 for asset in market_data.assets}
        self.news_urgency = 1

    def estimate_sentiment(self, orderBook):
        sentiment = self.sentiment_score.get(orderBook.asset, 0.5)
        variation = random.uniform(-0.05, 0.05)
        return max(0, min(1, sentiment + variation))

    def estimate_importance(self):
        variation = random.randint(-2, 2)
        return max(0, min(self.news_urgency + variation, 10))

    def trade(self, orderBook):
        sentiment = self.sentiment_score.get(orderBook.asset, 0.5)
        bounded_sentiment = max(0.1, min(0.9, sentiment))
        direction = "buy" if bounded_sentiment > random.random() else "sell"
        price = orderBook.get_last_price()
        quantity = random.randint(1, 100)
        position = self.account.get_position(orderBook.asset)

        if direction == "buy" and position + quantity < 100000:
            self.place_order(orderBook, direction, price, quantity, "market")
        elif direction == "sell" and position - quantity > 0:
            self.place_order(orderBook, direction, price, quantity, "market")
        else:
            self.adjust_sentiment(direction)

    def adjust_sentiment(self, direction):
        if direction == "buy":
            self.sentiment_score[self.account.accountID] = max(0, self.sentiment_score[self.account.accountID] - random.uniform(0.01, 0.2))
        else:
            self.sentiment_score[self.account.accountID] = min(1, self.sentiment_score[self.account.accountID] + random.uniform(0.01, 0.2))

    def set_reversion_urgency(self, urgency):
        self.news_urgency = urgency

    def shift_sentiment_to_mean(self):
        for asset, sentiment in self.sentiment_score.items():
            shift_amount = (0.5 - sentiment) / (200 * self.news_urgency)
            self.sentiment_score[asset] += shift_amount
            self.sentiment_score[asset] = max(0, min(1, self.sentiment_score[asset]))
            if round(self.sentiment_score[asset], 2) == 0.5:
                self.sentiment_score[asset] += random.uniform(-0.1, 0.1)
        if all(round(s, 1) == 0.5 for s in self.sentiment_score.values()):
            if self.news_urgency > 1:
                self.news_urgency -= 1


class TATrader(HedgeFund):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)

    def manage_trades(self, market):
        price_list = list(market_data.price_history[market])[-20:]
        if not price_list:
            return
        mean_price = sum(price_list) / len(price_list)
        std_dev = math.sqrt(sum((x - mean_price) ** 2 for x in price_list) / len(price_list))
        quantity = random.randint(10, 100)
        current_price = market_data.markets[market].get_last_price()

        if len(market_data.price_history[market]) > 300:
            high_lt = max(list(market_data.price_history[market])[-300:])
            low_lt = min(list(market_data.price_history[market])[-300:])
            self.set_conditional_orders(market, high_lt, low_lt)

        self.apply_moving_averages(market, price_list, current_price, std_dev)

    def set_conditional_orders(self, market, high_lt, low_lt):
        if self.get_conditional_orders(market, "sell") == 0 and self.account.get_position(market) > 0:
            self.place_conditional_order(market_data.markets[market], "sell", low_lt, self.account.get_position(market) // 20, low_lt, "below")
        elif self.get_conditional_orders(market, "buy") == 0 and self.account.get_position(market) < 0:
            self.place_conditional_order(market_data.markets[market], "buy", high_lt, -self.account.get_position(market) // 20, high_lt, "above")

    def apply_moving_averages(self, market, price_list, current_price, std_dev):
        if len(price_list) < 8:
            return
        ma_8 = sum(price_list[-8:]) / 8
        if current_price < ma_8 and current_price > 0:
            self.place_order(market_data.markets[market], "buy", current_price, random.randint(10, 100), "market")
        elif current_price > ma_8:
            self.place_order(market_data.markets[market], "sell", current_price, random.randint(10, 100), "market")


class HFTFund(HedgeFund):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        self.sentiment_error = random.uniform(-0.3, 0.3)
        self.importance_error = random.uniform(-2, 2)
        self.fair_value_error = random.uniform(-2, 2)
        print(f"HFTFund Initialized with Sentiment Error: {self.sentiment_error}, Importance Error: {self.importance_error}, Fair Value Error: {self.fair_value_error}")

    def update_positioning(self, market):
        self.update_orders_in_legs(market_data.markets[market])
        self.partial_execute_market(market_data.markets[market])

    def estimate_fair_value(self, market):
        return estimate_underlying_value(market) + self.fair_value_error

    def estimate_importance(self, retail_trader):
        return max(0, min(10, retail_trader.estimate_importance() + self.importance_error))

    def estimate_sentiment(self, retail_trader, market):
        return max(0, min(1, retail_trader.estimate_sentiment(market_data.markets[market]) + self.sentiment_error))

    def trade_the_news(self, market, retail_trader):
        sentiment = self.estimate_sentiment(retail_trader, market)
        importance = self.estimate_importance(retail_trader)

        if sentiment <= 0.3:
            self.front_run_trade(market, "sell", importance)
        elif sentiment >= 0.7:
            self.front_run_trade(market, "buy", importance)

    def front_run_trade(self, market, direction, importance):
        current_price = market_data.markets[market].get_last_price()
        quantity = int(pow(importance, 2) * 5)
        if direction == "sell":
            self.place_order(market_data.markets[market], "sell", current_price, quantity, "market")
            self.execute_trade_in_legs(market_data.markets[market], "buy", current_price, math.floor(quantity))
            print(f"Selling {quantity} shares in {market}")
        elif direction == "buy":
            self.place_order(market_data.markets[market], "buy", current_price, quantity, "market")
            self.execute_trade_in_legs(market_data.markets[market], "sell", current_price, math.floor(quantity))
            print(f"Buying {quantity} shares in {market}")

# ========================================= End of Class Definitions ======================================== #

# ============================================ Initialize Agents ============================================= #
retail_trader = RetailTrader("RETAIL_TRADER", 1_000_000)
hedge_fund = HedgeFund("HEDGE_FUND", 5_000_000)
hft_fund = HFTFund("EVENTS_TRADING_FUND", 10_000_000)
hft_fund2 = HFTFund("EVENTS_TRADING_FUND_2", 10_000_000)
hft_fund3 = HFTFund("EVENTS_TRADING_FUND_3", 10_000_000)
ta_traders = TATrader("TA_TRADING_FIRM", 1_000_000)
long_term_investors = HedgeFund("LONG_TERM_INVESTORS", 100_000_000_000_000)
market_maker = MarketMaker("MARKET_MAKER", 100_000_000_000_000, spreads=market_data.spreads_by_market)

make_markets()

# ============================================ GUI Setup ============================================ #
class MarketSimulatorGUI:
    def __init__(self, root, market_data, retail_trader, market_maker, hft_funds, ta_traders, long_term_investors):
        self.root = root
        self.market_data = market_data
        self.retail_trader = retail_trader
        self.market_maker = market_maker
        self.hft_funds = hft_funds
        self.ta_traders = ta_traders
        self.long_term_investors = long_term_investors
        self.tick = 0
        self.setup_gui()
        self.start_simulation()

    def setup_gui(self):
        self.root.title("Market Simulation")
        self.root.geometry("2000x800")
        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True)

        # Dropdown for asset selection
        self.asset_var = tk.StringVar(value=self.market_data.assets[0])
        asset_dropdown = ttk.Combobox(frame, textvariable=self.asset_var, values=self.market_data.assets)
        asset_dropdown.grid(row=0, column=0, padx=5, pady=5)
        asset_dropdown.bind("<<ComboboxSelected>>", self.on_asset_change)

        # Chart length selection
        ttk.Label(frame, text="Chart Length").grid(row=0, column=1, padx=5)
        self.chart_length_var = tk.IntVar(value=500)
        chart_spinbox = tk.Spinbox(frame, from_=1, to=1000, textvariable=self.chart_length_var)
        chart_spinbox.grid(row=0, column=2, padx=5)

        # Generate News and Chat buttons
        ttk.Button(frame, text="Generate News", command=self.gen_news_thread).grid(row=0, column=3, padx=5)
        ttk.Button(frame, text="Generate Chat", command=self.gen_chat_thread).grid(row=0, column=4, padx=5)

        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, rowspan=3, columnspan=3, padx=5, pady=5)

        # News Feed
        self.news_feed = tk.Text(frame, height=40, width=40)
        self.news_feed.grid(row=1, column=3, rowspan=3, padx=5, pady=5)
        self.news_feed.insert(tk.END, "News Feed:\n")
        self.news_feed.see(tk.END)

        # Chat Window
        self.chat_window = tk.Text(frame, height=40, width=40)
        self.chat_window.grid(row=1, column=4, rowspan=3, padx=5, pady=5)
        self.chat_window.insert(tk.END, "Chat Window:\n")
        self.chat_window.see(tk.END)

        # Price Table
        price_frame = ttk.Frame(frame)
        price_frame.grid(row=1, column=5, rowspan=1, padx=5, pady=5, sticky="nsew")
        self.price_tree = ttk.Treeview(price_frame, columns=("Asset", "Price", "NAV"), show="headings")
        for col in ("Asset", "Price", "NAV"):
            self.price_tree.heading(col, text=col)
            self.price_tree.column(col, width=150, anchor="center")
        self.price_tree.pack(fill=tk.BOTH, expand=True)
        for asset, price in self.market_data.last_prices.items():
            self.price_tree.insert("", "end", values=(asset, f"{price:.2f}", f"{estimate_underlying_value(asset):.2f}"))

        # Sentiment Table
        sentiment_frame = ttk.Frame(frame)
        sentiment_frame.grid(row=2, column=5, rowspan=2, padx=5, pady=5, sticky="nsew")
        self.sentiment_tree = ttk.Treeview(sentiment_frame, columns=("Asset", "Sentiment"), show="headings")
        for col in ("Asset", "Sentiment"):
            self.sentiment_tree.heading(col, text=col)
            self.sentiment_tree.column(col, width=150, anchor="center")
        self.sentiment_tree.pack(fill=tk.BOTH, expand=True)
        self.update_sentiments()

    def update_sentiments(self):
        for item in self.sentiment_tree.get_children():
            self.sentiment_tree.delete(item)
        for asset, sentiment in self.retail_trader.sentiment_score.items():
            self.sentiment_tree.insert("", "end", values=(asset, f"{sentiment:.2f}"))

    def update_prices(self):
        for item in self.price_tree.get_children():
            self.price_tree.delete(item)
        for asset, price in self.market_data.last_prices.items():
            nav = estimate_underlying_value(asset)
            self.price_tree.insert("", "end", values=(asset, f"{price:.2f}", f"{nav:.2f}"))

    def update_charts(self):
        selected_asset = self.asset_var.get()
        self.ax.clear()
        history = list(self.market_data.price_history[selected_asset])[-self.chart_length_var.get():]
        self.ax.plot(history, label=selected_asset)
        self.ax.set_title(f"{selected_asset} Price History")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")
        self.ax.legend()
        self.canvas.draw()

    def update_news_feed(self):
        try:
            while True:
                headline = self.market_data.news_queue.get_nowait()
                self.news_feed.insert(tk.END, f"{headline}\n\n")
                self.news_feed.see(tk.END)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.update_news_feed)

    def update_chat_window(self):
        try:
            while True:
                message = self.market_data.chat_queue.get_nowait()
                self.chat_window.insert(tk.END, f"{message}\n\n")
                self.chat_window.see(tk.END)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.update_chat_window)

    def add_to_news_feed(self, headline):
        self.market_data.news_queue.put(headline)

    def add_to_chat_window(self, message):
        self.market_data.chat_queue.put(message)

    def on_asset_change(self, event):
        market_data.current_asset = self.asset_var.get()
        self.chat_window.delete(1.0, tk.END)
        self.chat_window.insert(tk.END, "Chat Window:\n")
        self.update_charts()

    def gen_news(self):
        headline = model.invoke(
            f"Generate a news headline for the simulated world involving countries Simula, Rivala, Allia, and Factoria. Keep it under 15 words. Recent headlines: {list(market_data.recentHeadlines)}."
        )
        print(headline)
        market_data.recentHeadlines.append(headline)
        if len(market_data.recentHeadlines) > 10:
            market_data.recentHeadlines.popleft()

        sentiment = model.invoke(
            f"Predict economic sentiment for Simula 500, Rivala ETF, Allia ETF, Factoria ETF. Recent headline: {market_data.recentHeadlines[-1]}. Format: Asset: score, ..."
        )
        urgency = model.invoke(
            f"Score the impact of this headline between 1-10: {market_data.recentHeadlines[-1]}."
        )
        try:
            urgency = int(urgency)
        except ValueError:
            urgency = 1
        self.retail_trader.set_reversion_urgency(urgency)
        self.add_to_news_feed(f"{urgency} {headline}")

        sentiment_scores = {}
        for asset_score in sentiment.split(', '):
            try:
                asset, score = asset_score.split(': ')
                sentiment_scores[asset] = max(0.2, min(0.8, float(score)))
            except ValueError:
                print(f"Error parsing sentiment score: {asset_score}")

        sentiment_scores["Gold"] = 0.7 + random.uniform(-0.3, 0.3)
        for asset in self.market_data.assets:
            self.retail_trader.sentiment_score[asset] = sentiment_scores.get(asset, 0.5)

        for market in self.market_data.assets:
            self.market_maker.make_market(self.market_data.markets[market])
            for hft in self.hft_funds:
                hft.trade_the_news(market, self.retail_trader)
            self.retail_trader.trade(self.market_data.markets[market])
            self.market_maker.provide_liquidity(self.market_data.markets[market])

    def gen_chat(self):
        message = model.invoke(
            f"In the simulated world with countries Simula, Rivala, Allia, and Factoria, as a retail trader with sentiment score {self.retail_trader.sentiment_score[self.market_data.current_asset]}, what do you think about {self.market_data.current_asset}? Keep it under 10 words."
        )
        self.add_to_chat_window(message)

    def gen_news_thread(self):
        threading.Thread(target=self.gen_news, daemon=True).start()

    def gen_chat_thread(self):
        threading.Thread(target=self.gen_chat, daemon=True).start()

    def start_simulation(self):
        self.gen_news()
        self.gen_chat()
        self.root.after(100, self.update_news_feed)
        self.root.after(100, self.update_chat_window)
        self.run_simulation()

    def run_simulation(self):
        if not self.market_data:
            return

        for market in self.market_data.assets:
            self.market_maker.make_market(self.market_data.markets[market])
            self.retail_trader.trade(self.market_data.markets[market])

        for asset in self.market_data.price_history:
            if len(self.market_data.price_history[asset]) > self.price_history.maxlen:
                self.market_data.price_history[asset].pop()

        self.retail_trader.shift_sentiment_to_mean()

        for market in self.market_data.assets:
            self.market_maker.provide_liquidity(self.market_data.markets[market])
            for hft in self.hft_funds:
                hft.update_positioning(market)
            self.ta_traders.manage_trades(market)
            self.retail_trader.check_conditional_orders(market)
            self.market_data.markets[market].clear_far_orders()

            if len(self.market_data.price_history[market]) < 100:
                continue

            estimated_fair_value = estimate_underlying_value(market)
            target_price = estimated_fair_value
            buy_price = 0.9 * target_price

            self.market_data.markets[market].cancel_orders_by_account("LONG_TERM_INVESTORS")
            long_term_investors.place_order(self.market_data.markets[market], "buy", buy_price, 1000, "limit")

        self.update_charts()
        self.update_prices()
        self.update_sentiments()
        self.tick += 1

        if self.tick > 10:
            self.tick = 0

        self.root.after(100, self.run_simulation)


# Initialize GUI
root = tk.Tk()
gui = MarketSimulatorGUI(
    root,
    market_data,
    retail_trader,
    market_maker,
    [hft_fund, hft_fund2, hft_fund3],
    ta_traders,
    long_term_investors
)
root.mainloop()