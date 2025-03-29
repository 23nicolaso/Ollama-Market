from langchain_ollama import OllamaLLM
import queue
from market_simulator.config import (
    ASSETS, INITIAL_PRICES, SPREADS, ANNUAL_RETURNS,
    LLM_MODEL, MAX_HISTORY_LENGTH, ASSETS_CONFIG, RFR
)
from datetime import datetime, timedelta
import random
from market_simulator.utils.db_utils import get_price_history as get_db_price_history
from market_simulator.utils.db_utils import db_manager as db
import numpy as np

class CircularBuffer:
    # CIRCULAR BUFFER TO STORE PAST PRICES INSTEAD OF ARRAY FOR FAST SPEED
    def __init__(self, size=500):
        self.size = size
        self.buffer = np.empty(size, dtype=np.float64)  # Preallocated buffer
        self.index = 0  # Tracks the next write position
        self.full = False  # Tracks if the buffer has filled
        
    def getLastPrice(self):
        """Returns the most recent price in the buffer."""
        if self.index == 0:
            # If index is 0, the last item is at the end of the buffer (if full)
            # or there are no items yet
            return self.buffer[-1] if self.full else None
        # Otherwise, the last item is at index-1
        return self.buffer[self.index - 1]

    def append(self, value):
        """Adds a new float value to the buffer, overwriting the oldest one if full."""
            # Convert value to float if it's a sequence with one element
        if hasattr(value, '__len__'):
            value = float(value[0])
        else:
            value = float(value)
            
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.size  # Circular increment
        if self.index == 0:  # Marks buffer as full once it wraps
            self.full = True
    
    def clear(self):
        """Clears the contents of the circular buffer."""
        self.index = 0
        self.full = False
        self.buffer = np.empty(self.size, dtype=np.float64)  # Reset the buffer with empty values

    def get(self):
        """Returns the buffer in correct order (newest last)."""
        if not self.full:
            return self.buffer[:self.index]  # Only valid data
        return np.concatenate((self.buffer[self.index:], self.buffer[:self.index]))  # Reorder

    def mean(self):
        """Returns the mean price without reordering."""
        if not self.full:
            return np.mean(self.buffer[:self.index])
        return np.mean(self.buffer)

    def std(self):
        """Returns the standard deviation without reordering."""
        if not self.full:
            return np.std(self.buffer[:self.index], ddof=0)  
        return np.std(self.buffer, ddof=0)

    def sum(self):
        """Returns the sum without reordering."""
        if not self.full:
            return np.sum(self.buffer[:self.index])
        return np.sum(self.buffer)

    def __len__(self):
        """Returns the number of elements currently in the buffer."""
        return self.size if self.full else self.index

# Add these to the global variables
news_queue = queue.Queue()
chat_queue = queue.Queue()

# Global variables
accounts = {}  # Stores list of Account objects, indexed by accountID
initial_prices = INITIAL_PRICES
last_prices = {}  # Stores last price for each asset
spreads_by_market = SPREADS
average_annual_return_by_market = ANNUAL_RETURNS
economic_health_by_market = {asset: 1 for asset in ASSETS}
simulation_age = 0  # Stores the age of the simulation in total sets of 10 ticks
price_history = {}  # Stores list of past prices for each asset
assets = ASSETS  # Stores list of assets
markets = {}  # Stores list of OrderBook objects, indexed by asset
recentHeadlines = []  # Stores list of recently generated headlines

# store risk parameters for each asset
RISKS = {asset: config["risk"] for asset, config in ASSETS_CONFIG.items() if config["use_rp"]}
EV = {asset: config["initial_price"] * (1+config["annual_return"]) for asset, config in ASSETS_CONFIG.items()}
EST_INFLATION = 0.03

def calculate_fair_value(asset):
    # Calculate a fair value for an asset using risk free rate, risk parameters
    if RISKS.get(asset):
        required_ror = RISKS[asset] + EST_INFLATION + RFR
    else:
        required_ror = RFR + EST_INFLATION
    fair_value = EV[asset]/(1+required_ror)
    return fair_value

def calculate_r_adj_ytm(asset):
    if RISKS.get(asset):
        return (EV[asset]/last_prices[asset]-(RISKS[asset]))

def randomly_alter_risk_params(asset):
    # Randomly alter risk param, with a very small chance of a major change across the board. If major change, there should be a news
    # article created explaining it. 
    # Randomly alter risk parameters with small probability
    if RISKS.get(asset):  # Only alter if asset uses risk parameters
        # Small random changes (±2%) with 20% probability
        if random.random() < 0.2:
            RISKS[asset] *= random.uniform(0.98, 1.02)
            
        # Major changes (±20%) with 1% probability
        if random.random() < 0.01:
            mult = random.uniform(0.8, 1.2)
            RISKS[asset] *= mult
            
            return (mult-1)*100
        
        return None

# Initialize last_prices and price_history from DB or defaults
for asset in ASSETS:
    price_history[asset] = CircularBuffer(MAX_HISTORY_LENGTH)
    db_prices = db.get_price_history(asset)
    if db_prices:
        last_prices[asset] = db_prices[-1][0]  # Get most recent price
        for price, _ in db_prices[-MAX_HISTORY_LENGTH:]:
            price_history[asset].append(price)
    else:
        last_prices[asset] = INITIAL_PRICES[asset]
        price_history[asset].append([INITIAL_PRICES[asset]])

# Initialize LLM
model = OllamaLLM(model=LLM_MODEL)

def invoke_model(prompt):
    """Invokes the LLM with a given prompt"""
    response = model.invoke(prompt)
    return response

def update_price_history(asset, price):
    """Updates the price history for a given asset"""
    if asset in price_history:
        rounded_price = round(price, 2)
        price_history[asset].append(rounded_price)
        db.queue_price_update(asset, rounded_price)
    else:
        rounded_price = round(price, 2)
        price_history[asset] = CircularBuffer(MAX_HISTORY_LENGTH)
        price_history[asset].append(rounded_price)

def makeMarkets():
    """Creates OrderBook objects for all assets"""
    from market_simulator.models.reworked_order_book import OrderBook
    for asset in assets:
        markets[asset] = OrderBook(asset, last_prices[asset])

def estimateUnderlyingValue(asset):
    """Estimates the underlying value of an asset based on economic factors"""
    estimated_price = initial_prices[asset] * (1 + average_annual_return_by_market[asset] / 252 / 24) ** simulation_age
    economic_health_adjustment = economic_health_by_market[asset] * 10 * spreads_by_market[asset]
    adjusted_price = max(0, estimated_price + economic_health_adjustment)
    return adjusted_price

def get_price_history(asset, start_time=None, end_time=None):
    """
    Gets price history for an asset, combining database and in-memory data.
    
    Args:
        asset (str): The asset to get history for
        start_time (datetime, optional): Start time for history query
        end_time (datetime, optional): End time for history query
        
    Returns:
        list: List of (price, timestamp) tuples
    """
    if end_time is None:
        end_time = datetime.now(ZoneInfo("UTC"))
    
    # Get in-memory prices with timestamps
    current_time = end_time
    memory_prices = []
    if asset in price_history:
        time_step = timedelta(seconds=0.1)  # 100ms between price updates
        prices = price_history[asset].get()  # Get prices in correct order from CircularBuffer
        for i, price in enumerate(reversed(prices)):
            timestamp = current_time - (i * time_step)
            memory_prices.append((price, timestamp))
        memory_prices.reverse()
    
    # If we don't need historical data, return just in-memory prices
    if start_time and all(timestamp >= start_time for _, timestamp in memory_prices):
        return [p for p in memory_prices if start_time <= p[1] <= end_time]
    
    # Get historical prices from database
    historical_prices = db.get_price_history(asset, start_time, end_time)
    
    # Combine historical and in-memory prices, avoiding duplicates
    if not memory_prices:
        return historical_prices
    
    if not historical_prices:
        return memory_prices
    
    # Find where to splice the data
    splice_time = memory_prices[0][1]
    combined_prices = [p for p in historical_prices if p[1] < splice_time]
    combined_prices.extend(memory_prices)
    
    return combined_prices

def reset_markets():
    """Reset all markets to initial state"""
    # Clear all order books
    for market in markets.values():
        market.bids.clear()
        market.asks.clear()
        market.urgentBuys.clear()
        market.urgentSells.clear()
    
    # Clear all executional trader intended trades
    for account in accounts.values():
        if hasattr(account, 'intendedOrders'):
            account.intendedOrders.clear()
        if hasattr(account, 'conditionalOrders'):
            account.conditionalOrders.clear()