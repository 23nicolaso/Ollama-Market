from ollama import Client
import queue
from market_simulator.config import (
    ASSETS, INITIAL_PRICES, SPREADS,
    LLM_MODEL, MAX_HISTORY_LENGTH, RFR
)
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import re 
import random
from market_simulator.utils.db_utils import get_price_history as get_db_price_history
from market_simulator.utils.db_utils import db_manager as db
import numpy as np

from market_simulator.models.marketState import MarkovModel

simulation_last_news_tick = 0

markov_model = MarkovModel()

def setLastNewsTick(tick):
    global simulation_last_news_tick
    simulation_last_news_tick = tick

def wasNewsRecent(tick):
    return tick - simulation_last_news_tick > 500 


class CircularBuffer:
    # CIRCULAR BUFFER TO STORE PAST PRICES INSTEAD OF ARRAY FOR FAST SPEED
    def __init__(self, size=500, initial_price=0):
        self.size = size
        self.buffer = np.empty(size, dtype=np.float64)  # Preallocated buffer
        self.index = 0  # Tracks the next write position
        self.full = False  # Tracks if the buffer has filled
        self.initial_price = initial_price
        
    def getPercentageChange(self):
        """Returns % change since market open"""
        return (self.buffer[self.index-1] - self.initial_price)/self.initial_price * 100

    def getLastNPrices(self, n):
        """Returns the n most recent prices"""
        if n <= 0:
            return np.array([])
            
        if not self.full:
            # If buffer not full, return last n prices from available data
            return self.buffer[max(0, self.index - n):self.index]
            
        # If buffer is full, handle wrap-around case
        if n >= self.size:
            return self.get()  # Return all prices
            
        # Get last n prices with wrap-around
        if self.index >= n:
            return self.buffer[self.index - n:self.index]
        else:
            # Need to wrap around
            return np.concatenate((self.buffer[self.size - (n - self.index):], 
                                 self.buffer[:self.index]))

    
    def getPriceChange(self, n):
        if n <= 1:
             # Need at least two points to calculate a change
            return 0.0

        prices = self.getLastNPrices(n)

        # Check if we have enough data points for the calculation
        if len(prices) < n or len(prices) < 2:
            # Not enough data in the buffer for the requested window 'n'
            # or fewer than 2 points overall to calculate change
            return 0.0

        oldest_price = prices[0]
        newest_price = prices[-1]

        # Avoid division by zero
        if oldest_price == 0:
            # Define behavior: return 0, inf, or raise error? Returning 0 is safest.
            return 0.0

        # Calculate percentage change
        price_change = (newest_price - oldest_price) / oldest_price
        return price_change

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
    
    def isFull(self):
        """Return bool if full"""
        return self.full

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

    def mean(self, n = MAX_HISTORY_LENGTH):
        """Returns the mean price of the past n prices"""
        if self.size < n:
            return self.initial_price
        
        if n == self.size and self.full:
            return np.mean(self.buffer)
            
        if not self.full and n == self.size:
            return np.mean(self.buffer[:self.index])
            
        if self.index >= n:
            return np.mean(self.buffer[self.index - n:self.index])
        else:
            # Need to wrap around
            return np.mean(np.concatenate((self.buffer[self.size - (n - self.index):], 
                                         self.buffer[:self.index])))


    def std(self, n = MAX_HISTORY_LENGTH):
        """Returns the standard deviation of the past n prices"""
        if self.size < n:
            return 1
        
        if n == self.size and self.full:
            return np.std(self.buffer, ddof=0)
            
        if not self.full and n == self.size:
            return np.std(self.buffer[:self.index], ddof=0)
            
        if self.index >= n:
            return np.std(self.buffer[self.index - n:self.index], ddof=0)
        else:
            # Need to wrap around
            return np.std(np.concatenate((self.buffer[self.size - (n - self.index):], 
                                        self.buffer[:self.index])), ddof=0)

    def crossed_over_mean(self, n=MAX_HISTORY_LENGTH):
        """Returns True if price just crossed over the n-period mean, False otherwise"""
        if self.index < 2:  # Need at least 2 points to detect a crossover
            return False
            
        current_price = self.buffer[self.index - 1]
        prev_price = self.buffer[self.index - 2]
        mean_price = self.mean(n)
        
        # Check for upward crossover
        if prev_price <= mean_price and current_price > mean_price:
            return True
            
        # Check for downward crossover
        if prev_price >= mean_price and current_price < mean_price:
            return True
            
        return False

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
action_queue = queue.Queue()  # Live log of agent decisions

# Maps display name -> (agent_object, initial_cash) for the P&L leaderboard.
# Populated in main.py after agents are created.
agent_registry = {}

# Global variables
accounts = {}  # Stores list of Account objects, indexed by accountID
initial_prices = INITIAL_PRICES
last_prices = {}  # Stores last price for each asset
spreads_by_market = SPREADS
economic_health_by_market = {asset: 1 for asset in ASSETS}
simulation_age = 0  # Stores the age of the simulation in total sets of 10 ticks
price_history = {}  # Stores list of past prices for each asset
assets = ASSETS  # Stores list of assets
markets = {}  # Stores list of OrderBook objects, indexed by asset
recentHeadlines = []  # Stores list of recently generated headlines

# Initialize last_prices and price_history from DB or defaults
for asset in ASSETS:
    price_history[asset] = CircularBuffer(MAX_HISTORY_LENGTH, INITIAL_PRICES[asset])
    db_prices = db.get_price_history(asset)
    if db_prices:
        last_prices[asset] = db_prices[-1][0]  # Get most recent price
        for price, _ in db_prices[-MAX_HISTORY_LENGTH:]:
            price_history[asset].append(price)
    else:
        last_prices[asset] = INITIAL_PRICES[asset]
        price_history[asset].append([INITIAL_PRICES[asset]])

# Initialize LLM
model = Client()

def invoke_model(prompt):
    """Invokes the LLM with a given prompt"""
    response = model.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

def invoke_model_stream(prompt):
    """Yields a stream of LLM responses for a given prompt"""
    response = model.chat(
        model=LLM_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        stream=True
    )
    for chunk in response:
        content = chunk['message']['content']
        yield content

def update_price_history(asset, price):
    """Updates the price history for a given asset"""
    if asset in price_history:
        rounded_price = round(price, 2)
        price_history[asset].append(rounded_price)
        db.queue_price_update(asset, rounded_price)
    else:
        rounded_price = round(price, 2)
        price_history[asset] = CircularBuffer(MAX_HISTORY_LENGTH, INITIAL_PRICES[asset])
        price_history[asset].append(rounded_price)

def makeMarkets():
    """Creates OrderBook objects for all assets"""
    from market_simulator.models.sortedListOB import OrderBook
    for asset in assets:
        markets[asset] = OrderBook(asset, last_prices[asset])


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