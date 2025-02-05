from langchain_ollama import OllamaLLM
import queue
from market_simulator.config import (
    ASSETS, INITIAL_PRICES, SPREADS, ANNUAL_RETURNS,
    LLM_MODEL, MAX_HISTORY_LENGTH
)
from datetime import datetime, timedelta
from market_simulator.utils.db_utils import get_price_history as get_db_price_history
from market_simulator.utils.db_utils import db_manager as db

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

# Initialize last_prices and price_history from DB or defaults
for asset in ASSETS:
    db_prices = db.get_price_history(asset)
    if db_prices:
        last_prices[asset] = db_prices[-1][0]  # Get most recent price
        price_history[asset] = [price for price, _ in db_prices]
    else:
        last_prices[asset] = INITIAL_PRICES[asset]
        price_history[asset] = [INITIAL_PRICES[asset]]

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
        if len(price_history[asset]) > MAX_HISTORY_LENGTH:
            price_history[asset] = price_history[asset][-MAX_HISTORY_LENGTH:]
    else:
        rounded_price = round(price, 2)
        price_history[asset] = [rounded_price]

def makeMarkets():
    """Creates OrderBook objects for all assets"""
    from market_simulator.models.order_book import OrderBook
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
        for i, price in enumerate(reversed(price_history[asset])):
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
        market.lastPrice = initial_prices[market.asset]
    
    # Clear all executional trader intended trades
    for account in accounts.values():
        if hasattr(account, 'intendedOrders'):
            account.intendedOrders.clear()
        if hasattr(account, 'conditionalOrders'):
            account.conditionalOrders.clear()