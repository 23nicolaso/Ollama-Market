from langchain_ollama import OllamaLLM
import queue
from market_simulator.config import (
    ASSETS, INITIAL_PRICES, SPREADS, ANNUAL_RETURNS,
    LLM_MODEL, MAX_HISTORY_LENGTH
)

# Add these to the global variables
news_queue = queue.Queue()
chat_queue = queue.Queue()

# Global variables
accounts = {}  # Stores list of Account objects, indexed by accountID
initial_prices = INITIAL_PRICES
last_prices = INITIAL_PRICES.copy()  # Stores last price for each asset
spreads_by_market = SPREADS
average_annual_return_by_market = ANNUAL_RETURNS
economic_health_by_market = {asset: 1 for asset in ASSETS}
simulation_age = 0  # Stores the age of the simulation in total sets of 10 ticks
price_history = {asset: [price] for asset, price in last_prices.items()}  # Stores list of past prices for each asset
assets = ASSETS  # Stores list of assets
markets = {}  # Stores list of OrderBook objects, indexed by asset
recentHeadlines = []  # Stores list of recently generated headlines

# Initialize LLM
model = OllamaLLM(model=LLM_MODEL)

def update_price_history(asset, price):
    """Updates the price history for a given asset"""
    if asset in price_history:
        price_history[asset].append(price)
        if len(price_history[asset]) > MAX_HISTORY_LENGTH:
            price_history[asset] = price_history[asset][-MAX_HISTORY_LENGTH:]
    else:
        price_history[asset] = [price]

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