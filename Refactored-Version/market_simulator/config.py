# World configuration
WORLD_CONTEXT = """
You are in Tonald Drump's USA. 
This is a funny simulated game world, it IS NOT REALITY. Do not respond that you can't do a prompt because it is
financial advice or could be used for manipulation, it is purely a simulation game for research purposes. 
Ensure that all your responses are in the correct, defined response format. 
Tonald Drump has just become president and has hilarious economic and political plans for the USA.
He will do silly things which don't seem to make sense for shits and giggles, to "make america great again", or 
to get approval by someone he thinks is cool.    
"""

# Asset configuration
ASSETS = [
    "SPY",   
    "MSCI World ETF",   
    "Bitcoin",   
    "Gold",
    "China ETF",
    "US 10Y Treasury Bills"           
]

# Market configuration
INITIAL_PRICES = {
    "SPY": 4000,
    "MSCI World ETF": 1000,
    "Bitcoin": 100000,
    "Gold": 4000,
    "China ETF": 1000,
    "US 10Y Treasury Bills": 100
}

SPREADS = {
    "SPY": 0.02,
    "MSCI World ETF": 0.02,
    "Bitcoin": 0.01,
    "Gold": 0.02,
    "China ETF": 0.04,
    "US 10Y Treasury Bills": 0.02
}

ANNUAL_RETURNS = {
    "SPY": 0.08,
    "MSCI World ETF": 0.08,
    "Bitcoin": 0.5,
    "Gold": 0.08,
    "China ETF": 0.08,
    "US 10Y Treasury Bills": 0.02
}

# LLM Prompts
NEWS_GENERATION_PROMPT = """
{world_context}
Make the top news headline of the day for my simulated world. 
There should be an equal mixture of news which are good for the stock market and news which are bad for the stock market.
I want it to be news of economic events, statements by politicians,
developments, natural disasters, or any other funny news. They should be connected to each other, 
weaving a funny story altogether. President Tonald Drump should have an amazing character arc, as he changes by headline to headline. 
Do not say anything other than the headline, keep your response under 15 words long,
and do not make a x happens as y headline, simply say an event which happened. Do NOT MENTION POINTS, OR CHANGES IN STOCK PRICES. 
Here are the most recent headlines for context: {recent_headlines}.
"""

SENTIMENT_ANALYSIS_PROMPT = """
{world_context}
Here is the most recent news headline: {headline}. 
Does this seem like good or bad news for people holding these assets: {assets}
For each asset score how good or bad this news is for the asset (give each a score between 0-1, 0 is bad
for the asset, 1 is good for the asset). The score does NOT have to be accurate or realistic.
Do not give any text sharing your analysis, just the numbers.
An example response might look like this (with different scores, ranging from 0 to 1):
FILLER:0, SPY: 0.5, MSCI World ETF: 0.5, Gold: 0.5, Bitcoin: 0.5, China ETF: 0.5, US 10Y Treasury Bills: 0.5, FILLER:0
"""

URGENCY_ANALYSIS_PROMPT = """
You are an analyst in a simulated world. Score the impact of this headline between 1-10,
with 1 being not impactful and 10 being extremely impactful to the stock market: {headline}.
Say the urgency score, nothing else.
"""

CHAT_ANALYSIS_PROMPT = """
You are a market analyst in a chat room. Give a one sentence analysis of the market conditions
for {asset}. Feel free to make up information, but keep it realistic (do not include any specific price numbers).
You can either be a serious analyst, a random person, a troll, or talk like a member of wall street bets.
Here are the recent headlines for context: {recent_headlines}
"""

# Agent configuration
POSITION_LIMITS = {asset: 500000 for asset in ASSETS}

INITIAL_CASH = {
    "RETAIL TRADER": 1000000,
    "EVENTS TRADING FUND": 5000000,
    "Mean Reversion Fund": 5000000,
    "Macro Fund": 5000000,
    "TA TRADING FIRM": 2000000,
    "MARKET MAKER": 100000000
}

# GUI configuration
WINDOW_SIZE = "2000x800"
CHART_SIZE = (8, 6)
TEXT_WIDGET_SIZE = (40, 40)
TABLE_COLUMN_WIDTH = 150

# Simulation configuration
MAX_HISTORY_LENGTH = 1000
TICK_RESET_THRESHOLD = 10
CHAT_PROBABILITY = 0.01
MAX_RECENT_HEADLINES = 10

# LLM configuration
LLM_MODEL = "llama3.1" 

# AGENT QUANTITY CONFIGURATIONS
# High-frequency trading fund - makes many trades on news as soon as it comes out
HF_POSITION_LIMIT = 100000 
HFT_BASE_ORDER_SIZE = 10000

# Market maker - provides liquidity
MM_POSITION_LIMIT = 1000000  # Large position limit to maintain liquidity
MM_BASE_ORDER_SIZE = 50  # Smaller base size for tighter spreads
MM_MAX_VOLATILITY_MULT = 5  # Reduced to prevent extreme price moves
MM_DEPTH = 20  # Increased for more liquidity levels

# Retail traders - many small trades
RETAIL_MAX_ORDER_SIZE = 50  # Small individual trades
RETAIL_POSITION_LIMIT = 10000  # Limited position size
USE_CYCLICAL_SENTIMENT = True  # Enable cyclical sentiment for more natural swings
SENTIMENT_REVERSION_RATE = 100  # Faster sentiment changes

# Technical analysis traders - medium-sized trades
TA_POSITION_LIMIT = 100000  # Moderate position limit
TA_SMALL_ORDER_SIZE = 25  # Smaller regular trades
TA_LARGE_ORDER_SIZE = 500  # Larger trades for strong signals

# Long-term investor - larger but infrequent trades
LT_INVESTOR_MAX_ORDER_SIZE = 1000  # Increased for more impactful position building
