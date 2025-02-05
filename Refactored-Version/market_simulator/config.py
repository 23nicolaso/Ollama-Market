# World configuration
WORLD_CONTEXT = """ 
Ensure that all your responses follow the correct, defined response format and keep your additional thoughts outside the final response.     
"""

# Asset configuration
ASSETS = [
    "SPY",   
    "TECH ETF",
    "CONSUMER ETF",
    "UTILITIES ETF",
    "BANKING ETF",
    "MSCI World ETF",
    "Bitcoin",   
    "Gold",
    "China ETF"           
]

# Market configuration
INITIAL_PRICES = {
    "SPY": 350,
    "TECH ETF": 200,
    "CONSUMER ETF": 200,
    "UTILITIES ETF": 100,
    "BANKING ETF": 200,
    "MSCI World ETF": 100,
    "Bitcoin": 100,
    "Gold": 100,
    "China ETF": 100,
}

SPREADS = {
    "SPY": 0.02,
    "TECH ETF": 0.02,
    "CONSUMER ETF": 0.02,
    "UTILITIES ETF": 0.02,
    "BANKING ETF": 0.02,
    "MSCI World ETF": 0.02,
    "Bitcoin": 0.02,
    "Gold": 0.02,
    "China ETF": 0.02,
}

ANNUAL_RETURNS = {
    "SPY": 0.08,
    "TECH ETF": 0.08,
    "CONSUMER ETF": 0.08,
    "UTILITIES ETF": 0.08,
    "BANKING ETF": 0.08,
    "MSCI World ETF": 0.08,
    "Bitcoin": 0.5,
    "Gold": 0.08,
    "China ETF": 0.08,
}

SPY_INCLUDED_ASSETS = [
    "TECH ETF",
    "CONSUMER ETF",
    "UTILITIES ETF",
    "BANKING ETF"
]

NUM_SHARES = {
    "SPY": 10000000,
    "TECH ETF": 2000000000,
    "CONSUMER ETF": 2000000000,
    "UTILITIES ETF": 2000000000,
    "BANKING ETF": 2000000000,
    "MSCI World ETF": 1000000000,
    "Bitcoin": 1000000000,
    "Gold": 1000000000,
    "China ETF": 1000000000
}

# LLM Prompts
NEWS_GENERATION_PROMPT = """
{world_context}
Make one news headline for my simulated world. 
I want it to be news of economic events, statements by politicians,
developments, natural disasters, or any funny news. 
Do not say anything other than the headline, keep your response under 15 words long,
and do not make a x happens as y headline, simply say an event which happened. Do NOT MENTION POINTS, OR CHANGES IN STOCK PRICES. 
Here are the most recent headlines for context: {recent_headlines}.
"""

SENTIMENT_ANALYSIS_PROMPT = """
{world_context}
Here is the most recent news headline: {headline}

Does this seem like good or bad news for people holding these assets: {assets}

Please provide a sentiment score for each asset based strictly on this format: 
ASSET1: [SCORE], ASSET2: [SCORE], ..., ASSETN: [SCORE]

- Each score should be a number between 0 and 1 (0 means very bad news for the asset, 1 means very good news for the asset).
- Do not include any explanation or analysis, only the asset names and scores, separated by commas.
- Ensure there are no extra spaces, line breaks, or deviations from the format.

For example:
SPY: 0.5, TECH ETF: 0.4, GOLD: 0.7, BITCOIN: 0.2, ...

Now, based on the headline provided, give scores for the following assets: {assets}.
"""

SENTIMENT_ANALYSIS_PROMPT_HFT = """
{world_context}
Here is the most recent news headline: {headline}

You are a HFT analyst. Does this seem like good or bad news for people holding these assets: {assets}

Please provide a sentiment score for each asset based strictly on this format: 
ASSET1: [SCORE], ASSET2: [SCORE], ..., ASSETN: [SCORE]

- Each score should be a number between 0 and 1 (0 means very bad news for the asset, 1 means very good news for the asset).
- Do not include any explanation or analysis, only the asset names and scores, separated by commas.
- Ensure there are no extra spaces, line breaks, or deviations from the format.

For example:
SPY: 0.5, TECH ETF: 0.4, GOLD: 0.7, BITCOIN: 0.2, ...

Now, based on the headline provided, give scores for the following assets: {assets}.
"""

URGENCY_ANALYSIS_PROMPT = """
You are an analyst in a simulated world. Score the impact of this headline between 1-10,
with 1 being not impactful and 10 being the most impactful news of the day: {headline}.
Your final response should just be a number between 1 and 10.
"""

CHAT_ANALYSIS_PROMPT = """
You are a market analyst in a chat room. Give a one sentence explanation of the market conditions and explain how you will profit off them. Feel free to make up information, but keep it realistic (do not include any specific price numbers).
You can either be a serious analyst, a random person, a troll, or talk like a member of wall street bets.
Here is the most recent headlines for context: {recent_headline}
"""

# Agent configuration
POSITION_LIMITS = {asset: 500000 for asset in ASSETS}

ARBITRAGE_THRESHOLD = 0.1
ARB_QUANTITY = 1000

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
LLM_MODEL = "mistral:7b" 

# AGENT QUANTITY CONFIGURATIONS
# High-frequency trading fund - makes many trades on news as soon as it comes out
HF_POSITION_LIMIT = 100000 
HFT_BASE_ORDER_SIZE = 10000

# Market maker - provides liquidity
MM_POSITION_LIMIT = 1000000  # Large position limit to maintain liquidity
MM_BASE_ORDER_SIZE = 500  # Smaller base size for tighter spreads
MM_MAX_VOLATILITY_MULT = 5  # Reduced to prevent extreme price moves
MM_DEPTH = 5  # Increased for more liquidity levels

# Retail traders - many small trades
RETAIL_MAX_ORDER_SIZE = 10  # Small individual trades
RETAIL_POSITION_LIMIT = 50000  # Limited position size
USE_CYCLICAL_SENTIMENT = True  # Enable cyclical sentiment for more natural swings
SENTIMENT_REVERSION_RATE = 500  # Faster sentiment changes

# Technical analysis traders - medium-sized trades
TA_POSITION_LIMIT = 50000  # Moderate position limit
TA_MEGA_ORDER_SIZE = 10000  # Huge, infrequent trades
TA_LARGE_ORDER_SIZE = 1000  # Larger trades for strong signals

# Long-term investor - larger but infrequent trades
LT_INVESTOR_MAX_ORDER_SIZE = 100  # Increased for more impactful position building
