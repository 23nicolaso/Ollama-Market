# World configuration
WORLD_CONTEXT = """ 
Ensure that all your responses follow the correct, defined response format.     
"""

STATE_STRING = "Traders sell shares in anticipation of key economic data releases"

STATE_PROMPT = """
The state string is your memory of the current situation in this world.
Your old state string is: <{state_string}>.
Please update the state string to better represent the situation in the simulated world with this new headline for context: {headline}
"""

RFR = 0.03 # risk free rate of 3% to start

def get_state_string():
    return STATE_STRING

def set_state_string(new_state: str):
    global STATE_STRING
    import re
    # Extract text between asterisks if present, otherwise use full string
    match = re.search(r'\*([^*]+)\*', new_state)
    STATE_STRING = match.group(1) if match else new_state

# Asset configuration
ASSETS_CONFIG = {
    "SPY": {
        "initial_price": 1050.5,
        "spread": 0.02,
        "num_shares": 10000000,
        "included_in_spy": False,
        "use_rp": False
    },
    "TECHNOLOGY": {
        "initial_price": 400,
        "spread": 0.02,
        "num_shares": 1000000000,
        "included_in_spy": True
    },
    "CONSUMER": {
        "initial_price": 400,
        "spread": 0.02,
        "num_shares": 1000000000,
        "included_in_spy": True
    },
    "HEALTHCARE": {
        "initial_price": 400,
        "spread": 0.02,
        "num_shares": 5000000000,
        "included_in_spy": True
    },
    "FINANCIAL": {
        "initial_price": 400,
        "spread": 0.02,
        "num_shares": 1000000000,
        "included_in_spy": True
    },
    "ENERGY": {
        "initial_price": 400,
        "spread": 0.02,
        "num_shares": 500000000,
        "included_in_spy": True
    },
    "INDUSTRIAL": {
        "initial_price": 400,
        "spread": 0.02,
        "num_shares": 2000000000,
        "included_in_spy": True
    },
    "BITCOIN": {
        "initial_price": 100000,
        "spread": 1,
        "num_shares": 1000000000,
        "included_in_spy": False
    },
    "GOLD": {
        "initial_price": 3000,
        "spread": 0.02,
        "num_shares": 1000000000,
        "included_in_spy": False
    },
    # "VIX": {
    #     "initial_price": 10,
    #     "spread": 0.01,
    #     "num_shares": 1000000,
    #     "included_in_spy": False
    # },
    "TBILLS": {
        "initial_price": 1000,
        "spread": 0.01,
        "num_shares": 1000000,
        "included_in_spy": False
    }
}

RISK_ON_ASSETS = ["TECHNOLOGY", "CONSUMER", "FINANCIAL", "INDUSTRIAL", "BITCOIN"]
RISK_OFF_ASSETS = ["HEALTHCARE", "ENERGY", "GOLD", "TBILLS"]
RISK_FIRM_POSITION_SIZE = 1000000

# Derived configurations
ASSETS = list(ASSETS_CONFIG.keys())
INITIAL_PRICES = {asset: config["initial_price"] for asset, config in ASSETS_CONFIG.items()}
SPREADS = {asset: config["spread"] for asset, config in ASSETS_CONFIG.items()}
NUM_SHARES = {asset: config["num_shares"] for asset, config in ASSETS_CONFIG.items()}
SPY_INCLUDED_ASSETS = [asset for asset, config in ASSETS_CONFIG.items() if config["included_in_spy"]]

# LLM Prompts
NEWS_GENERATION_PROMPT = """
{world_context}
WORLD STATE: <{state_string}>.
Make a huge, impactful news headline for my simulated world.
I want it to be the top headline of the day, summarizing a shocking event, statements by politicians,
developments, natural disasters. Do not make it something generic.
Do not say anything other than the headline, keep your response under 15 words long,
and do not make a x happens as y headline. Do NOT MENTION POINTS, OR CHANGES IN STOCK PRICES.
"""

EXPLANATION_NEWS_PROMPT = """
{world_context}
WORLD STATE: <{state_string}>.
In the financial sector, this data was just released {update}.
Make a headline summarizing this news.
Keep the headline under 15 words long, only respond with the headline itself, and don't mention points or changes in stock prices.
"""

SENTIMENT_ANALYSIS_PROMPT = """
{world_context}
WORLD STATE: <{state_string}>.
NEW NEWS HEADLINE: {headline}

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
WORLD STATE: <{state_string}>.
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

CHAT_PERSONAS = [
    "RetailRandy",
    "WSBDegen",
    "MacroMike",
    "PermaBear",
    "PermaBull",
    "QuantKing",
]

CHAT_ANALYSIS_PROMPT = """You are {persona} in a live market chat room.
WORLD STATE: {state_string}
RECENT HEADLINE: {recent_headline}

Stay in character as {persona}. Write 1-2 punchy sentences reacting to the market, give your predictions.
- RetailRandy: casual retail investor, easily excited or scared, simple language
- WSBDegen: wallstreetbets style, uses 'moon', 'YOLO', 'regards', 'tendies'
- MacroMike: serious macro analyst, references yields, credit spreads, Fed policy
- PermaBear: always bearish, sees doom everywhere, references 2008 constantly
- PermaBull: always bullish, buys every dip, 'stonks only go up'
- QuantKing: speaks in alphas, factors, vol-adjusted returns, Sharpe ratios

No stage directions, no parenthetical emotions, no metadata. Just the chat message."""

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
MAX_HISTORY_LENGTH = 10000
TICK_RESET_THRESHOLD = 10
CHAT_PROBABILITY = 0.01
MAX_RECENT_HEADLINES = 10

# LLM configuration
LLM_MODEL = "gemma3:4b" 

# AGENT QUANTITY CONFIGURATIONS
# Hedge Fund: 
HF_POSITION_LIMIT = 10000
HF_BASE_ORDER_SIZE = 500000

# High-frequency trading fund - makes many short term trades
HFT_BASE_ORDER_SIZE = 50000
HFT_POSITION_LIMIT = 100000

# Market maker - provides liquidity
MM_POSITION_LIMIT = 1000000  # Large position limit to maintain liquidity
MM_BASE_ORDER_SIZE = 3000
MM_MAX_VOLATILITY_MULT = 10
MM_DEPTH = 5
NEARBY_RANGE = 0.05

# Retail traders - many small trades
RETAIL_MAX_ORDER_SIZE = 10000  # Small individual trades
RETAIL_POSITION_LIMIT = 10000000  # Limited position size
USE_CYCLICAL_SENTIMENT = True  # Enable cyclical sentiment for more natural swings
SENTIMENT_REVERSION_RATE = 500  # Faster sentiment changes

# Technical analysis traders - small-sized trades by retail-like investors
TA_MEGA_ORDER_SIZE = 10000  # 
TA_LARGE_ORDER_SIZE = 10000  # 
TA_POSITION_LIMIT = 100000000

# Long-term investor - larger but infrequent trades
LT_INVESTOR_MAX_ORDER_SIZE = 200000  # Increased for more impactful position building
LTI_POS_LIMIT = 20000000 # MASSIVE

TTS_MODEL = "bf_emma"
TTS_LANG = "a"