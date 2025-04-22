# World configuration
WORLD_CONTEXT = """ 
Ensure that all your responses follow the correct, defined response format.     
"""

STATE_STRING = "President Sterling has just been elected in the US government, and is launching widespread reforms!"

STATE_PROMPT = f"""
The state string is your memory of the current situation in this world. 
Your old state string is: <{STATE_STRING}>. 
Please update the state string to better represent the situation in the simulated world with this new headline for context: {{headline}}
"""

RFR = 0.03 # risk free rate of 3% to start

def get_state_string():
    return STATE_STRING

def set_state_string(str):
    global STATE_STRING
    import re
    # Extract text between asterisks if present, otherwise use full string
    match = re.search(r'\*([^*]+)\*', str)
    STATE_STRING = match.group(1) if match else str
    print(STATE_STRING)

# Asset configuration
ASSETS_CONFIG = {
    "SPY": {
        "initial_price": 350,
        "spread": 0.02,
        "annual_return": 0.08,
        "num_shares": 10000000,
        "included_in_spy": False,
        "use_rp": False
    },
    "TECH ETF": {
        "initial_price": 200,
        "spread": 0.02,
        "annual_return": 0.08,
        "num_shares": 2000000000,
        "risk": 0.05,
        "included_in_spy": True,
        "use_rp": True
    },
    "CONSUMER ETF": {
        "initial_price": 200,
        "spread": 0.02,
        "annual_return": 0.08,
        "num_shares": 2000000000,
        "risk": 0.04,
        "included_in_spy": True,
        "use_rp": True
    },
    "UTILITIES ETF": {
        "initial_price": 100,
        "spread": 0.02,
        "annual_return": 0.09,
        "num_shares": 2000000000,
        "risk": 0.02,
        "included_in_spy": True,
        "use_rp": True
    },
    "BANKING ETF": {
        "initial_price": 200,
        "spread": 0.02,
        "annual_return": 0.08,
        "num_shares": 2000000000,
        "risk": 0.03,
        "included_in_spy": True,
        "use_rp": True
    },
    "MSCI World ETF": {
        "initial_price": 100,
        "spread": 0.02,
        "annual_return": 0.08,
        "risk":0.03,
        "num_shares": 1000000000,
        "included_in_spy": False,
        "use_rp": True
    },
    "Bitcoin": {
        "initial_price": 100,
        "spread": 0.02,
        "annual_return": 0.08,
        "num_shares": 1000000000,
        "included_in_spy": False,
        "use_rp": False
    },
    "Gold": {
        "initial_price": 100,
        "spread": 0.02,
        "annual_return": 0.08,
        "num_shares": 1000000000,
        "included_in_spy": False,
        "use_rp": False
    },
    "China ETF": {
        "initial_price": 100,
        "spread": 0.02,
        "annual_return": 0.12,
        "num_shares": 1000000000,
        "risk": 0.06,
        "included_in_spy": False,
        "use_rp": True
    }
}

# Derived configurations
ASSETS = list(ASSETS_CONFIG.keys())
INITIAL_PRICES = {asset: config["initial_price"] for asset, config in ASSETS_CONFIG.items()}
SPREADS = {asset: config["spread"] for asset, config in ASSETS_CONFIG.items()}
ANNUAL_RETURNS = {asset: config["annual_return"] for asset, config in ASSETS_CONFIG.items()}
NUM_SHARES = {asset: config["num_shares"] for asset, config in ASSETS_CONFIG.items()}
SPY_INCLUDED_ASSETS = [asset for asset, config in ASSETS_CONFIG.items() if config["included_in_spy"]]

# LLM Prompts
NEWS_GENERATION_PROMPT = f"""
{{world_context}}
The state string is your memory of the current situation in this world. 
Your old state string is: <{STATE_STRING}>. 
Make one dramatic breaking news headline for my simulated world. 
I want it to be news of economic events, statements by politicians,
developments, natural disasters, or any funny news. 
Do not say anything other than the headline, keep your response under 15 words long,
and do not make a x happens as y headline, simply say an event which happened. Do NOT MENTION POINTS, OR CHANGES IN STOCK PRICES. 
Here are the most recent headlines for context: {{recent_headlines}}.
"""

EXPLANATION_NEWS_PROMPT = f"""
{{world_context}}
The state string is your memory of the current situation in this world. 
Your old state string is: <{STATE_STRING}>. 
In the financial sector, this data was just released {{update}}. 
Make a headline summarizing this news.
Keep the headline under 15 words long, only respond with the headline itself, and don't mention points or changes in stock prices. 
"""

SENTIMENT_ANALYSIS_PROMPT = f"""
{{world_context}}
The state string is your memory of the current situation in this world. 
Your old state string is: <{STATE_STRING}>. 
Here is the most recent news headline: {{headline}}

Does this seem like good or bad news for people holding these assets: {{assets}}

Please provide a sentiment score for each asset based strictly on this format: 
ASSET1: [SCORE], ASSET2: [SCORE], ..., ASSETN: [SCORE]

- Each score should be a number between 0 and 1 (0 means very bad news for the asset, 1 means very good news for the asset).
- Do not include any explanation or analysis, only the asset names and scores, separated by commas.
- Ensure there are no extra spaces, line breaks, or deviations from the format.

For example:
SPY: 0.5, TECH ETF: 0.4, GOLD: 0.7, BITCOIN: 0.2, ...

Now, based on the headline provided, give scores for the following assets: {{assets}}.
"""

SENTIMENT_ANALYSIS_PROMPT_HFT = f"""
{{world_context}}
The state string is your memory of the current situation in this world. 
Your old state string is: <{STATE_STRING}>. 
Here is the most recent news headline: {{headline}}

You are a HFT analyst. Does this seem like good or bad news for people holding these assets: {{assets}}

Please provide a sentiment score for each asset based strictly on this format: 
ASSET1: [SCORE], ASSET2: [SCORE], ..., ASSETN: [SCORE]

- Each score should be a number between 0 and 1 (0 means very bad news for the asset, 1 means very good news for the asset).
- Do not include any explanation or analysis, only the asset names and scores, separated by commas.
- Ensure there are no extra spaces, line breaks, or deviations from the format.

For example:
SPY: 0.5, TECH ETF: 0.4, GOLD: 0.7, BITCOIN: 0.2, ...

Now, based on the headline provided, give scores for the following assets: {{assets}}.
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

ARBITRAGE_THRESHOLD = 0.05
ARB_QUANTITY = 10000

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
MAX_HISTORY_LENGTH = 500
TICK_RESET_THRESHOLD = 10
CHAT_PROBABILITY = 0.01
MAX_RECENT_HEADLINES = 10

# LLM configuration
LLM_MODEL = "gemma3:4b-it-qat" 

# AGENT QUANTITY CONFIGURATIONS
# High-frequency trading fund - makes many trades on news as soon as it comes out
HF_POSITION_LIMIT = 50000000
HF_BASE_ORDER_SIZE = 1000000
HFT_BASE_ORDER_SIZE = 10000
HFT_POSITION_LIMIT = 100000

# Market maker - provides liquidity
MM_POSITION_LIMIT = 1000000  # Large position limit to maintain liquidity
MM_BASE_ORDER_SIZE = 2000 
MM_MAX_VOLATILITY_MULT = 10
MM_DEPTH = 5
NEARBY_RANGE = 0.05

# Retail traders - many small trades
RETAIL_MAX_ORDER_SIZE = 2500  # Small individual trades
RETAIL_POSITION_LIMIT = 100000  # Limited position size
USE_CYCLICAL_SENTIMENT = True  # Enable cyclical sentiment for more natural swings
SENTIMENT_REVERSION_RATE = 500  # Faster sentiment changes

# Technical analysis traders - small-sized trades by retail-like investors
TA_POSITION_LIMIT = 1000000  # 
TA_MEGA_ORDER_SIZE = 10000  # 
TA_LARGE_ORDER_SIZE = 1000  # 

# Long-term investor - larger but infrequent trades
LT_INVESTOR_MAX_ORDER_SIZE = 50000  # Increased for more impactful position building
LTI_POS_LIMIT = 20000000 # MASSIVE
