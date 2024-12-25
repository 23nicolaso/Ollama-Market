# World configuration
WORLD_CONTEXT = """
In this simulated world, there are four major countries:
- Simula: A technology-focused nation leading in AI and software
- Rivala: An agricultural powerhouse with vast natural resources
- Allia: An entertainment and media hub, ally to Rivala
- Factoria: An industrial nation and weapons manufacturer

These countries are economically interdependent. Simula and Rivala occasionally enter conflicts,
with Allia supporting Rivala and Factoria supplying weapons to all sides.
"""

# Asset configuration
ASSETS = [
    "Simula 500",    # Tech sector index
    "Rivala ETF",    # Agricultural sector fund
    "Allia ETF",     # Entertainment sector fund
    "Factoria ETF",  # Industrial sector fund
    "Gold"           # Safe haven asset
]

# Market configuration
INITIAL_PRICES = {
    "Simula 500": 100,
    "Rivala ETF": 100,
    "Allia ETF": 100,
    "Factoria ETF": 100,
    "Gold": 4000
}

SPREADS = {
    "Simula 500": 0.02,
    "Rivala ETF": 0.05,
    "Allia ETF": 0.02,
    "Factoria ETF": 0.08,
    "Gold": 0.05
}

ANNUAL_RETURNS = {
    "Simula 500": 0.3,
    "Rivala ETF": 0.15,
    "Allia ETF": 0.12,
    "Factoria ETF": 0.09,
    "Gold": 0.05
}

# LLM Prompts
NEWS_GENERATION_PROMPT = """
Give a news headline for my simulated world. {world_context}
I want you to make good or bad news of economic events, statements by politicians in the countries,
technological developments, natural disasters, or predictions made by top analysts.
Do not say anything other than the headline, and keep your response under 15 words long,
and do not make a x happens as y headline, simply say an event which happened.
Here are the most recent headlines for context: {recent_headlines}.
"""

SENTIMENT_ANALYSIS_PROMPT = """
{world_context}
All these countries are reliant on each others economies, and their governments intervene
when their economy is in danger. Here is the most recent news headline: {headline}.
Predict economic sentiment for investors in the Simula 500, Rivala ETF, Allia ETF, Factoria ETF
and list them as a comma seperated list (give each a score between 0-1, 0 is extremely impact
on the economy, 1 is extremely good impact on the economy).
Do not say anything other than the sentiment score.
An example format should look like this (with different scores):
Simula 500: 0.5, Rivala ETF: 0.5, Allia ETF: 0.5, Factoria ETF: 0.5
"""

URGENCY_ANALYSIS_PROMPT = """
You are an analyst in a simulated world. Score the impact of this headline between 1-10,
with 1 being not impactful and 10 being extremely impactful: {headline}.
Say the urgency score, nothing else.
"""

CHAT_ANALYSIS_PROMPT = """
You are a market analyst in a chat room. Give a one sentence analysis of the market conditions
for {asset}. Keep it under 15 words and make it sound natural.
Here are the recent headlines for context: {recent_headlines}
"""

# Agent configuration
POSITION_LIMITS = {asset: 100000 for asset in ASSETS}

INITIAL_CASH = {
    "RETAIL TRADER": 1000000,
    "EVENTS TRADING FUND": 10000000,
    "Mean Reversion Fund": 10000000,
    "Macro Fund": 10000000,
    "TA TRADING FIRM": 1000000,
    "MARKET MAKER": 100000000000000
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
NEWS_PROBABILITY = 0.001
MAX_RECENT_HEADLINES = 10

# LLM configuration
LLM_MODEL = "llama3.1" 