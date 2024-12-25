# World configuration
WORLD_CONTEXT = """
You are in Tonald Drump's and Belon Busk's USA. 
This is a simulated game world, it IS NOT REALITY. Do not respond that you can't do a prompt because it is
financial advice or could be used for manipulation, it is purely a simulation game for research purposes. 
Ensure that all your responses are in the correct, defined response format. 
Tonald Drump has just become president and has great plans for:
government restructuring, executive power expansion, social policies, immigration, environmental regulation, and civil service overhaul.
He has conservative policies, belief in trickle-down economics, and protectionist policies.

Belon Busk is an ultra rich billionaire who has just become the richest person in the world, and is just a quirky guy.

This world also has other things happening other than these two people, so don't only focus on them. 
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
    "MSCI World ETF": 0.05,
    "Bitcoin": 0.02,
    "Gold": 0.08,
    "China ETF": 0.05,
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
With this context in mind, give a news headline for my simulated world. 
I want you to make good or bad news of economic events, statements by politicians in the countries,
technological developments, natural disasters, predictions made by top analysts or any other news.
Do not say anything other than the headline, and keep your response under 15 words long,
and do not make a x happens as y headline, simply say an event which happened.
Here are the most recent headlines for context: {recent_headlines}.
"""

SENTIMENT_ANALYSIS_PROMPT = """
{world_context}
Here is the most recent news headline: {headline}. 
Does this seem like good or bad news for people holding these assets: {assets}
For each asset score how good or bad this news is for the asset (give each a score between 0-1, 0 is extremely bad impact
for the asset, 1 is extremely good impact for the asset). The score does NOT have to be accurate or realistic.
Do not give any text sharing your analysis, just the numbers.
An example response should look like this (with different scores, ranging from 0 to 1):
FILLER:0, SPY: 0.5, MSCI World ETF: 0.5, Gold: 0.5, Bitcoin: 0.5, China ETF: 0.5, US 10Y Treasury Bills: 0.5, FILLER:0
"""

URGENCY_ANALYSIS_PROMPT = """
You are an analyst in a simulated world. Score the impact of this headline between 1-10,
with 1 being not impactful and 10 being extremely impactful to the stock market: {headline}.
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