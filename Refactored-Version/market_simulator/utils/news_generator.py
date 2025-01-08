import threading
import random
from market_simulator.utils.market_utils import model, recentHeadlines, news_queue, chat_queue, markets
from market_simulator.config import (
    WORLD_CONTEXT, NEWS_GENERATION_PROMPT, SENTIMENT_ANALYSIS_PROMPT,
    URGENCY_ANALYSIS_PROMPT, ASSETS, CHAT_ANALYSIS_PROMPT
)

# Global variables to store agent references
_retail_trader = None
_hft_fund = None
_market_maker = None
_long_term_investor = None

def init_agents(retail_trader, hft_fund, market_maker, long_term_investor):
    """Initialize the global agent references"""
    global _retail_trader, _hft_fund, _market_maker, _long_term_investor
    _retail_trader = retail_trader
    _hft_fund = hft_fund
    _market_maker = market_maker
    _long_term_investor = long_term_investor

def generate_news(custom_headline=None):
    """Generates a news headline and updates market sentiment"""
    if custom_headline:
        headline = custom_headline
    else:
        headline = model.invoke(
            NEWS_GENERATION_PROMPT.format(
                world_context=WORLD_CONTEXT,
                recent_headlines=recentHeadlines
            )
        )
    print(headline)
    recentHeadlines.append(headline)
    if len(recentHeadlines) > 10:
        recentHeadlines.pop(0)

    # Get sentiment scores for each asset
    sentiment_scores = model.invoke(
        SENTIMENT_ANALYSIS_PROMPT.format(
            world_context=WORLD_CONTEXT,
            headline=recentHeadlines[-1],
            assets=", ".join(ASSETS)
        )
    )

    # Get urgency score
    urgency_score = model.invoke(
        URGENCY_ANALYSIS_PROMPT.format(
            headline=recentHeadlines[-1]
        )
    )

    try:
        urgency_score = int(urgency_score)
    except:
        urgency_score = 1

    _retail_trader.setReversionUrgency(urgency_score)
    news_queue.put(str(urgency_score) + " " + headline)

    # Parse the sentiment score response
    print(sentiment_scores)
    sentiment_scores_dict = {}
    for asset_sentiment in sentiment_scores.split(', '):
        try:
            if "FILLER" in asset_sentiment:
                continue
            asset, score = asset_sentiment.split(':')
            score = score.replace(" ", "")
            sentiment_scores_dict[asset] = min(0.7, max(0.1, float(score)))
        except:
            print(f"Error parsing sentiment score: {asset_sentiment}")
    
    # Ensure all assets have a sentiment score, defaulting to 0.5 if not set
    for asset in ASSETS:
        if asset not in sentiment_scores_dict:
            sentiment_scores_dict[asset] = 0.5

    _retail_trader.retailSentimentScore = sentiment_scores_dict
    
    # Simulate HFT trading the news
    for market in markets:
        _market_maker.makeMarket(markets[market])
        _hft_fund.tradeTheNews(market, _retail_trader)
        _retail_trader.trade(markets[market])
        _long_term_investor.tradeNews(market, sentiment_scores_dict[market], urgency_score)
        _market_maker.provideLiquidity(markets[market])

def generate_chat():
    """Generates chat messages about market conditions"""
    chat = model.invoke(
        CHAT_ANALYSIS_PROMPT.format(
            recent_headline=recentHeadlines
        )
    )
    print(f"Chat: {chat}")
    chat_queue.put(chat)

def generate_news_thread(headline=None):
    """Creates a thread to generate news"""
    news_thread = threading.Thread(target=generate_news, args=(headline,))
    news_thread.daemon = True
    news_thread.start()

def generate_chat_thread():
    """Creates a thread to generate chat messages"""
    chat_thread = threading.Thread(target=generate_chat)
    chat_thread.daemon = True
    chat_thread.start() 