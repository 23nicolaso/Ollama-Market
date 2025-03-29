import threading
import re
from market_simulator.utils.market_utils import invoke_model, recentHeadlines, news_queue, chat_queue, markets, calculate_fair_value
from market_simulator.config import (
    WORLD_CONTEXT, NEWS_GENERATION_PROMPT, SENTIMENT_ANALYSIS_PROMPT,
    URGENCY_ANALYSIS_PROMPT, ASSETS, SENTIMENT_ANALYSIS_PROMPT_HFT, EXPLANATION_NEWS_PROMPT
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

def generate_news(custom_headline=None, explain_this=None):
    """Generates a news headline and updates market sentiment"""
    # Begin by randomly updating the risk parameters:
    if custom_headline:
        headline = custom_headline
    else:
        if explain_this:
            risk = explain_this[0]
            asset = explain_this[1]

            headline = invoke_model(
                EXPLANATION_NEWS_PROMPT.format(
                    world_context=WORLD_CONTEXT,
                    risk=risk,
                    asset=asset
                )
            )
        else:
            headline = invoke_model(
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
    sentiment_scores = invoke_model(
        SENTIMENT_ANALYSIS_PROMPT.format(
            world_context=WORLD_CONTEXT,
            headline=recentHeadlines[-1],
            assets=", ".join(ASSETS)
        )
    )

    hft_sentiment_scores = invoke_model(
        SENTIMENT_ANALYSIS_PROMPT_HFT.format(
            world_context=WORLD_CONTEXT,
            headline=recentHeadlines[-1],
            assets=", ".join(ASSETS)
        )
    )

    # Get urgency score
    urgency_score = invoke_model(
        URGENCY_ANALYSIS_PROMPT.format(
            headline=recentHeadlines[-1]
        )
    )

    try:
        urgency_score = int(re.search(r'\d+', urgency_score).group())
    except:
        print("Error parsing urgency score")
        urgency_score = 1

    _retail_trader.setReversionUrgency(urgency_score)
    news_queue.put(str(urgency_score) + " " + headline)

    # Parse the sentiment score response
    print(sentiment_scores)
    pattern = r'(\w[\w\s]*):\s*([\d.]+)' # apply regex to the sentiment scores
    sentiment_scores_dict = {}
    hft_sentiment_scores_dict = {}
    scores = {match[0]: float(match[1]) for match in re.findall(pattern, sentiment_scores)}
    hft_scores = {match[0]: float(match[1]) for match in re.findall(pattern, hft_sentiment_scores)}

    # First attempt to get sentiment scores
    for asset in ASSETS:
        if asset in scores:
            sentiment_scores_dict[asset] = scores[asset]
        else:
            # Try one more time for missing assets
            retry_sentiment = invoke_model(
                SENTIMENT_ANALYSIS_PROMPT.format(
                    world_context=WORLD_CONTEXT,
                    headline=recentHeadlines[-1],
                    assets=asset
                )
            )
            retry_scores = {match[0]: float(match[1]) for match in re.findall(pattern, retry_sentiment)}
            
            # Use retry score if successful, otherwise default to 0.5
            if asset in retry_scores:
                sentiment_scores_dict[asset] = retry_scores[asset]
            else:
                sentiment_scores_dict[asset] = 0.5

    for asset in ASSETS:
            if asset in hft_scores:
                hft_sentiment_scores_dict[asset] = hft_scores[asset]
            else:
                hft_sentiment_scores_dict[asset] = 0.5

    _retail_trader.retailSentimentScore = sentiment_scores_dict
    
    # Simulate HFT trading the news
    for market in markets:
        _market_maker.makeMarket(markets[market])
        _hft_fund.tradeTheNews(market, hft_sentiment_scores_dict[market])
        _retail_trader.trade(markets[market])
        _long_term_investor.tradeNews(market, sentiment_scores_dict[market], urgency_score)
        _market_maker.provideLiquidity(markets[market])

def generate_chat():
    """Generates chat messages about market conditions"""
    return
    # chat = invoke_model(
    #     CHAT_ANALYSIS_PROMPT.format(
    #         recent_headline=recentHeadlines
    #     )
    # )
    # print(f"Chat: {chat}")
    # chat_queue.put(chat)

def generate_news_thread(headline=None, explain_this=None):
    """Creates a thread to generate news"""
    news_thread = threading.Thread(target=generate_news, args=(headline,explain_this))
    news_thread.daemon = True
    news_thread.start()

def generate_chat_thread():
    """Creates a thread to generate chat messages"""
    chat_thread = threading.Thread(target=generate_chat)
    chat_thread.daemon = True
    chat_thread.start() 