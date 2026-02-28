import threading
import re
from market_simulator.utils.market_utils import markov_model as mm
from market_simulator.utils.tts_this import tts_this
from market_simulator.utils.market_utils import invoke_model, recentHeadlines, news_queue, chat_queue, action_queue, markets
from market_simulator.config import (
    WORLD_CONTEXT, NEWS_GENERATION_PROMPT, SENTIMENT_ANALYSIS_PROMPT,
    URGENCY_ANALYSIS_PROMPT, ASSETS, SENTIMENT_ANALYSIS_PROMPT_HFT, EXPLANATION_NEWS_PROMPT,
    STATE_PROMPT, set_state_string, get_state_string,
    CHAT_ANALYSIS_PROMPT, CHAT_PERSONAS
)

# Global variables to store agent references
_retail_trader = None
_hft_fund = None
_market_maker = None
_long_term_investor = None
_mean_reversion_fund = None
_risk_on_off_firm = None
_ollama_fund = None
_gemma_fund = None

def init_agents(retail_trader, hft_fund, market_maker, long_term_investor, mean_reversion_fund, risk_on_off_firm, ollama_fund, gemma_fund):
    """Initialize the global agent references"""
    global _retail_trader, _hft_fund, _market_maker, _long_term_investor, _mean_reversion_fund, _risk_on_off_firm, _ollama_fund, _gemma_fund
    _retail_trader = retail_trader
    _hft_fund = hft_fund
    _market_maker = market_maker
    _long_term_investor = long_term_investor
    _mean_reversion_fund = mean_reversion_fund
    _risk_on_off_firm = risk_on_off_firm
    _ollama_fund = ollama_fund
    _gemma_fund = gemma_fund

def generate_news(custom_headline=None, explain_this=None):
    """Generates a news headline and updates market sentiment"""
    # Begin by randomly updating the risk parameters:
    if custom_headline:
        headline = custom_headline
    else:
        if explain_this:
            headline = invoke_model(
                EXPLANATION_NEWS_PROMPT.format(
                    world_context=WORLD_CONTEXT,
                    state_string=get_state_string(),
                    update=explain_this
                )
            )
        else:
            headline = invoke_model(
                NEWS_GENERATION_PROMPT.format(
                    world_context=WORLD_CONTEXT,
                    state_string=get_state_string(),
                )
            )
    recentHeadlines.append(headline)
    if len(recentHeadlines) > 10:
        recentHeadlines.pop(0)

    # Get sentiment scores for each asset
    sentiment_scores = invoke_model(
        SENTIMENT_ANALYSIS_PROMPT.format(
            world_context=WORLD_CONTEXT,
            state_string=get_state_string(),
            headline=recentHeadlines[-1],
            assets=", ".join(ASSETS)
        )
    )

    hft_sentiment_scores = invoke_model(
        SENTIMENT_ANALYSIS_PROMPT_HFT.format(
            world_context=WORLD_CONTEXT,
            state_string=get_state_string(),
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

    updated_state_string = invoke_model(
        STATE_PROMPT.format(
            state_string=get_state_string(),
            headline=recentHeadlines[-1]
        )
    )
    set_state_string(updated_state_string)

    try:
        urgency_score = int(re.search(r'\d+', urgency_score).group())
    except (AttributeError, ValueError):
        urgency_score = 1

    _retail_trader.setReversionUrgency(urgency_score)
    news_queue.put(str(urgency_score) + " " + headline)

    # Parse the sentiment score response
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
                    state_string=get_state_string(),
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

    # Feed per-asset sentiment into retail traders so the hold phase has directional content.
    for asset in ASSETS:
        _retail_trader.updateSentiment(asset, sentiment_scores_dict[asset])

    tts_this(headline, sentiment_scores_dict["SPY"], urgency_score)
    mm.update_on_sentiment(round(sentiment_scores_dict["SPY"],1), urgency_score)
    _market_maker.newsUpdate(urgency_score)
    # Simulate HFT trading the news
    for market in markets:
        _market_maker.makeMarket(markets[market])
        hft_score = hft_sentiment_scores_dict[market]
        _hft_fund.tradeTheNews(market, hft_score)
        direction = "BUY" if hft_score > 0.5 else "SELL"
        action_queue.put(f"HFT: {direction} {market} on news (score {hft_score:.2f})")
        _long_term_investor.tradeNews(market, sentiment_scores_dict[market], urgency_score)

    _ollama_fund.analyzeAndTradeNews(headline)
    _gemma_fund.analyzeAndTradeNews(headline)

    if sentiment_scores_dict["SPY"] > 0.5:
        _risk_on_off_firm.risk_on()
    else:
        _risk_on_off_firm.risk_off()

    _mean_reversion_fund.set_market_return_profile(mm.get_economy_state()['sector_performance'])
    
def generate_chat():
    """Generates an AI persona chat message and posts it to chat_queue."""
    import random
    persona = random.choice(CHAT_PERSONAS)
    recent_headline = recentHeadlines[-1] if recentHeadlines else ""
    message = invoke_model(
        CHAT_ANALYSIS_PROMPT.format(
            persona=persona,
            recent_headline=recent_headline,
            state_string=get_state_string(),
        )
    )
    chat_queue.put(f"[{persona}] {message}")

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