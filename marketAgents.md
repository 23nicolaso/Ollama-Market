# Market Agents
## Retail Trader
When a new news headline is generated, the Retail Trader agent will update their sentiment score based on the news headline. The sentiment score directly determines the probability of the retail trader buying or selling the asset by market order. This sentiment score then mean reverts over time, based on the urgency of the news headline.  

## High-Frequency Trading (HFT) Fund
The HFT Fund agent reacts quickly to news and market conditions, front-running trades based on sentiment scores. 
When sentiment score is very high or very low, the HFT fund will quickly place large market orders to price the sentiment in, and 
then will slowly close their position in legs and by limit orders to take profit.

## Technical Analysis (TA) Trading Firm
The TA Trading Firm agent uses technical indicators such as moving averages and VWAP bands to make trading decisions. It uses a mixture of mean reversion and momentum strategies. It also places conditional orders at the 500 period high and low to emulate stop loss and take profit strategies.

## Market Maker
The Market Maker agent provides liquidity to the market by constantly quoting a bid and ask spread around the last price traded. It aims to profit from the bid-ask spread.