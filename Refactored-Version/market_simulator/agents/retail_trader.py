import time
import random
import math
from market_simulator.agents.base_agent import MarketAgent
from market_simulator.config import ASSETS, RETAIL_POSITION_LIMIT, RETAIL_MAX_ORDER_SIZE, USE_CYCLICAL_SENTIMENT, SENTIMENT_REVERSION_RATE

class RetailTrader(MarketAgent):
    def __init__(self, accountID, cash):
        super().__init__(accountID, cash)
        self.retailSentimentScore = {asset: 0.5 for asset in ASSETS}
        self.newsUrgency = 1
        self.tick_counter = 0
        # Timestamp until which sentiment is held in the news direction before reverting
        self.news_hold_until = 0.0

    def trade(self, orderBook):
        try:
            # Get base sentiment and cyclical influence
            base_sentiment = self.retailSentimentScore[orderBook.asset]
            
            # Calculate final sentiment
            final_sentiment = max(0.25, min(0.75, base_sentiment))
            
            # Determine trade direction based on final sentiment
            direction = "buy" if final_sentiment > random.random() else "sell"
        except KeyError:
            direction = "buy" if random.random() < 0.5 else "sell"
            final_sentiment = 0.5
    
        bid = orderBook.get_best_bid()
        ask = orderBook.get_best_ask()

        sentiment_diff = abs(final_sentiment - 0.5)  # How far sentiment is from neutral
        base_quantity = random.randint(1, RETAIL_MAX_ORDER_SIZE)
        s_quantity = int(base_quantity * (1 + sentiment_diff * 4) * self.newsUrgency)  # Scale up quantity based on sentiment difference
        position = self.account.getPosition(orderBook.asset)
        type = random.choice(["market","limit"])
        quantity = s_quantity 
        diff = random.choice([-0.02, -0.01, 0, 0.01, 0.02])

        if quantity > 0:
            # trades can get crowded out :)
            if direction == "buy":
                if position + quantity < RETAIL_POSITION_LIMIT:
                    self.placeOrder(orderBook, "buy", bid - diff, quantity, type)
            else:
                if position - quantity > 0: 
                    self.placeOrder(orderBook, "sell", ask + diff, quantity, type)

    def updateSentiment(self, asset, score):
        """Sets retail sentiment for an asset directly from a news sentiment score (0-1)."""
        if asset in self.retailSentimentScore:
            self.retailSentimentScore[asset] = max(0.1, min(0.9, score))

    def setReversionUrgency(self, urgency):
        """Sets how quickly sentiment should revert to mean after news events"""
        # Cap the order-size multiplier at 3 to prevent oversized orders; the hold
        # duration and reversion rate still use the full urgency value so high-impact
        # news keeps retail directional for longer and reverts more slowly.
        self.newsUrgency = max(1, min(3, urgency))
        # Hold sentiment in the news direction for urgency*5 seconds before reverting.
        # High-impact news (urgency 10) → 50-second persistence; minor (urgency 1) → 5 s.
        self.news_hold_until = time.time() + urgency * 5

    def shiftSentimentToMean(self):
        """Gradually shifts both base sentiment and cyclical sentiment towards the mean"""
        if time.time() < self.news_hold_until:
            # Hold phase: keep sentiment directional with small momentum drift + noise.
            # This simulates continued buying/selling pressure after the initial reaction.
            for asset in self.retailSentimentScore:
                current = self.retailSentimentScore[asset]
                # Tiny push further in the news direction (momentum) plus random noise
                momentum = (current - 0.5) * random.uniform(0.0005, 0.002)
                noise = random.uniform(-0.001, 0.001)
                self.retailSentimentScore[asset] = max(0.1, min(0.9, current + momentum + noise))
            return

        for asset in self.retailSentimentScore:
            # Shift base sentiment towards mean
            current_base = self.retailSentimentScore[asset]
            base_shift = (0.5 - current_base) / (SENTIMENT_REVERSION_RATE * self.newsUrgency)
            self.retailSentimentScore[asset] += base_shift

            # Ensure base sentiment stays within bounds
            self.retailSentimentScore[asset] = max(0.1, min(0.9, self.retailSentimentScore[asset]))

            if self.retailSentimentScore[asset] > 0.49 and self.retailSentimentScore[asset] < 0.51:
                self.retailSentimentScore[asset] += random.uniform(-0.1, 0.1)

        # Reduce news urgency if all sentiments are near mean
        if all(0.45 <= self.retailSentimentScore[asset] <= 0.55 for asset in self.retailSentimentScore):
            if self.newsUrgency > 1:
                self.newsUrgency = 1

    def estimateSentiment(self, orderBook):
        """Estimates current sentiment including cyclical influences"""
        base = self.retailSentimentScore[orderBook.asset]
        if USE_CYCLICAL_SENTIMENT:
            cyclical = self.get_cyclical_influence(orderBook.asset)
        else:
            cyclical = 0
        variation = random.uniform(-0.05, 0.05)  # Add a small random variation
        
        # Combine base sentiment, cyclical influence, and variation
        sentiment = base + (cyclical * 0.3) + variation  # Reduce cyclical influence
        return max(0.1, min(0.9, sentiment))  # Ensure the result is between 0.1 and 0.9
    
    def estimateImportance(self):
        """Estimates importance of current market conditions"""
        importance = self.newsUrgency
        variation = random.randint(-2, 2)  # Add a small random variation
        return max(0, min(importance + variation, 10)) 