from market_simulator.models.order_level import OrderLevel
from market_simulator.utils.market_utils import last_prices, accounts
from market_simulator.price_server import emit_trade_update
from market_simulator.portfolio_utils import calculate_portfolio_status, emit_portfolio_update

class OrderBook:
    def __init__(self, asset, initialPrice):
        self.asset = asset
        self.bids = {}
        self.asks = {}
        self.urgentBuys = []
        self.urgentSells = []
        last_prices[asset] = initialPrice

    def getBids(self):
        return self.bids

    def getAsks(self):
        return self.asks
    
    def getMidPrice(self):
        if not self.bids or not self.asks:
            return None
        return (self.getBestBid().getPrice() + self.getBestAsk().getPrice()) / 2

    def getUrgentOrders(self):
        return self.urgentBuys, self.urgentSells

    def cancelAllOldOrders(self):
        for bid in self.bids.values():
            bid.cancelOldOrders()
        for ask in self.asks.values():
            ask.cancelOldOrders()
        self.clearEmptyOrderlevels()

    def cancelOrdersByAccount(self, accountID):
        for bid in self.bids.values():
            bid.cancelOrdersFromID(accountID)
        for ask in self.asks.values():
            ask.cancelOrdersFromID(accountID)

    def display(self):
        print(f"{self.asset} Orderbook")
        print(f"Total Buy Orders: {len(self.bids)}")
        print(f"Total Sell Orders: {len(self.asks)}")
        print(f"Last Price: {self.getLastPrice()}")
        print(f"Bids: {[str(value.netQuantity) + ' at '  + str(value.price) for value in sorted(self.bids.values(), key=lambda x: x.price, reverse=True)]}")
        print(f"Asks: {[str(value.netQuantity) + ' at ' + str(value.price) for value in sorted(self.asks.values(), key=lambda x: x.price)]}")
        print(f"Bid Size: {self.getBidSize()}")
        print(f"Ask Size: {self.getAskSize()}")

    def displayPrice(self):
        print(f"{self.asset} price: {round(self.getLastPrice(), 2)}")

    def getBidSize(self):
        return sum([bid.netQuantity for bid in self.bids.values()])

    def getAskSize(self):
        return sum([ask.netQuantity for ask in self.asks.values()])

    def clearFarOrders(self):
        # To stop my computer from exploding, I clear orders that are too far away from the current price
        for bid in self.bids.values():
            if bid.price < self.getLastPrice() - 5:
                bid.cancelAll()

        for ask in self.asks.values():
            if ask.price > self.getLastPrice() + 5:
                ask.cancelAll()
        
        self.clearEmptyOrderlevels()
 
    def clearEmptyOrderlevels(self):
        bid_prices = list(self.bids.keys())
        ask_prices = list(self.asks.keys())
        
        while bid_prices:
            price = bid_prices.pop()
            if self.bids[price].netQuantity == 0:
                del self.bids[price]
        
        while ask_prices:
            price = ask_prices.pop()
            if self.asks[price].netQuantity == 0:
                del self.asks[price]

    def addOrder(self, direction, price, quantity, orderType, accountID):
        book = self.bids if direction == "buy" else self.asks
        if orderType=="limit":
            if price in book:
                book[price].addOrder(quantity, accountID)
            else:
                book[price] = OrderLevel(price, quantity, self.asset, 1 if direction == "buy" else -1)
                book[price].addOrder(quantity, accountID)
            self.matchBooks()
        else:
            if direction == "buy":
                self.urgentBuys.append((quantity, accountID))
            else:
                self.urgentSells.append((quantity, accountID))

            self.fillUrgentOrders()

    def getBestBid(self):
        if self.bids:
            return self.bids[max(self.bids.keys())]
        else:
            return None

    def getBestAsk(self):
        if self.asks:
            return self.asks[min(self.asks.keys())]
        else:
            return None

    def matchBooks(self):
        while True:
            bestBid = self.getBestBid()
            bestAsk = self.getBestAsk()
            if not bestBid or not bestAsk or bestBid.getPrice() < bestAsk.getPrice():
                break
            
            quantityToFill = min(bestBid.getQuantity(), bestAsk.getQuantity())
            
            # Get the account IDs from the orders
            buyAccountID = bestBid.getAccountID()
            sellAccountID = bestAsk.getAccountID()

            if not buyAccountID or not sellAccountID:
                break # invalid book matching, no buy/sell accounts. 
            
            # Execute the trade
            bestBid.fulfillQuantity(quantityToFill)
            bestAsk.fulfillQuantity(quantityToFill)
            
            # Emit trade updates
            tradePrice = bestBid.getPrice()  # Could also use bestAsk.getPrice() as they're equal
            emit_trade_update(buyAccountID, self.asset, "buy", quantityToFill, tradePrice)
            emit_trade_update(sellAccountID, self.asset, "sell", quantityToFill, tradePrice)

            if bestBid.getQuantity() == 0:
                self.bids.pop(bestBid.getPrice())
            if bestAsk.getQuantity() == 0:
                self.asks.pop(bestAsk.getPrice())

    def getUnfilledUrgentOrders(self):
        return self.urgentBuys, self.urgentSells

    def fillUrgentOrders(self):
        filledOrders = 0
        # match buys and sells at the middle of the bid and ask
        while self.urgentBuys and self.urgentSells:
            if self.urgentBuys[0][1] == self.urgentSells[0][1]:
                quantityFilled = min(self.urgentBuys[0][0], self.urgentSells[0][0])
                self.urgentBuys[0] = (self.urgentBuys[0][0] - quantityFilled, self.urgentBuys[0][1])
                self.urgentSells[0] = (self.urgentSells[0][0] - quantityFilled, self.urgentSells[0][1])
                if self.urgentSells[0][0] == 0:
                    self.urgentSells.pop(0)
                if self.urgentBuys[0][0] == 0:
                    self.urgentBuys.pop(0)
                continue
            sellAccountID = self.urgentSells[0][1]
            buyAccountID = self.urgentBuys[0][1]
            quantityFilled = min(self.urgentBuys[0][0], self.urgentSells[0][0])  
            filledOrders += quantityFilled
            
            if self.getBestBid() and self.getBestAsk():
                midPrice = (self.getBestBid().getPrice() + self.getBestAsk().getPrice()) / 2
            else:
                midPrice = self.getLastPrice()
            
            accounts[buyAccountID].addPosition(self.asset, quantityFilled)
            accounts[buyAccountID].addPosition("CASH", -midPrice*quantityFilled)
            accounts[sellAccountID].addPosition(self.asset, -quantityFilled)
            accounts[sellAccountID].addPosition("CASH", midPrice*quantityFilled)

            # Emit trade updates
            emit_trade_update(buyAccountID, self.asset, "buy", quantityFilled, midPrice)
            emit_trade_update(sellAccountID, self.asset, "sell", quantityFilled, midPrice)

            # After each trade execution, update portfolio status
            if buyAccountID != "MARKET MAKER":
                portfolio_data = calculate_portfolio_status(accounts[buyAccountID])
                emit_portfolio_update(buyAccountID, portfolio_data)
            
            if sellAccountID != "MARKET MAKER":
                portfolio_data = calculate_portfolio_status(accounts[sellAccountID])
                emit_portfolio_update(sellAccountID, portfolio_data)

            self.urgentBuys[0] = (self.urgentBuys[0][0] - quantityFilled, buyAccountID)
            self.urgentSells[0] = (self.urgentSells[0][0] - quantityFilled, sellAccountID)

            # Remove empty orders
            if self.urgentBuys[0][0] == 0:
                self.urgentBuys.pop(0)
            if self.urgentSells[0][0] == 0:
                self.urgentSells.pop(0)
        
        # Fill remaining market orders against the order book
        while self.urgentBuys:
            if self.urgentBuys[0][1] == "MARKET MAKER":
                return

            filledOrders += self.urgentBuys[0][0]
            if self.urgentBuys[0][0] == 0:
                self.urgentBuys.pop(0)
            else:    
                buyAccountID = self.urgentBuys[0][1]
                if self.getBestAsk():
                    if self.getBestAsk().getQuantity() == 0:
                        self.asks.pop(self.getBestAsk().getPrice())
                    else:
                        quantityFilled = min(self.urgentBuys[0][0], self.getBestAsk().getQuantity())
                        self.getBestAsk().fulfillQuantity(quantityFilled)
                        self.urgentBuys[0] = (self.urgentBuys[0][0] - quantityFilled, buyAccountID)
                        accounts[buyAccountID].addPosition(self.asset, quantityFilled)
                        accounts[buyAccountID].addPosition("CASH", -self.getBestAsk().getPrice()*quantityFilled)
                        last_prices[self.asset] = self.getBestAsk().getPrice()
                        
                        # Emit trade update
                        emit_trade_update(buyAccountID, self.asset, "buy", quantityFilled, self.getBestAsk().getPrice())
                else:
                    break

        while self.urgentSells:
            if self.urgentSells[0][1] == "MARKET MAKER":
                return

            filledOrders += self.urgentSells[0][0]
            if self.urgentSells[0][0] == 0:
                self.urgentSells.pop(0)
            else:
                sellAccountID = self.urgentSells[0][1]
                if self.getBestBid():
                    if self.getBestBid().getQuantity() == 0:
                        self.bids.pop(self.getBestBid().getPrice())
                    else:
                        quantityFilled = min(self.urgentSells[0][0], self.getBestBid().getQuantity())
                        self.getBestBid().fulfillQuantity(quantityFilled)
                        self.urgentSells[0] = (self.urgentSells[0][0] - quantityFilled, sellAccountID)
                        accounts[sellAccountID].addPosition(self.asset, -quantityFilled)
                        accounts[sellAccountID].addPosition("CASH", self.getBestBid().getPrice()*quantityFilled)
                        last_prices[self.asset] = self.getBestBid().getPrice()
                        
                        # Emit trade update
                        emit_trade_update(sellAccountID, self.asset, "sell", quantityFilled, self.getBestBid().getPrice())
                else:
                    break
        
        return filledOrders

    def getLastPrice(self):
        return last_prices.get(self.asset, None) 