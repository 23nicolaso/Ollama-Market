import tkinter as tk
import random
import threading
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from market_simulator.agents.market_maker import MarketMaker
from market_simulator.agents.retail_trader import RetailTrader
from market_simulator.agents.hedge_fund import HedgeFund
from market_simulator.agents.ta_trader import TATrader
from market_simulator.agents.hft_fund import HFTFund
from market_simulator.gui.main_window import MainWindow
from market_simulator.utils.market_utils import makeMarkets, markets, price_history, simulation_age, update_price_history, spreads_by_market
from market_simulator.utils.news_generator import generate_news_thread, generate_chat_thread, init_agents

def main():
    # Create root window
    root = tk.Tk()
    main_window = MainWindow(root)
    simulation_age = 0

    # Get reference to news feed frame
    news_feed_frame = main_window.news_feed_frame
    
    # Start update loops
    news_feed_frame._update_news_feed()
    news_feed_frame._update_chat_window()

    # Initialize agents
    retail_trader = RetailTrader("RETAIL TRADER", 1000000)
    hft_fund = HFTFund("EVENTS TRADING FUND", 10000000)
    mean_reversion_fund = HedgeFund("Mean Reversion Fund", 10000000, "mean_reversion")
    macro_fund = HedgeFund("Macro Fund", 10000000, "macro")
    ta_traders = TATrader("TA TRADING FIRM", 1000000)
    market_maker = MarketMaker("MARKET MAKER", 100000000000000, spreads=spreads_by_market)

    # Initialize markets
    makeMarkets()

    # Initialize news generator with agents
    init_agents(retail_trader, hft_fund, market_maker)

    # Generate initial news and chat
    generate_news_thread()
    generate_chat_thread()

    # Main simulation loop
    tick = 0
    while True:
        for market in markets:
            retail_trader.trade(markets[market])

        max_history_length = 1000
        for asset in price_history:
            if len(price_history[asset]) > max_history_length:
                price_history[asset] = price_history[asset][-max_history_length:]

        retail_trader.shiftSentimentToMean()

        for market in markets:
            market_maker.provideLiquidity(markets[market])

        if tick > 10:        
            simulation_age += 1
            if random.random() < 0.01:
                generate_chat_thread()
            if random.random() < 0.001:
                generate_news_thread()
        
            for market in markets:
                market_maker.makeMarket(markets[market])
                hft_fund.updatePositioning(market)

                mean_reversion_fund.calculate_target_positions()
                mean_reversion_fund.update_positions(market)
                
                macro_fund.calculate_target_positions()
                macro_fund.update_positions(market)
                
                update_price_history(market, markets[market].getLastPrice())
                ta_traders.manageTATrades(market)
                ta_traders.checkConditionalOrders(market)
                markets[market].clearEmptyOrderlevels()
                markets[market].cancelAllOldOrders()
                markets[market].clearFarOrders()

                if len(price_history[market]) < 100:
                    continue

                                # Update GUI components
            main_window.update_prices()
            main_window.update_sentiments(retail_trader)
            main_window.update()
            tick = 0
        
        root.update()
        tick += 1

if __name__ == "__main__":
    main() 