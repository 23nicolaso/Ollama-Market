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
from market_simulator.agents.long_term_investor import LongTermInvestor
from market_simulator.gui.main_window import MainWindow
from market_simulator.utils.market_utils import makeMarkets, markets, price_history, simulation_age, update_price_history, spreads_by_market
from market_simulator.utils.news_generator import generate_news_thread, generate_chat_thread, init_agents
from market_simulator.price_server import start_server as start_api_server
from market_simulator.web.server import run as run_web_server

def run_simulation(root, main_window):
    """Run the main simulation loop"""
    simulation_age = 0
    tick = 0

    # Get reference to news feed frame
    news_feed_frame = main_window.news_feed_frame
    
    # Start update loops
    news_feed_frame._update_news_feed()
    news_feed_frame._update_chat_window()

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
        
            for market in markets:
                retail_trader.trade(markets[market])
                market_maker.makeMarket(markets[market])
                hft_fund.updatePositioning(market)

                mean_reversion_fund.calculate_target_positions()
                mean_reversion_fund.update_positions(market)
                
                long_term_investor.trade(markets[market], retail_trader)
                long_term_investor.updatePositioning(market)

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
        
        try:
            root.update()
        except tk.TclError:
            # Window was closed
            break
        tick += 1

def main():
    # Create root window
    root = tk.Tk()
    main_window = MainWindow(root)

    # Initialize agents
    global retail_trader, hft_fund, mean_reversion_fund, ta_traders, market_maker, long_term_investor
    retail_trader = RetailTrader("RETAIL TRADER", 1000000)
    hft_fund = HFTFund("EVENTS TRADING FUND", 10000000)
    mean_reversion_fund = HedgeFund("Mean Reversion Fund", 10000000, "mean_reversion")
    ta_traders = TATrader("TA TRADING FIRM", 1000000)
    market_maker = MarketMaker("MARKET MAKER", 100000000000000, spreads=spreads_by_market)
    long_term_investor = LongTermInvestor("LONG TERM INVESTOR", 10000000)

    # Initialize markets
    makeMarkets()

    # Initialize news generator with agents
    init_agents(retail_trader, hft_fund, market_maker)

    # Generate initial news and chat
    generate_news_thread()
    generate_chat_thread()

    # Start API server in a separate thread
    api_thread = threading.Thread(target=start_api_server)
    api_thread.daemon = True
    api_thread.start()

    # Start web server in a separate thread
    web_thread = threading.Thread(target=lambda: run_web_server(8000))
    web_thread.daemon = True
    web_thread.start()

    # Start simulation in a separate thread
    sim_thread = threading.Thread(target=run_simulation, args=(root, main_window))
    sim_thread.daemon = True
    sim_thread.start()

    # Start tkinter main loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Shutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main() 