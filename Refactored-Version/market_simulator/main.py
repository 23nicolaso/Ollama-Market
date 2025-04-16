import tkinter as tk
import random
import threading
import sys
from pathlib import Path
from market_simulator.price_server import user_account

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from market_simulator.agents.market_maker import MarketMaker
from market_simulator.agents.retail_trader import RetailTrader
from market_simulator.agents.hedge_fund import HedgeFund
from market_simulator.agents.ta_trader import TATrader
from market_simulator.agents.hft_fund import HFTFund
from market_simulator.agents.spy_arb_fund import SpyArbFund
from market_simulator.agents.long_term_investor import LongTermInvestor
from market_simulator.gui.main_window import MainWindow
from market_simulator.utils.market_utils import makeMarkets, markets, price_history, update_price_history, spreads_by_market, randomly_alter_risk_params, setLastNewsTick, wasNewsRecent
from market_simulator.utils.news_generator import generate_news_thread, generate_chat_thread, init_agents
from market_simulator.price_server import start_server as start_api_server
from market_simulator.web.server import run as run_web_server
from market_simulator.utils.db_utils import init_db, shutdown_db

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
        simulation_age += 1
    
        for market in markets:
            market_maker.provideLiquidity(markets[market])
            market_maker.makeMarket(markets[market])
            retail_trader.trade(markets[market])
            retail_trader.shiftSentimentToMean()
            spy_arb_fund.arbitrage()
            hft_fund.updateOrdersInLegs(markets[market])
            # hft_fund.tradeMicrostructure(market) # NOTE REMOVED BECAUSE IT WAS ABNORMAL

            mean_reversion_fund.calculate_target_positions()
            mean_reversion_fund.refreshNBBOOrder(markets[market])
            
            long_term_investor.updatePositioning(market)

            user_account.updatePositioning(market)

            update_price_history(market, markets[market].lastPrice)

            if price_history[market].isFull(): #  only do some updates once initial price is set correctly
                if simulation_age & 0b111 == 0:
                    result = randomly_alter_risk_params(market)
            
                    if result is not None:
                        generate_news_thread(explain_this=(result, market))
                        setLastNewsTick(simulation_age)

        # Update GUI components
        main_window.update_prices()
        main_window.update_sentiments(retail_trader)
        main_window.update()
    
        try:
            root.update()
        except tk.TclError:
            # Window was closed
            break
        tick += 1

def main():
    # Initialize database
    init_db()
    
    try:
        # Create root window
        root = tk.Tk()
        main_window = MainWindow(root)

        # Initialize markets
        makeMarkets()
        
        # Initialize agents
        global retail_trader, hft_fund, mean_reversion_fund, ta_traders, market_maker, long_term_investor, spy_arb_fund
        retail_trader = RetailTrader("Retail", 1000000)
        hft_fund = HFTFund("HFT Fund", 10000000)
        spy_arb_fund = SpyArbFund("Spy ARB Fund", 10000000)
        mean_reversion_fund = HedgeFund("Mean Reversion Fund", 10000000, "mean_reversion")
        ta_traders = TATrader("TA Trading Firm", 1000000)
        market_maker = MarketMaker("Market Maker", 100000000000000, spreads=spreads_by_market)
        long_term_investor = LongTermInvestor("Long Term Investor", 10000000)

        # Initialize news generator with agents
        init_agents(retail_trader, hft_fund, market_maker, long_term_investor)

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
        root.mainloop()
    finally:
        # Ensure database is properly shutdown
        shutdown_db()

if __name__ == "__main__":
    main() 