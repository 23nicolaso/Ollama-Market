import tkinter as tk
import time
import random
import threading
import sys
from pathlib import Path
from market_simulator.price_server import user_account

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from market_simulator.config import get_state_string

from market_simulator.utils.tts_this import tts_this
from market_simulator.agents.market_maker import MarketMaker
from market_simulator.agents.retail_trader import RetailTrader
from market_simulator.agents.hedge_fund import HedgeFund
from market_simulator.agents.ta_trader import TATrader
from market_simulator.agents.hft_fund import HFTFund
from market_simulator.agents.spy_arb_fund import SpyArbFund
from market_simulator.agents.long_term_investor import LongTermInvestor
from market_simulator.agents.risk_on_risk_off import RiskOnRiskOffFirm
from market_simulator.gui.main_window import MainWindow
from market_simulator.utils.market_utils import makeMarkets, markets, markov_model, update_price_history, spreads_by_market, accounts, news_queue
from market_simulator.utils.news_generator import generate_news_thread, generate_chat_thread, init_agents
from market_simulator.price_server import start_server as start_api_server
from market_simulator.web.server import run as run_web_server
from market_simulator.utils.db_utils import init_db, shutdown_db

def run_simulation(root, main_window):
    """Run the main simulation loop"""
    markov_model.simulate_day()
    simulation_age = 0
    tick = 0
    release_time = None

    # Get reference to news feed frame
    news_feed_frame = main_window.news_feed_frame
    
    # Start update loops
    news_feed_frame._update_news_feed()
    news_feed_frame._update_chat_window()

    account_map = {
        0: "retail_trader",
        1: "hft_fund",
        2: "spy_arb_fund",
        3: "mean_reversion_fund",
        4: "ta_traders",
        5: "market_maker",
        6: "long_term_investor",
        7: "USER TRADER",
        8: "quant_firm",
        9: "risk_on_off_firm"
    }

    markov_model.simulate_day()
    release_time = int(time.time()) + 60*2

    # Quant firms begin placing bets on where financial data will take markets
    quant_firm.place_bets(markov_model.get_noisy_state()['sector_performance'])
    risk_on_off_firm.risk_off()
    
    # Gov Announces that Financial Data Will be Released in 5 mins
    tts_this("A massive Economic Data release is coming in 5 minutes.", 0.5, 10)

    while True: 
        time.sleep(0.01)        
        simulation_age += 1

        # for agent in accounts:
        #     print(account_map[agent], ": ", accounts[agent].getPositions())

        for market in markets:
            market_maker.provideLiquidity(markets[market])
            market_maker.makeMarket(markets[market])
            retail_trader.trade(markets[market])
            retail_trader.shiftSentimentToMean()
            spy_arb_fund.arbitrage()
            ta_traders.manageTATrades(market)
            ta_traders.updatePositioning(market)
            # hft_fund.updateOrdersInLegs(markets[market])
            # hft_fund.tradeMicrostructure(market)
            # hft_fund.refreshNBBOOrder(markets[market])
            # long_term_investor.trade(markets[market])
            # long_term_investor.updatePositioning(market)

            # risk_on_off_firm.refreshNBBOOrder(markets[market])

            # mean_reversion_fund.calculate_target_positions()
            # mean_reversion_fund.strategic_iceberg_update(markets[market], simulation_age)
            # quant_firm.refreshNBBOOrder(markets[market])

            user_account.updatePositioning(market)

            update_price_history(market, markets[market].last_price)

        if release_time:
            if int(time.time()) > release_time:
                real_state = markov_model.get_economy_state()
                release_time = None
                
                txt = f"Economic data released! Inflation comes in at {real_state['inflation']}, interest rate comes in at {real_state['interest_rate']}, unemployment at {real_state['unemployment']}, and economic growth is in a {real_state['economic_growth']}"
                news_queue.put(txt)
                tts_this(txt, 1 if real_state["sector_performance"]["TECHNOLOGY"] > 0 else 0, 10)

                mean_reversion_fund.set_market_return_profile(real_state['sector_performance'])
                quant_firm.close_bets()

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
        global retail_trader, hft_fund, mean_reversion_fund, ta_traders, market_maker, long_term_investor, spy_arb_fund, quant_firm, risk_on_off_firm
        retail_trader = RetailTrader(0, 1000000)
        hft_fund = HFTFund(1, 10000000)
        spy_arb_fund = SpyArbFund(2, 10000000)
        mean_reversion_fund = HedgeFund(3, 10000000, "mean_reversion")
        ta_traders = TATrader(4, 1000000)
        market_maker = MarketMaker(5, 100000000000000, spreads=spreads_by_market)
        long_term_investor = LongTermInvestor(6, 10000000)
        quant_firm = HedgeFund(8, 10000000, "quant_firm")
        risk_on_off_firm = RiskOnRiskOffFirm(9, 10000000)

        # Initialize news generator with agents
        init_agents(retail_trader, hft_fund, market_maker, long_term_investor, mean_reversion_fund, risk_on_off_firm)

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