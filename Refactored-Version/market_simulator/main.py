import tkinter as tk
import time
import random
import threading
import sys
from pathlib import Path
from market_simulator.price_server import user_account

sys.path.append(str(Path(__file__).parent.parent))

from market_simulator.config import get_state_string
from market_simulator.models.marketState import InflationState, InterestRateState

from market_simulator.utils.tts_this import tts_this
from market_simulator.agents.market_maker import MarketMaker
from market_simulator.agents.retail_trader import RetailTrader
from market_simulator.agents.hedge_fund import HedgeFund
from market_simulator.agents.ta_trader import TATrader
from market_simulator.agents.hft_fund import HFTFund
from market_simulator.agents.spy_arb_fund import SpyArbFund
from market_simulator.agents.long_term_investor import LongTermInvestor
from market_simulator.agents.risk_on_risk_off import RiskOnRiskOffFirm
from market_simulator.agents.llm_funds import LLMFund
from market_simulator.gui.main_window import MainWindow
from market_simulator.utils.market_utils import (
    makeMarkets, markets, markov_model, update_price_history,
    spreads_by_market, accounts, news_queue, agent_registry, action_queue,
)
from market_simulator.utils.news_generator import generate_news_thread, generate_chat_thread, init_agents
from market_simulator.models.options_market import options_market
from market_simulator.price_server import start_server as start_api_server
from market_simulator.web.server import run as run_web_server
from market_simulator.utils.db_utils import init_db, shutdown_db

# Tkinter GUI update rate: redraw chart + price table every N simulation ticks.
# At time.sleep(0.01) per tick that's N * 10 ms between redraws.
_GUI_TICK_INTERVAL = 100   # ~1 s


def _broadcast_loop():
    """
    Daemon thread: forwards action-queue entries and periodic leaderboard /
    economy state snapshots to all connected web clients via Socket.IO.
    Also auto-generates chat messages every 30 seconds.
    """
    from market_simulator.web.server import emit_action, emit_leaderboard, emit_economy, emit_chat
    from market_simulator.utils.news_generator import generate_chat_thread
    from market_simulator.utils.market_utils import chat_queue

    last_slow_emit = 0.0
    last_chat_gen  = 0.0

    while True:
        # ── Drain action queue → web clients ──────────────────────────────
        while not action_queue.empty():
            try:
                msg = action_queue.get_nowait()
                emit_action(msg)
            except Exception:
                pass

        # ── Drain chat queue → web clients ────────────────────────────────
        while not chat_queue.empty():
            try:
                msg = chat_queue.get_nowait()
                emit_chat(msg)
            except Exception:
                pass

        now = time.time()

        # ── Auto-generate a chat message every 30 s ────────────────────────
        if now - last_chat_gen >= 30.0:
            try:
                generate_chat_thread()
            except Exception:
                pass
            last_chat_gen = now

        if now - last_slow_emit >= 2.0:
            # ── Leaderboard ────────────────────────────────────────────────
            try:
                entries = []
                for name, (agent, init_cash) in agent_registry.items():
                    try:
                        value = agent.account.getValue()
                        pnl   = value - init_cash
                        pct   = pnl / init_cash * 100 if init_cash else 0.0
                        entries.append({"name": name, "value": value,
                                        "pnl": pnl, "pct": pct})
                    except Exception:
                        pass
                entries.sort(key=lambda e: e["value"], reverse=True)
                for i, e in enumerate(entries):
                    e["rank"] = i + 1
                emit_leaderboard({"entries": entries})
            except Exception:
                pass

            # ── Economy state ──────────────────────────────────────────────
            try:
                state = markov_model.get_economy_state()
                # sector_performance is already {str: float} from get_economy_state()
                emit_economy({
                    "market_trend":    state["market_trend"].name,
                    "inflation":       state["inflation"].name,
                    "interest_rate":   state["interest_rate"].name,
                    "unemployment":    state["unemployment"].name,
                    "economic_growth": state["economic_growth"].name,
                    "sectors": {
                        s: round(p, 4)
                        for s, p in state["sector_performance"].items()
                    },
                })
            except Exception:
                pass

            last_slow_emit = now

        time.sleep(0.1)


def run_simulation(root, main_window):
    """Main simulation loop — runs in a daemon thread."""
    simulation_age = 0
    gui_tick = 0

    markov_model.simulate_day()
    release_time = int(time.time()) + 60 * 10

    quant_firm.place_bets(markov_model.get_noisy_state()['sector_performance'])
    risk_on_off_firm.risk_off()

    ollama_thread = threading.Thread(target=ollama_fund.analyzeOutcomes, daemon=True)
    gemma_thread  = threading.Thread(target=gemma_fund.analyzeOutcomes,  daemon=True)
    ollama_thread.start()
    gemma_thread.start()

    tts_this("A massive Economic Data release is coming in 10 minutes.", 0.5, 10)

    while True:
        time.sleep(0.01)
        simulation_age += 1

        for market in markets:
            market_maker.provideLiquidity(markets[market])
            market_maker.makeMarket(markets[market])
            retail_trader.trade(markets[market])
            retail_trader.shiftSentimentToMean()

            hft_fund.stealthAlgo(markets[market])

            long_term_investor.trade(markets[market])
            long_term_investor.sniperAlgo(markets[market])

            risk_on_off_firm.stealthAlgo(markets[market])
            ollama_fund.opportunisticAlgo(markets[market])
            gemma_fund.stealthAlgo(markets[market])
            quant_firm.opportunisticAlgo(markets[market])

            mean_reversion_fund.calculate_target_positions()
            mean_reversion_fund.opportunisticAlgo(markets[market])
            market_maker.provideLiquidity(markets[market])
            market_maker.makeMarket(markets[market])

            update_price_history(market, markets[market].last_price)

        market_maker.hedge_options_delta(options_market)
        market_maker.stealthAlgo(markets["SPY"])

        if release_time and int(time.time()) > release_time:
            real_state  = markov_model.get_economy_state()
            release_time = None

            if real_state['inflation'] == InflationState.MODERATE:
                inflation_number = "up 2.9% YoY"
            elif real_state['inflation'] == InflationState.HIGH:
                inflation_number = "up 3.3% YoY"
            else:
                inflation_number = "up 2.5% YoY"

            if real_state['interest_rate'] == InterestRateState.MODERATE:
                interest_state = "hold rates steady"
            elif real_state['interest_rate'] == InterestRateState.HIGH:
                interest_state = "increase rates by 25-50 BPS"
            else:
                interest_state = "decrease rates by 25-50 BPS"

            txt = (f"Economic data released! CPI comes in at {inflation_number} "
                   f"and the federal reserve decided to {interest_state}")
            news_queue.put(txt)
            tts_this(txt, 1 if real_state["sector_performance"]["TECHNOLOGY"] > 0 else 0, 10)

            mean_reversion_fund.set_market_return_profile(real_state['sector_performance'])
            quant_firm.close_bets()
            ollama_fund.tradeResult(txt)
            gemma_fund.tradeResult(txt)

        # ── Tkinter GUI — throttled to ~1 Hz ──────────────────────────────
        gui_tick += 1
        if gui_tick >= _GUI_TICK_INTERVAL:
            main_window.update_prices()
            main_window.update()
            gui_tick = 0

        try:
            root.update()
        except tk.TclError:
            break


def main():
    init_db()

    try:
        root = tk.Tk()
        main_window = MainWindow(root)

        makeMarkets()

        global retail_trader, hft_fund, mean_reversion_fund, ta_traders
        global market_maker, long_term_investor, spy_arb_fund, quant_firm
        global risk_on_off_firm, ollama_fund, gemma_fund

        retail_trader      = RetailTrader(0,  1_000_000)
        hft_fund           = HFTFund(1,       10_000_000)
        spy_arb_fund       = SpyArbFund(2,    10_000_000)
        mean_reversion_fund= HedgeFund(3,     10_000_000, "mean_reversion")
        ta_traders         = TATrader(4,       1_000_000)
        market_maker       = MarketMaker(5, 100_000_000_000_000, spreads=spreads_by_market)
        long_term_investor = LongTermInvestor(6, 10_000_000)
        quant_firm         = HedgeFund(8,     10_000_000, "quant_firm")
        risk_on_off_firm   = RiskOnRiskOffFirm(9, 10_000_000)
        ollama_fund        = LLMFund(10,     100_000_000, "Ollama Fund")
        gemma_fund         = LLMFund(11,     100_000_000, "Gemma Fund")

        # Snapshot actual starting portfolio value (cash + any initial positions).
        # This is the correct PnL baseline — not just initial cash.
        agent_registry.update({
            "Retail Trader":  (retail_trader,       retail_trader.account.getValue()),
            "HFT Fund":       (hft_fund,            hft_fund.account.getValue()),
            "SPY Arb Fund":   (spy_arb_fund,        spy_arb_fund.account.getValue()),
            "Mean Reversion": (mean_reversion_fund, mean_reversion_fund.account.getValue()),
            "TA Trader":      (ta_traders,          ta_traders.account.getValue()),
            "Long-Term Inv.": (long_term_investor,  long_term_investor.account.getValue()),
            "Quant Firm":     (quant_firm,          quant_firm.account.getValue()),
            "Risk On/Off":    (risk_on_off_firm,    risk_on_off_firm.account.getValue()),
            "Ollama Fund":    (ollama_fund,         ollama_fund.account.getValue()),
            "Gemma Fund":     (gemma_fund,          gemma_fund.account.getValue()),
        })

        init_agents(retail_trader, hft_fund, market_maker, long_term_investor,
                    mean_reversion_fund, risk_on_off_firm, ollama_fund, gemma_fund)

        # ── Background threads ─────────────────────────────────────────────
        threading.Thread(target=start_api_server, daemon=True).start()
        threading.Thread(target=lambda: run_web_server(8000), daemon=True).start()
        threading.Thread(target=_broadcast_loop, daemon=True).start()
        threading.Thread(target=run_simulation, args=(root, main_window), daemon=True).start()

        root.mainloop()
    finally:
        shutdown_db()


if __name__ == "__main__":
    main()
