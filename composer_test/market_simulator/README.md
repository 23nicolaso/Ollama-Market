# Market Simulator

A simulated market environment featuring multiple trading agents, real-time price charts, and AI-generated news events.

## Overview

This project simulates a market environment with multiple assets and trading agents. The simulation includes:

- Multiple trading agents with different strategies:
  - Market Maker: Provides liquidity and maintains orderly markets
  - Retail Trader: Simulates retail trading behavior based on sentiment
  - Hedge Funds: Implements mean reversion and macro strategies
  - Technical Analysis Trader: Trades based on technical indicators
  - HFT Fund: High-frequency trading based on news events

- Real-time visualization:
  - Price charts for each asset
  - News feed with AI-generated headlines
  - Chat window simulating trader discussions
  - Price and sentiment tables

- Simulated world with four main countries:
  - Simula: Technology-focused country
  - Rivala: Agricultural powerhouse
  - Allia: Entertainment hub
  - Factoria: Industrial nation

## Requirements

- Python 3.8+
- Ollama (for AI-generated content)
- Required Python packages:
  - tkinter
  - matplotlib
  - langchain_ollama
  - numpy

## Installation

1. Clone the repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Ollama is running on your system

## Usage

Run the simulation:
```bash
python -m market_simulator.main
```

## Project Structure

```
market_simulator/
├── __init__.py
├── main.py
├── config.py
├── models/
│   ├── __init__.py
│   ├── account.py
│   ├── order_book.py
│   └── order_level.py
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── market_maker.py
│   ├── retail_trader.py
│   ├── hedge_fund.py
│   ├── ta_trader.py
│   └── hft_fund.py
├── gui/
│   ├── __init__.py
│   ├── main_window.py
│   ├── charts.py
│   └── news_feed.py
└── utils/
    ├── __init__.py
    ├── market_utils.py
    └── news_generator.py
```

## Features

- Real-time price updates and order book management
- AI-generated news events affecting market sentiment
- Multiple trading strategies interacting in the market
- Visual representation of market data
- Simulated chat messages from traders

## Contributing

Feel free to submit issues and enhancement requests! 