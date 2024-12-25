# Market Simulator

A Python-based market simulation that models various types of traders and their interactions in a multi-asset market environment.

## Features

### Market Structure
- Multiple tradable assets with configurable initial prices and spreads
- Order book with limit and market orders
- Real-time price discovery based on supply and demand

### Trading Agents
- Retail Trader: Simulates individual investors with sentiment-based trading
- Market Maker: Provides liquidity and maintains orderly markets
- HFT Fund: High-frequency trading strategies
- Hedge Funds: Mean reversion and macro strategies
- Technical Analysis Traders: Trades based on technical indicators

### News and Events System
- AI-powered news generation using Ollama
- Sentiment analysis affecting trader behavior
- Real-time chat messages between agents
- Economic health tracking for different sectors

### Real-Time Visualization
- Tkinter-based GUI for market monitoring
- Real-time price and sentiment displays
- News and chat feed windows
- Web-based charting interface using TradingView's Lightweight Charts
  - 10-second OHLC candles
  - Real-time updates
  - Historical data preservation
  - Multiple asset views

## Requirements

- Python 3.8+
- Ollama (for AI-generated content)
- Required Python packages:
  ```
  matplotlib>=3.5.0
  numpy>=1.21.0
  langchain_ollama>=0.1.0
  tk>=8.6.0
  ```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Ensure Ollama is installed and running for news generation

## Running the Simulation

Start the simulation with:
```bash
python -m market_simulator.main
```

This will launch:
1. The main GUI application
2. Price history API server (port 5000)
3. Web charting interface (port 8000)

Access the web charts at: `http://localhost:8000`

## Configuration

Key settings can be adjusted in `config.py`:
- Available assets and their properties
- Initial prices and spreads
- World context for news generation
- Various simulation parameters

## Architecture

The simulation runs multiple components in parallel:
1. Main simulation loop (market mechanics)
2. GUI updates and user interface
3. News and chat generation
4. Price history API server
5. Web interface server

```
market_simulator/
├── __init__.py
├── main.py                 # Main simulation loop
├── config.py              # Centralized configuration
├── setup.py              # Package setup and dependencies
├── requirements.txt      # Project dependencies
├── models/
│   ├── __init__.py
│   ├── account.py         # Account management
│   ├── order_book.py      # Order book implementation with OrderLevel inner class
│   └── order_level.py     # Order level management
├── agents/
│   ├── __init__.py
│   ├── base_agent.py      # Base MarketAgent class
│   ├── executional_trader.py  # Base class for complex trading strategies
│   ├── market_maker.py    # Market making agent
│   ├── retail_trader.py   # Retail trading agent
│   ├── hedge_fund.py      # Mean reversion and macro strategies
│   ├── ta_trader.py       # Technical analysis trader
│   └── hft_fund.py        # High-frequency trading agent
├── gui/
│   ├── __init__.py
│   ├── main_window.py     # Main GUI window with Tkinter
│   ├── charts.py          # Matplotlib price chart visualization
│   └── news_feed.py       # News and chat display components
├── web/
│   ├── index.html         # Web-based charting interface
│   └── server.py          # Web server for charts
├── price_server.py        # Flask price history API server
└── utils/
    ├── __init__.py
    ├── market_utils.py    # Market-related utilities and globals
    └── news_generator.py  # Ollama-based news generation
```

## Dependencies

- Python 3.8+
- Numpy for efficient data handling
- Flask for API server
- Flask-SocketIO for real-time updates
- TradingView Lightweight Charts for web visualization
- Tkinter for main GUI
- Ollama for AI-powered news generation

## Contributing

Feel free to submit issues and pull requests for:
- New trading strategies
- UI improvements
- Performance optimizations
- Additional features

### Version 0.1.1 (Latest)
- Added centralized configuration in config.py
- Implemented sentiment analysis for news events
- Added fair value calculation and display
- Updated GUI with price and sentiment tables
- Added world context configuration
- Improved news generation with impact analysis
- Added executional_trader.py as base class for complex strategies

### Version 0.1.0 (Initial)
- Basic market simulation with multiple agents
- Simple price charts
- Basic news generation
- Initial GUI implementation 