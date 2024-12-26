# Market Simulator

This project simulates a fictional stock market. It uses Python and Meta's LLama 3.1-8B language model to create a dynamic market environment with multiple trading agents, real-time price updates, and a graphical user interface for monitoring market activity.
This was mainly done to experiment with prompt engineering and as random programming practice. 

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
- Real-time news notifications in web interface
- Interactive news submission through GUI

### Real-Time Visualization
- Tkinter-based GUI for market monitoring
- Real-time price and sentiment displays
- News and chat feed windows
- Web-based charting interface using TradingView's Lightweight Charts
  - 10-second OHLC candles
  - Real-time updates
  - Historical data preservation
  - Multiple asset views
  - Real-time news notifications with sleek overlay design
  - Interactive order book visualization
  - Trade log with agent activity tracking

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
3. News and chat generation with real-time web notifications
4. Price history API server
5. Web interface server with Socket.IO for real-time updates

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
│   └── news_feed.py       # News and chat display with web emission
├── web/
│   ├── index.html         # Web-based charting interface with news notifications
│   └── server.py          # Flask-SocketIO server for real-time updates
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

### Version 0.1.2 (Latest)
- Added real-time news notifications to web interface
- Integrated Flask-SocketIO for seamless updates
- Enhanced web UI with interactive order book
- Added trade log with agent activity tracking
- Improved news feed with web emission
- Added interactive news submission through GUI
- Tweaked agent configurations
- Added market depth / order book visualization

### Version 0.1.1
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