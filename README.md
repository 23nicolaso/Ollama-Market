# Market Simulator

This project simulates a fictional stock market. It uses Python, HTML and Meta's LLama 3.1-8B language model to create a dynamic market environment with multiple trading agents, real-time price updates, and a graphical user interface for monitoring market activity.
This was mainly done to experiment with prompt engineering and as fun programming practice. 

## Features

### Market Structure
- Fully configurable universe of tradable assets with different initial prices and spreads
- Order book with functional limit, market orders
- Iceberg-style orders for trading large volumes
- Real-time price discovery based on supply and demand
- Arbitrage fund that arbitrages the SPY basket against the SPY index 

### Trading Agents
- Retail Trader: Simulates individual investors with sentiment-based trading
- Market Maker: Provides liquidity and maintains orderly markets
- HFT Fund: High-frequency trading strategy which trades news as soon as it comes out
- Hedge Funds: Mean reversion strategies
- Technical Analysis Traders: Trades based on technical indicators
- SPY Arbitrage Fund: Arbitrages the SPY basket against the SPY index

### News and Events System
- AI-powered news generation using Ollama
- Sentiment analysis affecting trader behavior
- Real-time chat messages between agents
- Economic health tracking for different sectors
- Real-time news notifications in web interface
- Interactive news submission through GUI

### Real-Time Visualization
- Tkinter-based GUI for market monitoring and running server commands
![TKINTERGUI](https://github.com/23nicolaso/Ollama-Market/blob/main/Refactored-Version/images/Screenshot%20(219).png)

- Real-time price charts using TradingView's Lightweight Charts
  - Interactive OHLC candlestick charts
  - 10-second candle intervals
  - Multiple asset views with dropdown selection
  - Real-time price updates via WebSocket
- Advanced order book visualization
  - Depth chart showing cumulative volume
  - Real-time bid/ask updates
  - Interactive tooltips showing price levels
- Trading Interface
  - Place market/limit/iceberg orders
  - Real-time portfolio value updates
  - Position tracking across multiple assets
  - Cash balance monitoring
- News & Events
  - Real-time news notifications with sleek overlay design
  - Interactive news submission through GUI
  - Agent chat messages in real-time
  
![TRADINGCHART](https://github.com/23nicolaso/Ollama-Market/blob/main/Refactored-Version/images/Screenshot%20(250).png)
![DEPTHOFMARKET](https://github.com/23nicolaso/Ollama-Market/blob/main/Refactored-Version/images/Screenshot%20(249).png)


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

## System Architecture

The system uses a multi-threaded architecture with real-time communication:

1. Core Market Engine
   - Main simulation loop handling market mechanics
   - Order book management and trade matching
   - Agent behavior and decision making

2. GUI Layer
   - Tkinter-based main control interface
   - Real-time price charts and order book
   - News feed and chat windows
   - Interactive trading controls

3. Web Interface
   - Flask + Socket.IO server for real-time updates
   - TradingView charts for price visualization
   - WebSocket events for:
     - Price updates
     - Order book changes
     - News notifications
     - Trade confirmations
     - Portfolio updates

4. Communication Flow
   - Core engine → GUI updates via queues
   - Core engine → Web clients via Socket.IO
   - Web clients → Core engine via Socket.IO events
   - News generation → All interfaces via event emission

```
market_simulator/
├── __init__.py
├── main.py                 # Main simulation loop
├── config.py              # Centralized configuration
├── requirements.txt      # Project dependencies
├── models/
│   ├── __init__.py
│   ├── account.py         # Account management
│   ├── order_book.py      # Order book implementation
│   └── order_level.py     # Order level management
├── agents/
│   ├── __init__.py
│   ├── base_agent.py      # Base MarketAgent class
│   ├── executional_trader.py  # Base class for complex strategies
│   ├── market_maker.py    # Market making agent
│   ├── retail_trader.py   # Retail trading agent
│   ├── hedge_fund.py      # Mean reversion strategies
│   ├── ta_trader.py       # Technical analysis trader
│   └── hft_fund.py        # High-frequency trading agent
├── gui/
│   ├── __init__.py
│   ├── main_window.py     # Main GUI window
│   ├── news_feed.py       # News and chat display
│   └── charts.py          # Price chart visualization
├── web/
│   ├── static/
│   │   ├── css/          # Web interface styling
│   │   ├── js/           # Client-side JavaScript
│   │   └── index.html    # Main web interface
│   └── server.py         # Flask-SocketIO server
└── utils/
    ├── __init__.py
    ├── market_utils.py    # Market-related utilities
    └── news_generator.py  # Ollama-based news generation
```

## Configuration

The system is highly configurable through multiple files:

- `config.py`: Core simulation parameters
  - Available assets and initial prices
  - Spread configurations
  - Agent parameters
  - World context for news generation

- `web/server.py`: Web interface settings
  - Server ports and hosts
  - WebSocket configurations
  - CORS settings

- `gui/news_feed.py`: News and chat settings
  - Update frequencies
  - Display formats
  - Message queuing

## Dependencies

- Python 3.8+
- Numpy >= 1.24.0
- Flask >= 2.0.0
- Flask-SocketIO >= 5.0.0
- Flask-CORS >= 3.0.0
- Eventlet >= 0.30.0
- Matplotlib >= 3.5.0
- Langchain-Ollama >= 0.1.0
- Tkinter >= 8.6.0

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
