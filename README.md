# Ollama-Powered Fake Stock Market Simulation

This project simulates a fictional stock market in a world with four imaginary countries. It uses Python and the Ollama language model to create a dynamic market environment with multiple trading agents, real-time price updates, and a graphical user interface for monitoring market activity.

## Features

- Simulated stock market with assets tied to fictional countries:
  - Simula: A technology-focused nation
  - Rivala: An agricultural powerhouse
  - Allia: An entertainment industry leader
  - Factoria: An industrial giant
- Various realistic, simulated trading agents (Market Maker, Retail Trader, Hedge Fund, HFT Fund, Technical Analysis Traders)
- Real-time price updates, charts and orderbooks
- Accurate market microstructure, with limit orders, market orders, conditional orders
- News generation using Ollama LLM, reflecting events in the fictional world
- Sentiment analysis and market impact simulation
- GUI for monitoring market activity and news

## World Background

The simulation is set in a world with four main countries:
- Simula: Known for its advanced technology sector
- Rivala: A major agricultural producer
- Allia: Famous for its entertainment industry
- Factoria: A leading industrial nation and weapons manufacturer

These countries have complex economic and political relationships, including occasional conflicts between Simula and Rivala, with Allia allied to Rivala and Factoria supplying weapons to all sides.

## Requirements

- Python 3.7+
- Ollama
- tkinter
- matplotlib
- langchain_ollama

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ollama-fake-stock-market.git
   cd ollama-fake-stock-market
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Ensure Ollama is installed and running on your system.

## Usage

Run the simulation:
python SingleFileImplementation.py


The GUI will open, displaying:
- A dropdown to select different assets
- A chart showing price history
- A news feed with generated headlines
- Real-time price updates for all assets

## Components

- `OrderBook`: Manages buy and sell orders for each asset
- `Account`: Represents a trading account with positions and cash
- `MarketAgent`: Base class for all trading agents
- `MarketMaker`: Provides liquidity to the market
- `RetailTrader`: Simulates retail investor behavior
- `HedgeFund` and `ExecutionalTrader`: More sophisticated trading strategies
- `genNews()`: Generates news headlines using Ollama and updates market sentiment

## Customization

You can modify the simulation by:
- Adjusting initial prices and cash for different agents
- Adding new trading strategies
- Tweaking news generation and sentiment analysis parameters

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Disclaimer

This is a simulated market for educational and entertainment purposes only. It does not reflect real-world financial markets and should not be used for actual trading decisions.
