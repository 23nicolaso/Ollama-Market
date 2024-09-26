# Market Structure and Order Management System

## Overview

This document provides an overview of the market structure, order types, and settlement mechanisms implemented in the OLLAMA Market Simulator. The system is designed to simulate a financial market with multiple assets, order books, and various types of market participants.

## Key Components

### 1. Account Class

The `Account` class represents a participant's account in the market. It has the following key features:

- Stores cash and asset positions
- Allows adding and retrieving positions
- Calculates the total account value based on current asset prices

### 2. OrderBook Class

The `OrderBook` class is the core of the market structure. It manages buy and sell orders for a specific asset. Key features include:

- Maintains separate lists for bids (buy orders) and asks (sell orders)
- Handles limit and market orders
- Matches orders and executes trades
- Manages the order book state and last traded price

#### OrderLevel Class

The `OrderLevel` class is a nested class within `OrderBook`. It represents a price level in the order book and has the following characteristics:

- Stores orders at a specific price
- Manages the total quantity of orders at that price level
- Handles order fulfillment and cancellation

### 3. Order Types

The orderbook supports two main types of orders:

1. **Limit Orders**: Orders placed at a specific price, which are added to the order book if not immediately matched.
2. **Market Orders**: Orders executed immediately at the best available price.

In addition, some market agents have specialized order types, which are not shown in the orderbook:
1. **Conditional Orders**: Orders that are executed only if a price condition is met.
2. **Execute In Legs**: Large orders that are broken down into many smaller orders to mask whale activity.

## Market Structure

The market is structured as follows:

- Multiple assets are traded (e.g., "Simula 500", "Rivala ETF", "Allia ETF", "Factoria ETF", "Gold")
- Each asset has its own order book
- Market makers provide liquidity by continuously placing buy and sell orders
- Various types of traders (retail, hedge funds, high-frequency traders) interact with the market

## Order Execution and Settlement

1. **Order Placement**: 
   - Limit orders are added to the appropriate price level in the order book
   - Market orders are added to the urgent orders list, which is checked continuously by the market agents.

2. **Order Matching**:
   - The system continuously checks for matching limit orders and matching market orders.
   - When a match is found, the order level is updated and the trade is fulfilled.
   - Urgent orders are matched against other market orders, and then limit orders.

3. **Trade Execution**:
   - The `tradeAtPrice` method in the `Account` class is called to update positions
   - Cash and asset positions are adjusted accordingly

4. **Settlement**:
   - Settlement occurs instantly upon trade execution

5. **Position Management**:
   - The `Account` class keeps track of all positions
   - Positions are updated in real-time as trades are executed

## Special Features

1. **Order Cancellation**: The system allows for cancellation of old orders and clearing of far-away orders to maintain efficiency.

2. **Price History**: The system maintains a history of prices for each asset, which can be used for analysis and visualization.

## Conclusion

This market structure provides a simplified but functional simulation of a financial market. It incorporates key elements such as order books, different order types, and real-time settlement, allowing for the simulation of various market scenarios and trading strategies.

