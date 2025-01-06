from market_simulator.utils.market_utils import last_prices
from market_simulator.price_server import socketio

def calculate_portfolio_status(account):
    """Calculate portfolio status including positions and P&L"""
    portfolio_data = {
        "cash": round(account.getCash(), 2),
        "positions": {},
        "pnl": 0
    }
    
    # Calculate positions and P&L
    for asset, quantity in account.positions.items():
        if asset != "CASH":
            portfolio_data["positions"][asset] = quantity
            # Calculate P&L based on current market value
            portfolio_data["pnl"] += (last_prices[asset] * quantity)
    
    # Subtract initial cash to get P&L
    portfolio_data["pnl"] = round(portfolio_data["pnl"] + portfolio_data["cash"] - account.initial_cash, 2)
    
    return portfolio_data

def emit_portfolio_update(account_id, portfolio_data):
    """Emit portfolio update to the frontend"""
    socketio.emit('portfolio_update', portfolio_data) 