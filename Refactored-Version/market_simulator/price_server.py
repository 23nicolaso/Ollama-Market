from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
import time
from market_simulator.utils.db_utils import db_manager as db
from market_simulator.utils.market_utils import price_history, assets, markets, get_price_history
from market_simulator.agents.user_trader import UserTrader
from market_simulator.models.options_market import options_market
from datetime import datetime

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
user_account = UserTrader(7, 1000000)

def emit_orderbook_update(asset):
    """Emit order book update through websocket"""
    if asset in markets:
        market = markets[asset]
    
        bids = market.get_bids()
        asks = market.get_asks()
        socketio.emit('orderbook_update', {
            'asset': asset,
            'bids': bids,
            'asks': asks,
            'timestamp': time.time()
        })


@app.route('/assets')
def get_assets():
    """Return list of available assets"""
    return jsonify(assets)

@app.route('/price_history/<asset>')
def get_price_history(asset):
    """Return price history for a specific asset"""
    try:
        # Get historical data from database
        historical_prices = db.get_price_history(asset)
        
        return jsonify({
            'asset': asset,
            'prices': historical_prices
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@socketio.on('place_order')
def handle_place_order(data):
    try:
        # Extract order details
        ticker = data['ticker']
        direction = data['direction']
        quantity = data['quantity']
        order_type = data['order_type']
        price = data.get('price', 0)  # Optional for market orders
        
        if ticker not in markets:
            return {'status': 'error', 'message': 'Invalid ticker'}
        
        # Create and execute the order
        order = type('Order', (), {
            'market': ticker,
            'direction': direction,
            'price': price,
            'quantity': quantity,
            'order_type': order_type
        })()
        user_account.execute_order(order)
        
        return {'status': 'success', 'message': f'{order_type.capitalize()} {direction} order placed for {quantity} {ticker}'}
        
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@socketio.on('place_option_order')
def handle_place_option_order(data):
    try:
        premium, error = options_market.place_option_order(
            int(data['strike']),
            int(data['expiry_id']),
            data['option_type'],
            int(data['qty']),
            data['direction'],
        )
        if error:
            return {'status': 'error', 'message': error}
        cash_delta = -premium if data['direction'] == 'buy' else +premium
        user_account.account.addPosition("CASH", cash_delta)
        return {'status': 'success', 'premium': premium}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def emit_updates():
    """Emit price and order book updates through websocket"""
    account_tick = 0
    options_tick = 0
    USER_INITIAL_CASH = 1_000_000

    while True:
        for asset in assets:
            # Emit price updates
            if asset in price_history and price_history[asset]:
                last_price = price_history[asset].getLastPrice()
                socketio.emit('price_update', {
                    'asset': asset,
                    'price': float(last_price),
                    'open': float(price_history[asset].initial_price),
                })

                socketio.emit('volume_update', {
                    'asset': asset,
                    'volume': int(markets[asset].net_volume)
                })

            # Emit order book updates
            emit_orderbook_update(asset)

        # Emit user account update every ~1 second (every 10 ticks)
        account_tick += 1
        if account_tick >= 10:
            account_tick = 0
            try:
                total_value = float(user_account.account.getValue())
                cash        = float(user_account.account.getCash())
                positions   = {
                    k: float(v)
                    for k, v in user_account.account.getPositions().items()
                    if k != "CASH" and abs(float(v)) > 0.001
                }
                socketio.emit('account_update', {
                    'cash':        cash,
                    'total_value': total_value,
                    'pnl':         total_value - USER_INITIAL_CASH,
                    'positions':   positions,
                })
            except Exception:
                pass

        # ── Options chain update every ~2 s (every 20 ticks) ─────────────
        options_tick += 1
        if options_tick >= 20:
            options_tick = 0
            try:
                def _settlement_cb(net_cash, label, spy_price):
                    if net_cash != 0:
                        user_account.account.addPosition("CASH", net_cash)
                    socketio.emit('option_settled', {
                        'label':     label,
                        'net_cash':  round(net_cash, 2),
                        'spy_price': round(spy_price, 2),
                    })
                options_market.settle_expired(_settlement_cb)
                socketio.emit('options_update', {
                    'chain':     options_market.get_chain(),
                    'positions': options_market.get_user_positions(),
                })
            except Exception:
                pass

        time.sleep(0.1)  # Update every 0.1 seconds

def start_server(host='0.0.0.0', port=5000):
    """Start the price server"""
    # Start the update thread
    update_thread = threading.Thread(target=emit_updates)
    update_thread.daemon = True
    update_thread.start()
    
    # Start the Flask-SocketIO server
    socketio.run(app, host=host, port=port)

if __name__ == '__main__':
    start_server() 