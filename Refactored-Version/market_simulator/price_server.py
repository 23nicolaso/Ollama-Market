from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
import time
from market_simulator.utils.db_utils import db_manager as db
from market_simulator.utils.market_utils import price_history, assets, markets, get_price_history
from market_simulator.utils.news_generator import generate_news_thread
from market_simulator.agents.user_trader import UserTrader
from datetime import datetime

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
user_account = UserTrader("USER_001", 1000000)

def emit_trade_update(agent, asset, side, quantity):
    """Emit a trade update through websocket"""
    socketio.emit('trade_update', {
        'agent': agent,
        'asset': asset,
        'side': side,
        'quantity': quantity
    })

def emit_orderbook_update(asset):
    """Emit order book update through websocket"""
    if asset in markets:
        market = markets[asset]
        try:
            bidsandasks = market.getBidAskPairs()
            socketio.emit('orderbook_update', {
                'asset': asset,
                'bids': bidsandasks[0],
                'asks': bidsandasks[1],
                'timestamp': time.time()
            })
        except:
            bidsandasks = {}

def generate_periodic_news():
    """Generate news every 5 minutes"""
    while True:
        generate_news_thread()
        time.sleep(300)  # Sleep for 5 minutes

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

def emit_updates():
    """Emit price and order book updates through websocket"""
    while True:
        for asset in assets:
            # Emit price updates
            if asset in price_history and price_history[asset]:
                socketio.emit('price_update', {
                    'asset': asset,
                    'price': float(price_history[asset].getLastPrice()),
                    'history': price_history[asset].get().tolist()
                })
            
            # Emit order book updates
            emit_orderbook_update(asset)
            
        time.sleep(0.1)  # Update every 0.1 seconds

def start_server(host='0.0.0.0', port=5000):
    """Start the price server"""
    # Start the update thread
    update_thread = threading.Thread(target=emit_updates)
    update_thread.daemon = True
    update_thread.start()
    
    # Start the news generation thread
    news_thread = threading.Thread(target=generate_periodic_news)
    news_thread.daemon = True
    news_thread.start()
    
    # Start the Flask-SocketIO server
    socketio.run(app, host=host, port=port)

if __name__ == '__main__':
    start_server() 