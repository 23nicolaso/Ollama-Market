from flask import Flask, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
import threading
import time
from market_simulator.utils.market_utils import price_history, assets, markets
from market_simulator.utils.news_generator import generate_news_thread

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

def emit_trade_update(agent, asset, side, quantity, price):
    """Emit a trade update through websocket"""
    socketio.emit('trade_update', {
        'agent': agent,
        'asset': asset,
        'side': side,
        'quantity': quantity,
        'price': price,
        'timestamp': time.time()
    })

def emit_orderbook_update(asset):
    """Emit order book update through websocket"""
    if asset in markets:
        market = markets[asset]
        bids = {str(price): level.getQuantity() for price, level in market.getBids().items()}
        asks = {str(price): level.getQuantity() for price, level in market.getAsks().items()}
        
        socketio.emit('orderbook_update', {
            'asset': asset,
            'bids': bids,
            'asks': asks,
            'timestamp': time.time()
        })

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
    if asset in price_history:
        return jsonify({
            'asset': asset,
            'prices': price_history[asset]
        })
    return jsonify({'error': 'Asset not found'}), 404

def emit_updates():
    """Emit price and order book updates through websocket"""
    while True:
        for asset in assets:
            # Emit price updates
            if asset in price_history and price_history[asset]:
                socketio.emit('price_update', {
                    'asset': asset,
                    'price': price_history[asset][-1],
                    'history': price_history[asset][-100:]  # Send last 100 prices
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